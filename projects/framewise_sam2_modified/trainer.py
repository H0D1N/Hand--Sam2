"""SAM2Modified 单帧左右手训练与验证循环。"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
import argparse
from typing import Any
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from pathlib import Path

from .losses import dual_hand_loss, iou_target_from_logits, object_targets_from_masks
from .utils import (
    make_dual_hand_tensorboard_image,
    save_dual_hand_visualization,
    upsample_logits,
)

LOSS_NAMES = ("bce", "dice", "iou", "object_score")


def gradient_l2_norm(model: torch.nn.Module) -> float:
    grad_norms = [
        parameter.grad.detach().float().norm(2)
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]

    if not grad_norms:
        return 0.0

    return torch.stack(grad_norms).norm(2).item()


def log_tensorboard_training_step(
        writer: Any,
        outputs: dict,
        left_masks: torch.Tensor,
        right_masks: torch.Tensor,
        total_loss: torch.Tensor,
        loss_details: dict,
        optimizer: torch.optim.Optimizer,
        global_step: int,
) -> None:
    writer.add_scalar("train/loss", total_loss.detach().item(), global_step)
    writer.add_scalar(
        "optimizer/learning_rate",
        optimizer.param_groups[0]["lr"],
        global_step,
    )

    foreground_ious = []

    for hand_name, target_masks in (
        ("left", left_masks),
        ("right", right_masks),
    ):
        for loss_name, loss_value in loss_details[hand_name].items():
            writer.add_scalar(
                f"train_loss/{hand_name}_{loss_name}",
                loss_value.detach().item(),
                global_step,
            )

        score_logits = outputs[hand_name]["object_score_logits"].detach()
        object_targets = object_targets_from_masks(target_masks)

        writer.add_scalar(
            f"train_object/{hand_name}_logit_mean",
            score_logits.mean().item(),
            global_step,
        )
        writer.add_scalar(
            f"train_object/{hand_name}_predicted_present_rate",
            (score_logits > 0).float().mean().item(),
            global_step,
        )
        writer.add_scalar(
            f"train_object/{hand_name}_gt_present_rate",
            object_targets.float().mean().item(),
            global_step,
        )

        if object_targets.any().item():
            batch_ious = iou_target_from_logits(
                outputs[hand_name]["high_res_masks"].detach(),
                target_masks,
            )[object_targets]
            mean_iou = batch_ious.mean()
            foreground_ious.append(batch_ious.flatten())

            writer.add_scalar(
                f"train_iou/{hand_name}",
                mean_iou.item(),
                global_step,
            )

    if foreground_ious:
        writer.add_scalar(
            "train_iou/mean",
            torch.cat(foreground_ious).mean().item(),
            global_step,
        )


@torch.inference_mode()
def log_tensorboard_probe(
        model: torch.nn.Module,
        probe: dict,
        writer: Any,
        device: torch.device,
        args: argparse.Namespace,
        global_step: int,
) -> None:
    """对固定 val probe 推理并记录三栏图和 IoU。"""
    was_training = model.training
    model.eval()
    dataset_iou_sums: dict[str, float] = {}
    dataset_hand_counts: dict[str, int] = {}
    batch_size = max(args.val_batch_size, 1)
    amp_enabled = device.type == "cuda" and args.amp

    try:
        for start in range(0, probe["image"].size(0), batch_size):
            end = min(start + batch_size, probe["image"].size(0))
            images = probe["image"][start:end].to(device, non_blocking=True)
            left_masks = probe["left_mask"][start:end].to(
                device,
                non_blocking=True,
            )
            right_masks = probe["right_mask"][start:end].to(
                device,
                non_blocking=True,
            )

            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            context = (
                torch.amp.autocast(device_type="cuda", enabled=True)
                if amp_enabled else nullcontext()
            )

            with context:
                outputs = model.forward_single_image(
                    images=images,
                    mask_inputs=None,
                    multimask_output=args.multimask_output,
                )

            for local_index in range(end - start):
                probe_index = start + local_index
                dataset_name = probe["dataset_name"][probe_index]
                label = probe["label"][probe_index]

                left_prediction = (
                    outputs["left"]["high_res_masks"][local_index] > 0
                )
                right_prediction = (
                    outputs["right"]["high_res_masks"][local_index] > 0
                )

                summary_image = make_dual_hand_tensorboard_image(
                    normalized_image=images[local_index],
                    left_pred_mask=left_prediction,
                    right_pred_mask=right_prediction,
                    left_gt_mask=left_masks[local_index],
                    right_gt_mask=right_masks[local_index],
                    output_size=args.tensorboard_probe_image_size,
                )
                writer.add_image(
                    f"probe/{label}",
                    summary_image,
                    global_step,
                )

                for hand_name, target_mask in (
                    ("left", left_masks[local_index:local_index + 1]),
                    ("right", right_masks[local_index:local_index + 1]),
                ):
                    if not target_mask.any().item():
                        continue

                    probe_iou = iou_target_from_logits(
                        outputs[hand_name]["high_res_masks"][
                            local_index:local_index + 1
                        ],
                        target_mask,
                    ).item()
                    writer.add_scalar(
                        f"probe_iou/{label}/{hand_name}",
                        probe_iou,
                        global_step,
                    )
                    dataset_iou_sums[dataset_name] = (
                        dataset_iou_sums.get(dataset_name, 0.0)
                        + probe_iou
                    )
                    dataset_hand_counts[dataset_name] = (
                        dataset_hand_counts.get(dataset_name, 0) + 1
                    )
    finally:
        model.train(was_training)

    for dataset_name, iou_sum in dataset_iou_sums.items():
        writer.add_scalar(
            f"probe_iou_by_dataset/{dataset_name}",
            iou_sum / dataset_hand_counts[dataset_name],
            global_step,
        )

    writer.flush()

def run_training_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device,
        args: argparse.Namespace,
        epoch: int,
        tensorboard_writer: Any = None,
        tensorboard_probe: dict | None = None,
) -> dict[str, float]:
    model.train()

    loss_sums = {"loss": 0.0}
    total_samples = 0
    amp_enabled = device.type == "cuda" and args.amp
    num_steps = len(loader)
    grad_accum_steps = args.grad_accum_steps
    updates_per_epoch = math.ceil(num_steps / grad_accum_steps)

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        if args.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        context = (torch.amp.autocast(device_type="cuda", enabled=True)if amp_enabled else nullcontext())

        with context:
            outputs = model.forward_single_image(
                images=images,
                mask_inputs=None,
                multimask_output=args.multimask_output,
            )
            total_loss, loss_details = dual_hand_loss(
                model_output=outputs,
                left_masks=left_masks,
                right_masks=right_masks,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                iou_weight=args.iou_weight,
                object_score_weight=args.object_score_weight,
            )

        if not torch.isfinite(total_loss).item():
            raise FloatingPointError(f"Epoch {epoch + 1}, step {step}: "f"loss={total_loss.detach().item()}")

        global_step = epoch * num_steps + step
        should_log_tensorboard = (
            tensorboard_writer is not None
            and (
                global_step == 1
                or global_step % args.tensorboard_log_interval == 0
            )
        )

        if should_log_tensorboard:
            log_tensorboard_training_step(
                writer=tensorboard_writer,
                outputs=outputs,
                left_masks=left_masks,
                right_masks=right_masks,
                total_loss=total_loss,
                loss_details=loss_details,
                optimizer=optimizer,
                global_step=global_step,
            )

        # 完整累积组按 grad_accum_steps 平均
        # 最后不足一组时按实际小批次数平均，避免最后一次参数更新的梯度偏小。
        
            # 当前梯度累积组从哪个 step 开始，step 从 1 开始计数。
        group_start = ((step - 1) // grad_accum_steps) * grad_accum_steps
        current_group_size = min(grad_accum_steps, num_steps - group_start)

        loss_for_backward = total_loss / current_group_size

        scaler.scale(loss_for_backward).backward()

        should_update = (
            step % args.grad_accum_steps == 0
            or step == num_steps
        )
        should_log_probe = False

        if should_update:
            if tensorboard_writer is not None:
                scaler.unscale_(optimizer)
                grad_norm = gradient_l2_norm(model)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            update_step = (
                epoch * updates_per_epoch
                + math.ceil(step / grad_accum_steps)
            )

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "optimizer/gradient_l2_norm",
                    grad_norm,
                    global_step,
                )

            should_log_probe = (
                tensorboard_writer is not None
                and tensorboard_probe is not None
                and args.tensorboard_probe_interval > 0
                and update_step % args.tensorboard_probe_interval == 0
            )

        batch_size = images.size(0)
        total_samples += batch_size

        loss_sums["loss"] += (total_loss.detach().item()* batch_size)

        for hand_name, hand_details in loss_details.items():
            for loss_name, loss_value in hand_details.items():
                key = f"{hand_name}_{loss_name}_loss"
                loss_sums[key] = (loss_sums.get(key, 0.0)+ loss_value.detach().item() * batch_size)

        if (
            step % args.log_interval == 0
            or step == len(loader)
        ):
            mean_details = {
                loss_name: (
                    loss_details["left"][loss_name]
                    + loss_details["right"][loss_name]
                ) / 2.0
                for loss_name in LOSS_NAMES
            }

            logging.info(
                "Epoch %d | step %d/%d | "
                "loss=%.4f | bce=%.4f | dice=%.4f | "
                "iou_loss=%.4f | object_score_loss=%.4f",
                epoch + 1,
                step,
                len(loader),
                total_loss.detach().item(),
                mean_details["bce"].detach().item(),
                mean_details["dice"].detach().item(),
                mean_details["iou"].detach().item(),
                mean_details["object_score"].detach().item(),
            )

        if should_log_probe:
            del outputs, total_loss, loss_details, loss_for_backward
            del images, left_masks, right_masks

            log_tensorboard_probe(
                model=model,
                probe=tensorboard_probe,
                writer=tensorboard_writer,
                device=device,
                args=args,
                global_step=global_step,
            )

    if total_samples == 0:
        raise ValueError("训练 DataLoader 中没有样本")

    return {
        name: value / total_samples
        for name, value in loss_sums.items()
    }

@torch.inference_mode()
def run_validation_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        device: torch.device,
        epoch: int,
        args: argparse.Namespace,
        visualization_fn=save_dual_hand_visualization,
        point_prompt_fn=None,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0
    total_foreground_hands = 0

    object_tp = 0
    object_tn = 0
    object_fp = 0
    object_fn = 0

    vis_dir = (
        Path(args.output_dir)
        / "visualizations"
        / f"val_epoch_{epoch + 1}"
    )
    num_vis_saved_by_dataset: dict[str, int] = {}
    max_vis_per_dataset = 100

    if not args.skip_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        if args.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        left_point_inputs = (
            point_prompt_fn(left_masks)
            if point_prompt_fn is not None
            else None
        )

        right_point_inputs = (
            point_prompt_fn(right_masks)
            if point_prompt_fn is not None
            else None
        )

        outputs = model.forward_single_image(
            images=images,
            left_point_inputs=left_point_inputs,
            right_point_inputs=right_point_inputs,
            mask_inputs=None,
            multimask_output=args.multimask_output,
        )

        loss, _ = dual_hand_loss(
            model_output=outputs,
            left_masks=left_masks,
            right_masks=right_masks,
            bce_weight=args.bce_weight,
            dice_weight=args.dice_weight,
            iou_weight=args.iou_weight,
            object_score_weight=args.object_score_weight,
        )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        gt_present = torch.cat((
            object_targets_from_masks(left_masks),
            object_targets_from_masks(right_masks),
        ))
        pred_present = torch.cat((
            outputs["left"]["object_score_logits"].reshape(-1) > 0,
            outputs["right"]["object_score_logits"].reshape(-1) > 0,
        ))

        object_tp += (pred_present & gt_present).sum().item()
        object_tn += (~pred_present & ~gt_present).sum().item()
        object_fp += (pred_present & ~gt_present).sum().item()
        object_fn += (~pred_present & gt_present).sum().item()

        # 可视化与可视化准备；具体计算左右手的iou& dice
        for sample_index in range(batch_size):
            original_left_mask = batch["original_left_mask"][sample_index].unsqueeze(0).to(device)
            original_right_mask = batch["original_right_mask"][sample_index].unsqueeze(0).to(device)
            original_size = original_left_mask.shape[-2:]

            left_logits = upsample_logits(
                outputs["left"]["high_res_masks"][sample_index:sample_index + 1], size=original_size)
            right_logits = upsample_logits(
                outputs["right"]["high_res_masks"][sample_index:sample_index + 1], size=original_size)

            left_iou = iou_target_from_logits(left_logits, original_left_mask)
            right_iou = iou_target_from_logits(right_logits, original_right_mask)

            left_dice = 2.0 * left_iou / (1.0 + left_iou)
            right_dice = 2.0 * right_iou / (1.0 + right_iou)

            if original_left_mask.any().item():
                total_iou += left_iou.item()
                total_dice += left_dice.item()
                total_foreground_hands += 1

            if original_right_mask.any().item():
                total_iou += right_iou.item()
                total_dice += right_dice.item()
                total_foreground_hands += 1

            dataset_name = batch["dataset_name"][sample_index]
            num_vis_saved = num_vis_saved_by_dataset.get(dataset_name, 0)

            if (
                not args.skip_visualizations
                and num_vis_saved < max_vis_per_dataset
            ):
                visualization_fn(
                    original_image=batch["original_image"][sample_index],
                    left_pred_mask=(left_logits > 0).float(),
                    right_pred_mask=(right_logits > 0).float(),
                    left_gt_mask=original_left_mask,
                    right_gt_mask=original_right_mask,
                    save_path=vis_dir / f"{batch['sample_id'][sample_index]}.png",
                )
                num_vis_saved_by_dataset[dataset_name] = num_vis_saved + 1

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            logging.info(
                "Val Epoch %d | step %d/%d | "
                "loss=%.4f | iou=%.4f | dice=%.4f",
                epoch + 1,
                step,
                len(loader),
                total_loss / total_samples,
                total_iou / total_foreground_hands,
                total_dice / total_foreground_hands,
            )

    object_total = object_tp + object_tn + object_fp + object_fn
    object_accuracy = (object_tp + object_tn) / max(object_total, 1)
    object_precision = object_tp / max(object_tp + object_fp, 1)
    object_recall = object_tp / max(object_tp + object_fn, 1)
    object_f1 = (
        2.0 * object_precision * object_recall
        / max(object_precision + object_recall, 1e-8)
    )

    return {
        "loss": total_loss / total_samples,
        "iou": total_iou / total_foreground_hands,
        "dice": total_dice / total_foreground_hands,
        "object_accuracy": object_accuracy,
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
    }
