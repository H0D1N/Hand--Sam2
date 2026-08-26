"""SAM2Modified 单帧左右手训练与验证循环。"""

from __future__ import annotations

import logging
import math
from contextlib import nullcontext
import argparse
from typing import Any
from torch.utils.data import DataLoader
import torch
from pathlib import Path

from .losses import dual_hand_loss, iou_target_from_logits, object_targets_from_masks
from .tensorboard_utils import gradient_l2_norm, log_tensorboard_training_step, log_tensorboard_probe
from .utils import upsample_logits
from .visualization import save_dual_hand_visualization

LOSS_NAMES = ("bce", "dice", "iou", "object_score")



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
        point_prompt_fn=None,
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

            left_points = (
                point_prompt_fn(left_masks)
                if point_prompt_fn is not None
                else None
            )

            right_points = (
                point_prompt_fn(right_masks)
                if point_prompt_fn is not None
                else None
            )

            outputs = model.forward_single_image(
                images=images,
                left_point_inputs=left_points,
                right_point_inputs=right_points,
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
                point_prompt_fn=point_prompt_fn,
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
) -> dict[str, dict[str, float | None]]:
    model.eval()

    stats = {
        name: {
            "loss_sum": 0.0, "samples": 0, "iou_sum": 0.0, "dice_sum": 0.0,
            "foreground_hands": 0, "object_tp": 0, "object_tn": 0, "object_fp": 0, "object_fn": 0,
        }
        for name in ("overall", "multiserver", "dexycb")
    }

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
        dataset_groups = ["dexycb" if name.lower() == "dexycb" else "multiserver" for name in batch["dataset_name"]]
        stats["overall"]["loss_sum"] += loss.item() * batch_size
        stats["overall"]["samples"] += batch_size

        for dataset_group in ("multiserver", "dexycb"):
            sample_indices = [index for index, group in enumerate(dataset_groups) if group == dataset_group]
            if not sample_indices:
                continue
            if len(sample_indices) == batch_size:
                group_loss = loss
            else:
                group_outputs = {
                    hand: {name: value[sample_indices] for name, value in outputs[hand].items()}
                    for hand in ("left", "right")
                }
                group_loss = dual_hand_loss(
                    model_output=group_outputs,
                    left_masks=left_masks[sample_indices], right_masks=right_masks[sample_indices],
                    bce_weight=args.bce_weight, dice_weight=args.dice_weight,
                    iou_weight=args.iou_weight, object_score_weight=args.object_score_weight,
                )[0]
            stats[dataset_group]["loss_sum"] += group_loss.item() * len(sample_indices)
            stats[dataset_group]["samples"] += len(sample_indices)

        gt_present = torch.cat((
            object_targets_from_masks(left_masks),
            object_targets_from_masks(right_masks),
        ))
        pred_present = torch.cat((
            outputs["left"]["object_score_logits"].reshape(-1) > 0,
            outputs["right"]["object_score_logits"].reshape(-1) > 0,
        ))

        for sample_index in range(batch_size):
            current_stats = (stats["overall"], stats[dataset_groups[sample_index]])
            sample_gt_present = gt_present[[sample_index, batch_size + sample_index]]
            sample_pred_present = pred_present[[sample_index, batch_size + sample_index]]
            for values in current_stats:
                values["object_tp"] += (sample_pred_present & sample_gt_present).sum().item()
                values["object_tn"] += (~sample_pred_present & ~sample_gt_present).sum().item()
                values["object_fp"] += (sample_pred_present & ~sample_gt_present).sum().item()
                values["object_fn"] += (~sample_pred_present & sample_gt_present).sum().item()

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
                for values in current_stats:
                    values["iou_sum"] += left_iou.item()
                    values["dice_sum"] += left_dice.item()
                    values["foreground_hands"] += 1

            if original_right_mask.any().item():
                for values in current_stats:
                    values["iou_sum"] += right_iou.item()
                    values["dice_sum"] += right_dice.item()
                    values["foreground_hands"] += 1

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
            overall = stats["overall"]
            logging.info(
                "Val Epoch %d | step %d/%d | "
                "loss=%.4f | iou=%.4f | dice=%.4f",
                epoch + 1,
                step,
                len(loader),
                overall["loss_sum"] / overall["samples"],
                overall["iou_sum"] / max(overall["foreground_hands"], 1),
                overall["dice_sum"] / max(overall["foreground_hands"], 1),
            )

    if stats["overall"]["samples"] == 0:
        raise ValueError("验证 DataLoader 中没有样本")

    result = {}
    metric_names = ("loss", "iou", "dice", "object_accuracy", "object_precision", "object_recall", "object_f1")
    for name, values in stats.items():
        if values["samples"] == 0:
            result[name] = {metric: None for metric in metric_names}
            continue
        object_total = sum(values[key] for key in ("object_tp", "object_tn", "object_fp", "object_fn"))
        object_precision = values["object_tp"] / max(values["object_tp"] + values["object_fp"], 1)
        object_recall = values["object_tp"] / max(values["object_tp"] + values["object_fn"], 1)
        result[name] = {
            "loss": values["loss_sum"] / values["samples"],
            "iou": values["iou_sum"] / max(values["foreground_hands"], 1),
            "dice": values["dice_sum"] / max(values["foreground_hands"], 1),
            "object_accuracy": (values["object_tp"] + values["object_tn"]) / max(object_total, 1),
            "object_precision": object_precision,
            "object_recall": object_recall,
            "object_f1": 2 * object_precision * object_recall / max(object_precision + object_recall, 1e-8),
        }
    return result
