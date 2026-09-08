"""多视角双手 Memory 模型的训练与验证循环。"""

import argparse
import logging
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader

from projects.framewise_sam2_modified.losses import iou_target_from_logits, object_targets_from_masks
from projects.framewise_sam2_modified.utils import upsample_logits
from .losses import LOSS_NAMES
from .visualization import save_multiview_visualization


def run_training_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    prompt_request,
    tensorboard_writer=None,
    visualization_fn=save_multiview_visualization,
) -> dict[str, float]:
    """训练一个 epoch。"""

    model.train()
    optimizer.zero_grad(set_to_none=True)

    loss_sums = {"loss": 0.0}
    total_clips = 0
    num_steps = len(loader)
    amp_enabled = device.type == "cuda" and args.amp
    max_vis_per_dataset = getattr(args, "max_vis_per_dataset", 100)
    num_vis_saved_per_dataset = {}
    num_sequences_per_dataset = {}

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        context = torch.amp.autocast(device_type="cuda") if amp_enabled else nullcontext()

        with context:
            frame_outputs = model(
                images=images,
                left_masks=left_masks,
                right_masks=right_masks,
                prompt_request=prompt_request,
            )
            total_loss, loss_details = loss_fn(
                frame_outputs,
                left_masks,
                right_masks,
            )

        if not torch.isfinite(total_loss).item():
            raise FloatingPointError(
                f"Epoch {epoch + 1}, step {step}: "
                f"loss={total_loss.detach().item()}"
            )

        # 最后一个梯度累积组可能不足 grad_accum_steps。
        group_start = ((step - 1) // args.grad_accum_steps) * args.grad_accum_steps
        group_size = min(args.grad_accum_steps, num_steps - group_start)
        scaler.scale(total_loss / group_size).backward()

        if step % args.grad_accum_steps == 0 or step == num_steps:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        else:
            grad_norm = None

        batch_size = images.size(0)
        total_clips += batch_size
        loss_sums["loss"] += total_loss.detach().item() * batch_size

        for hand, hand_losses in loss_details.items():
            for name, value in hand_losses.items():
                key = f"{hand}_{name}"
                loss_sums[key] = loss_sums.get(key, 0.0) + value.detach().item() * batch_size

        global_step = epoch * num_steps + step

        if (
            tensorboard_writer is not None
            and (global_step == 1 or global_step % args.tensorboard_log_interval == 0)
        ):
            tensorboard_writer.add_scalar("train/loss", total_loss.detach().item(), global_step)
            tensorboard_writer.add_scalar("optimizer/learning_rate", optimizer.param_groups[0]["lr"], global_step)

            if grad_norm is not None:
                tensorboard_writer.add_scalar("optimizer/gradient_l2_norm", grad_norm.detach().item(), global_step)

            for hand, hand_losses in loss_details.items():
                for name, value in hand_losses.items():
                    tensorboard_writer.add_scalar(
                        f"train_loss/{hand}_{name}",
                        value.detach().item(),
                        global_step,
                    )

        if not args.skip_visualizations:
            for clip_idx, dataset_name in enumerate(batch["dataset_name"]):
                step_counts = [output["left"]["multistep_pred_masks_high_res"].size(1) for output in frame_outputs]
                num_images = sum(step_counts)
                num_saved = num_vis_saved_per_dataset.get(dataset_name, 0)
                if max(step_counts) == 1 or num_saved + num_images > max_vis_per_dataset:
                    continue
                sequence_idx = num_sequences_per_dataset.get(dataset_name, 0)
                visualization_fn(
                    batch=batch,
                    frame_outputs=frame_outputs,
                    clip_idx=clip_idx,
                    sequence_idx=sequence_idx,
                    split="train",
                    epoch=epoch,
                    args=args,
                )
                num_vis_saved_per_dataset[dataset_name] = num_saved + num_images
                num_sequences_per_dataset[dataset_name] = sequence_idx + 1

        if step % args.log_interval == 0 or step == num_steps:
            mean_losses = {
                name: (loss_details["left"][name] + loss_details["right"][name]) / 2.0
                for name in LOSS_NAMES
            }
            logging.info(
                "Epoch %d | step %d/%d | loss=%.4f | "
                "mask_focal=%.4f | dice=%.4f | iou_loss=%.4f | class_loss=%.4f",
                epoch + 1,
                step,
                num_steps,
                total_loss.detach().item(),
                mean_losses["loss_mask"].detach().item(),
                mean_losses["loss_dice"].detach().item(),
                mean_losses["loss_iou"].detach().item(),
                mean_losses["loss_class"].detach().item(),
            )

    if total_clips == 0:
        raise ValueError("训练 DataLoader 中没有 Clip")

    return {
        name: value / total_clips
        for name, value in loss_sums.items()
    }


@torch.inference_mode()
def run_validation_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    prompt_request,
    visualization_fn=save_multiview_visualization,
    metric_start_frame: int = 0,
) -> dict[str, dict[str, float | None]]:
    """验证一个 epoch。"""

    model.eval()

    stats = {
        name: {
            "loss_sum": 0.0,
            "clips": 0,
            "iou_sum": 0.0,
            "dice_sum": 0.0,
            "foreground_hands": 0,
            "object_tp": 0,
            "object_tn": 0,
            "object_fp": 0,
            "object_fn": 0,
        }
        for name in ("overall", "multiserver", "dexycb")
    }
    max_vis_per_dataset = getattr(args, "max_vis_per_dataset", 100)
    num_vis_saved_per_dataset = {}
    num_sequences_per_dataset = {}

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        frame_outputs = model(
            images=images,
            left_masks=left_masks,
            right_masks=right_masks,
            prompt_request=prompt_request,
        )
        loss, _ = loss_fn(frame_outputs, left_masks, right_masks)

        batch_size, num_views, num_frames = images.shape[:3]
        dataset_groups = [
            "dexycb" if name.lower() == "dexycb" else "multiserver"
            for name in batch["dataset_name"]
        ]

        stats["overall"]["loss_sum"] += loss.item() * batch_size
        stats["overall"]["clips"] += batch_size

        # 分别统计 MultiServer 和 DexYCB loss。
        for group in ("multiserver", "dexycb"):
            clip_indices = [
                index
                for index, current_group in enumerate(dataset_groups)
                if current_group == group
            ]
            if not clip_indices:
                continue

            group_loss = (
                loss
                if len(clip_indices) == batch_size
                else loss_fn(
                    frame_outputs,
                    left_masks,
                    right_masks,
                    sample_indices=clip_indices,
                )[0]
            )
            stats[group]["loss_sum"] += group_loss.item() * len(clip_indices)
            stats[group]["clips"] += len(clip_indices)

        # frame_outputs 是长度为 T 的 list，每帧预测 batch 为 B*V。
        for frame_idx, outputs in enumerate(frame_outputs):
            if frame_idx < metric_start_frame:
                continue

            left_gt = left_masks[:, :, frame_idx].flatten(0, 1)
            right_gt = right_masks[:, :, frame_idx].flatten(0, 1)

            left_gt_present = object_targets_from_masks(left_gt)
            right_gt_present = object_targets_from_masks(right_gt)
            left_pred_present = outputs["left"]["multistep_object_score_logits"][-1].reshape(-1) > 0
            right_pred_present = outputs["right"]["multistep_object_score_logits"][-1].reshape(-1) > 0

            for clip_idx in range(batch_size):
                group = dataset_groups[clip_idx]

                for view_idx in range(num_views):
                    flat_idx = clip_idx * num_views + view_idx
                    current_stats = (stats["overall"], stats[group])

                    for pred_present, gt_present in (
                        (left_pred_present[flat_idx], left_gt_present[flat_idx]),
                        (right_pred_present[flat_idx], right_gt_present[flat_idx]),
                    ):
                        for values in current_stats:
                            values["object_tp"] += int((pred_present & gt_present).item())
                            values["object_tn"] += int((~pred_present & ~gt_present).item())
                            values["object_fp"] += int((pred_present & ~gt_present).item())
                            values["object_fn"] += int((~pred_present & gt_present).item())

                    original_left = batch["original_left_mask"][clip_idx][view_idx][frame_idx].unsqueeze(0).to(device)
                    original_right = batch["original_right_mask"][clip_idx][view_idx][frame_idx].unsqueeze(0).to(device)
                    original_size = original_left.shape[-2:]

                    left_logits = upsample_logits(
                        outputs["left"]["pred_masks_high_res"][flat_idx:flat_idx + 1],
                        original_size,
                    )
                    right_logits = upsample_logits(
                        outputs["right"]["pred_masks_high_res"][flat_idx:flat_idx + 1],
                        original_size,
                    )

                    left_iou = iou_target_from_logits(left_logits, original_left).item()
                    right_iou = iou_target_from_logits(right_logits, original_right).item()
                    left_dice = 2.0 * left_iou / (1.0 + left_iou)
                    right_dice = 2.0 * right_iou / (1.0 + right_iou)

                    if original_left.any().item():
                        for values in current_stats:
                            values["iou_sum"] += left_iou
                            values["dice_sum"] += left_dice
                            values["foreground_hands"] += 1

                    if original_right.any().item():
                        for values in current_stats:
                            values["iou_sum"] += right_iou
                            values["dice_sum"] += right_dice
                            values["foreground_hands"] += 1

        if not args.skip_visualizations:
            for clip_idx, dataset_name in enumerate(batch["dataset_name"]):
                num_saved = num_vis_saved_per_dataset.get(dataset_name, 0)
                if num_saved + num_frames > max_vis_per_dataset:
                    continue
                sequence_idx = num_sequences_per_dataset.get(dataset_name, 0)
                visualization_fn(
                    batch=batch,
                    frame_outputs=frame_outputs,
                    clip_idx=clip_idx,
                    sequence_idx=sequence_idx,
                    split="val",
                    epoch=epoch,
                    args=args,
                )
                num_vis_saved_per_dataset[dataset_name] = num_saved + num_frames
                num_sequences_per_dataset[dataset_name] = sequence_idx + 1

        if step % args.log_interval == 0 or step == len(loader):
            overall = stats["overall"]
            logging.info(
                "Val Epoch %d | step %d/%d | loss=%.4f | iou=%.4f | dice=%.4f",
                epoch + 1,
                step,
                len(loader),
                overall["loss_sum"] / overall["clips"],
                overall["iou_sum"] / max(overall["foreground_hands"], 1),
                overall["dice_sum"] / max(overall["foreground_hands"], 1),
            )

    if stats["overall"]["clips"] == 0:
        raise ValueError("验证 DataLoader 中没有 Clip")

    results = {}

    for name, values in stats.items():
        if values["clips"] == 0:
            results[name] = {
                metric: None
                for metric in (
                    "loss",
                    "iou",
                    "dice",
                    "object_accuracy",
                    "object_precision",
                    "object_recall",
                    "object_f1",
                )
            }
            continue

        object_total = sum(
            values[key]
            for key in ("object_tp", "object_tn", "object_fp", "object_fn")
        )
        precision = values["object_tp"] / max(
            values["object_tp"] + values["object_fp"],
            1,
        )
        recall = values["object_tp"] / max(
            values["object_tp"] + values["object_fn"],
            1,
        )

        results[name] = {
            "loss": values["loss_sum"] / values["clips"],
            "iou": values["iou_sum"] / max(values["foreground_hands"], 1),
            "dice": values["dice_sum"] / max(values["foreground_hands"], 1),
            "object_accuracy": (
                values["object_tp"] + values["object_tn"]
            ) / max(object_total, 1),
            "object_precision": precision,
            "object_recall": recall,
            "object_f1": 2.0 * precision * recall / max(
                precision + recall,
                1e-8,
            ),
        }

    return results
