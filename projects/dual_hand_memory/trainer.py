"""双手 Memory 模型的序列训练与验证循环。"""

import argparse
import logging
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from .losses import LOSS_NAMES
from projects.framewise_sam2_modified.losses import iou_target_from_logits, object_targets_from_masks
from projects.framewise_sam2_modified.utils import upsample_logits
from projects.framewise_sam2_modified.visualization import save_dual_hand_memory_comparison_visualization


HIGH_CLASS_LOSS_THRESHOLD = 50.0


def _log_high_class_loss_batch(
    batch,
    frame_outputs,
    left_masks,
    right_masks,
    class_loss,
    epoch,
    step,
):
    """记录稳定复现高 class loss 所需的样本和逐轮预测。"""

    logging.warning(
        "High class loss | epoch=%d | step=%d | class_loss=%.4f",
        epoch + 1,
        step,
        class_loss,
    )

    for sample_idx in range(left_masks.size(0)):
        metadata = {
            key: batch[key][sample_idx]
            for key in ("dataset_name", "stream_id", "sample_id", "image_path")
            if key in batch
        }
        logging.warning("High class loss sample | %s", metadata)

        for hand, masks in (
            ("left", left_masks),
            ("right", right_masks),
        ):
            gt_present = (
                masks[sample_idx]
                .flatten(1)
                .any(dim=1)
                .detach()
                .cpu()
                .tolist()
            )
            object_scores = [
                [
                    score[sample_idx].detach().float().item()
                    for score in output[hand][
                        "multistep_object_score_logits"
                    ]
                ]
                for output in frame_outputs
            ]
            point_labels = [
                [
                    None
                    if point_input is None
                    else point_input["point_labels"][sample_idx]
                    .detach()
                    .cpu()
                    .tolist()
                    for point_input in output[hand].get(
                        "multistep_point_inputs",
                        [],
                    )
                ]
                for output in frame_outputs
            ]
            logging.warning(
                "High class loss %s | gt_present=%s | "
                "object_scores=%s | point_labels=%s",
                hand,
                gt_present,
                object_scores,
                point_labels,
            )

def run_training_epoch(
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    tensorboard_writer=None,
) -> dict[str, float]:
    model.train()

    loss_sums = {"loss": 0.0}
    total_clips = 0
    num_steps = len(loader)
    grad_accum_steps = args.grad_accum_steps
    amp_enabled = device.type == "cuda" and args.amp

    optimizer.zero_grad(set_to_none=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        context = (
            torch.amp.autocast(device_type="cuda", enabled=True)
            if amp_enabled else nullcontext()
        )

        with context:
            frame_outputs = model(
                images=images,
                left_masks=left_masks,
                right_masks=right_masks,
                prompt_mode=args.prompt_mode,
            )

            total_loss, loss_details = loss_fn(
                frame_outputs=frame_outputs,
                left_masks=left_masks,
                right_masks=right_masks,
            )

        if not torch.isfinite(total_loss).item():
            raise FloatingPointError(f"Epoch {epoch + 1}, step {step}: "f"loss={total_loss.detach().item()}")

        mean_class_loss = (
            loss_details["left"]["loss_class"]
            + loss_details["right"]["loss_class"]
        ) / 2.0
        if args.debug_high_class_loss and mean_class_loss.detach().item() >= HIGH_CLASS_LOSS_THRESHOLD:
            _log_high_class_loss_batch(
                batch=batch,
                frame_outputs=frame_outputs,
                left_masks=left_masks,
                right_masks=right_masks,
                class_loss=mean_class_loss.detach().item(),
                epoch=epoch,
                step=step,
            )
        

        # TensorBoard：记录当前训练step的Loss和学习率
        global_step = epoch * num_steps + step
        should_log_tensorboard = (
            tensorboard_writer is not None
            and (
                global_step == 1
                or global_step % args.tensorboard_log_interval == 0
            )
        )

        if should_log_tensorboard:
            tensorboard_writer.add_scalar(
                "train/loss",
                total_loss.detach().item(),
                global_step,
            )
            tensorboard_writer.add_scalar(
                "optimizer/learning_rate",
                optimizer.param_groups[0]["lr"],
                global_step,
            )

            for hand in ("left", "right"):
                for name, value in loss_details[hand].items():
                    tensorboard_writer.add_scalar(
                        f"train_loss/{hand}_{name}",
                        value.detach().item(),
                        global_step,
                    )


        # 完整累积组按 grad_accum_steps 平均
        # 最后不足一组时按实际小批次数平均，避免最后一次参数更新的梯度偏小。
        
            # 当前梯度累积组从哪个 step 开始，step 从 1 开始计数。
        group_start = ((step - 1) // grad_accum_steps) * grad_accum_steps
        current_group_size = min(grad_accum_steps, num_steps - group_start)

        loss_for_backward = total_loss / current_group_size

        scaler.scale(loss_for_backward).backward()

        should_update = step % grad_accum_steps == 0 or step == num_steps

        if should_update:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                (
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ),
                max_norm=args.max_grad_norm,
            )

            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar(
                    "optimizer/gradient_l2_norm",
                    grad_norm.detach().item(),
                    global_step,
                )

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_size = images.size(0)
        total_clips += batch_size
        loss_sums["loss"] += total_loss.detach().item() * batch_size

        for hand, hand_details in loss_details.items():
            for name, value in hand_details.items():
                key = f"{hand}_{name}"
                loss_sums[key] = loss_sums.get(key, 0.0) + value.detach().item() * batch_size

        if step % args.log_interval == 0 or step == num_steps:
            mean_details = {
                name: (
                    loss_details["left"][name]
                    + loss_details["right"][name]
                ) / 2.0
                for name in LOSS_NAMES
            }

            logging.info(
                "Epoch %d | step %d/%d | loss=%.4f | "
                "mask_focal=%.4f | dice=%.4f | iou_loss=%.4f | "
                "class_loss=%.4f",
                epoch + 1,
                step,
                num_steps,
                total_loss.detach().item(),
                mean_details["loss_mask"].detach().item(),
                mean_details["loss_dice"].detach().item(),
                mean_details["loss_iou"].detach().item(),
                mean_details["loss_class"].detach().item(),
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
    visualization_fn=save_dual_hand_memory_comparison_visualization,
) -> dict[str, dict[str, float | None]]:
    model.eval()

    stats = {
        name: {
            "loss_sum": 0.0, "clips": 0, "iou_sum": 0.0, "dice_sum": 0.0,
            "foreground_hands": 0, "object_tp": 0, "object_tn": 0, "object_fp": 0, "object_fn": 0,
        }
        for name in ("overall", "multiserver", "dexycb")
    }

    max_vis_per_dataset = 100
    vis_dir = Path(args.output_dir) / "visualizations" / f"val_epoch_{epoch + 1}"
    num_vis_saved_by_dataset = {}

    if not args.skip_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        frame_outputs = model(
            images=images,
            left_masks=left_masks,
            right_masks=right_masks,
            prompt_mode=args.prompt_mode,
        )
        loss, _ = loss_fn(frame_outputs, left_masks, right_masks,)

        batch_size = images.size(0)
        num_frames = images.size(1)
        dataset_groups = ["dexycb" if name.lower() == "dexycb" else "multiserver" for name in batch["dataset_name"]]
        stats["overall"]["loss_sum"] += loss.item() * batch_size
        stats["overall"]["clips"] += batch_size
        for dataset_group in ("multiserver", "dexycb"):
            sample_indices = [index for index, group in enumerate(dataset_groups) if group == dataset_group]
            if not sample_indices:
                continue
            group_loss = loss if len(sample_indices) == batch_size else loss_fn(frame_outputs, left_masks, right_masks, sample_indices=sample_indices)[0]
            stats[dataset_group]["loss_sum"] += group_loss.item() * len(sample_indices)
            stats[dataset_group]["clips"] += len(sample_indices)
        visual_clip_indices = {}

        # [B T C H W]
        # 遍历 T, 某一帧的所有 2B 个预测
        for frame_idx, outputs in enumerate(frame_outputs):

            # left_masks[:, frame_idx]  [B,1,H,W]
            # right_masks[:, frame_idx] [B,1,H,W]
            # 拼接后                     [2B,1,H,W]
            # gt_present                 [2B] 前 B 个表示左手是否存在，后 B 个表示右手是否存在
            gt_present = object_targets_from_masks(torch.cat((
                left_masks[:, frame_idx], 
                right_masks[:, frame_idx],
            )))

            # [2B] 
            pred_present = torch.cat((
                outputs["left"]["multistep_object_score_logits"][-1],
                outputs["right"]["multistep_object_score_logits"][-1],
            )).reshape(-1) > 0

            for sample_idx in range(batch_size):
                current_stats = (stats["overall"], stats[dataset_groups[sample_idx]])
                sample_gt_present = gt_present[[sample_idx, batch_size + sample_idx]]
                sample_pred_present = pred_present[[sample_idx, batch_size + sample_idx]]
                for values in current_stats:
                    values["object_tp"] += (sample_pred_present & sample_gt_present).sum().item()
                    values["object_tn"] += (~sample_pred_present & ~sample_gt_present).sum().item()
                    values["object_fp"] += (sample_pred_present & ~sample_gt_present).sum().item()
                    values["object_fn"] += (~sample_pred_present & sample_gt_present).sum().item()

                original_left_mask = batch["original_left_mask"][sample_idx][frame_idx].unsqueeze(0).to(device)
                original_right_mask = batch["original_right_mask"][sample_idx][frame_idx].unsqueeze(0).to(device)
                original_size = original_right_mask.shape[-2:]

                left_logits = upsample_logits(outputs["left"]["pred_masks_high_res"][sample_idx:sample_idx + 1], size=original_size)
                right_logits = upsample_logits(outputs["right"]["pred_masks_high_res"][sample_idx:sample_idx + 1], size=original_size)

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

                dataset_name = batch["dataset_name"][sample_idx]
                if frame_idx == 0:
                    num_vis_saved = num_vis_saved_by_dataset.get(dataset_name, 0)
                    if not args.skip_visualizations and num_vis_saved + num_frames <= max_vis_per_dataset:
                        visual_clip_indices[sample_idx] = num_vis_saved // num_frames
                        num_vis_saved_by_dataset[dataset_name] = num_vis_saved + num_frames

                clip_idx = visual_clip_indices.get(sample_idx)
                if clip_idx is not None:
                    no_memory_outputs = model.forward_single_image(images=images[sample_idx, frame_idx:frame_idx + 1], mask_inputs=None, multimask_output=False)
                    left_no_memory_logits = upsample_logits(no_memory_outputs["left"]["high_res_masks"], size=original_size)
                    right_no_memory_logits = upsample_logits(no_memory_outputs["right"]["high_res_masks"], size=original_size)
                    stage = "COND" if frame_idx == 0 else "MEMORY"
                    visualization_fn(
                        original_image=batch["original_image"][sample_idx][frame_idx],
                        left_gt_mask=original_left_mask,
                        right_gt_mask=original_right_mask,
                        left_no_memory_mask=left_no_memory_logits > 0,
                        right_no_memory_mask=right_no_memory_logits > 0,
                        left_memory_mask=left_logits > 0,
                        right_memory_mask=right_logits > 0,
                        title=f"clip {clip_idx:04d} | frame {frame_idx + 1}/{num_frames} | {stage}",
                        save_path=vis_dir / dataset_name / f"clip_{clip_idx:04d}_frame_{frame_idx:02d}.png",
                    )

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            overall = stats["overall"]
            logging.info(
                "Val Epoch %d | step %d/%d | loss=%.4f | iou=%.4f | dice=%.4f",
                epoch + 1, step, len(loader),
                overall["loss_sum"] / overall["clips"],
                overall["iou_sum"] / max(overall["foreground_hands"], 1),
                overall["dice_sum"] / max(overall["foreground_hands"], 1),
            )

    if stats["overall"]["clips"] == 0:
        raise ValueError("验证 DataLoader 中没有 Clip")

    result = {}
    for name, values in stats.items():
        if values["clips"] == 0:
            result[name] = {metric: None for metric in ("loss", "iou", "dice", "object_accuracy", "object_precision", "object_recall", "object_f1")}
            continue
        object_total = sum(values[key] for key in ("object_tp", "object_tn", "object_fp", "object_fn"))
        object_precision = values["object_tp"] / max(values["object_tp"] + values["object_fp"], 1)
        object_recall = values["object_tp"] / max(values["object_tp"] + values["object_fn"], 1)
        result[name] = {
            "loss": values["loss_sum"] / values["clips"],
            "iou": values["iou_sum"] / max(values["foreground_hands"], 1),
            "dice": values["dice_sum"] / max(values["foreground_hands"], 1),
            "object_accuracy": (values["object_tp"] + values["object_tn"]) / max(object_total, 1),
            "object_precision": object_precision,
            "object_recall": object_recall,
            "object_f1": 2 * object_precision * object_recall / max(object_precision + object_recall, 1e-8),
        }
    return result
