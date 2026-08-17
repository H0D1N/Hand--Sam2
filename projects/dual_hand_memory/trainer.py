"""双手 Memory 模型的序列训练与验证循环。"""

import argparse
import logging
from contextlib import nullcontext

import torch
from torch.utils.data import DataLoader
from pathlib import Path

from .losses import sequence_dual_hand_loss, LOSS_NAMES
from projects.framewise_sam2_modified.losses import iou_target_from_logits, object_targets_from_masks
from projects.framewise_sam2_modified.utils import upsample_logits
from projects.framewise_sam2_modified.visualization import save_dual_hand_visualization
from projects.framewise_sam2_modified.tensorboard_utils import gradient_l2_norm

def run_training_epoch(
    model: torch.nn.Module,
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

            total_loss, loss_details = sequence_dual_hand_loss(
                frame_outputs=frame_outputs,
                left_masks=left_masks,
                right_masks=right_masks,
                bce_weight=args.bce_weight,
                dice_weight=args.dice_weight,
                iou_weight=args.iou_weight,
                object_score_weight=args.object_score_weight,
            )

        if not torch.isfinite(total_loss).item():
            raise FloatingPointError(f"Epoch {epoch + 1}, step {step}: "f"loss={total_loss.detach().item()}")
        

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
            if tensorboard_writer is not None:
                scaler.unscale_(optimizer)
                tensorboard_writer.add_scalar(
                    "optimizer/gradient_l2_norm",
                    gradient_l2_norm(model),
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
                key = f"{hand}_{name}_loss"
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
                "bce=%.4f | dice=%.4f | iou_loss=%.4f | "
                "object_score_loss=%.4f",
                epoch + 1,
                step,
                num_steps,
                total_loss.detach().item(),
                mean_details["bce"].detach().item(),
                mean_details["dice"].detach().item(),
                mean_details["iou"].detach().item(),
                mean_details["object_score"].detach().item(),
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
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    epoch: int,
    visualization_fn=save_dual_hand_visualization,
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_clips = 0
    total_iou = 0.0
    total_dice = 0.0
    total_foreground_hands = 0

    object_tp = 0
    object_tn = 0
    object_fp = 0
    object_fn = 0

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
        loss, _ = sequence_dual_hand_loss(
            frame_outputs, left_masks, right_masks,
            bce_weight=args.bce_weight, dice_weight=args.dice_weight,
            iou_weight=args.iou_weight,
            object_score_weight=args.object_score_weight,
        )

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_clips += batch_size

        
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
                outputs["left"]["object_score_logits"],
                outputs["right"]["object_score_logits"],
            )).reshape(-1) > 0

            object_tp += (pred_present & gt_present).sum().item()
            object_tn += (~pred_present & ~gt_present).sum().item()
            object_fp += (pred_present & ~gt_present).sum().item()
            object_fn += (~pred_present & gt_present).sum().item()

            for sample_idx in range(batch_size):
                original_left_mask = batch["original_left_mask"][sample_idx][frame_idx].unsqueeze(0).to(device)
                original_right_mask = batch["original_right_mask"][sample_idx][frame_idx].unsqueeze(0).to(device)
                original_size = original_right_mask.shape[-2:]

                left_logits = upsample_logits(
                    outputs["left"]["high_res_masks"][sample_idx:sample_idx + 1], size=original_size,)
                right_logits = upsample_logits(
                    outputs["right"]["high_res_masks"][sample_idx:sample_idx + 1], size=original_size,)

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

                dataset_name = batch["dataset_name"][sample_idx]
                num_vis_saved = num_vis_saved_by_dataset.get(dataset_name, 0)

                if (
                    not args.skip_visualizations
                    and num_vis_saved < max_vis_per_dataset
                ):
                    visualization_fn(
                        original_image=(
                            batch["original_image"][sample_idx][frame_idx]
                        ),
                        left_pred_mask=(left_logits > 0).float(),
                        right_pred_mask=(right_logits > 0).float(),
                        left_gt_mask=original_left_mask,
                        right_gt_mask=original_right_mask,
                        save_path=(
                            vis_dir
                            / f"{batch['sample_id'][sample_idx][frame_idx]}.png"
                        ),
                    )
                    num_vis_saved_by_dataset[dataset_name] = num_vis_saved + 1

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            logging.info(
                "Val Epoch %d | step %d/%d | loss=%.4f | iou=%.4f | dice=%.4f",
                epoch + 1, step, len(loader),
                total_loss / total_clips,
                total_iou / max(total_foreground_hands, 1),
                total_dice / max(total_foreground_hands, 1),
            )

    if total_clips == 0:
        raise ValueError("验证 DataLoader 中没有 Clip")

    object_total = object_tp + object_tn + object_fp + object_fn
    object_accuracy = (object_tp + object_tn) / max(object_total, 1)
    object_precision = object_tp / max(object_tp + object_fp, 1)
    object_recall = object_tp / max(object_tp + object_fn, 1)
    object_f1 = 2 * object_precision * object_recall / max(
        object_precision + object_recall, 1e-8,
    )

    return {
        "loss": total_loss / total_clips,
        "iou": total_iou / max(total_foreground_hands, 1),
        "dice": total_dice / max(total_foreground_hands, 1),
        "object_accuracy": object_accuracy,
        "object_precision": object_precision,
        "object_recall": object_recall,
        "object_f1": object_f1,
    }
