"""SAM2Modified 单帧左右手训练与验证循环。"""

from __future__ import annotations

import logging
from contextlib import nullcontext
import argparse
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
from pathlib import Path

from .losses import dual_hand_loss, iou_target_from_logits
from .utils import save_dual_hand_visualization, upsample_logits

LOSS_NAMES = ("bce", "dice", "iou", "object_score")

def run_training_epoch(
        model: torch.nn.Module,
        loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler,
        device: torch.device,
        args: argparse.Namespace,
        epoch: int,
) -> dict[str, float]:
    model.train()

    loss_sums = {"loss": 0.0}
    total_samples = 0
    amp_enabled = device.type == "cuda" and args.amp

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
                point_inputs=None,
                mask_inputs=None,
                multimask_output=False,
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

        loss_for_backward = (total_loss / args.grad_accum_steps)

        scaler.scale(loss_for_backward).backward()

        if (
            step % args.grad_accum_steps == 0
            or step == len(loader)
        ):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

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
) -> dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_dice = 0.0
    total_samples = 0
    total_foreground_hands = 0

    vis_dir = (
        Path(args.output_dir)
        / "visualizations"
        / f"val_epoch_{epoch + 1}"
    )
    num_vis_saved = 0
    max_vis_to_save = 200

    if not args.skip_visualizations:
        vis_dir.mkdir(parents=True, exist_ok=True)

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        if args.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        outputs = model.forward_single_image(
            images=images,
            point_inputs=None,
            mask_inputs=None,
            multimask_output=False,
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

            if not args.skip_visualizations and num_vis_saved < max_vis_to_save:
                save_dual_hand_visualization(
                    original_image=batch["original_image"][sample_index],
                    left_pred_mask=(left_logits > 0).float(),
                    right_pred_mask=(right_logits > 0).float(),
                    left_gt_mask=original_left_mask,
                    right_gt_mask=original_right_mask,
                    save_path=vis_dir / f"{batch['sample_id'][sample_index]}.png",
                )
                num_vis_saved += 1

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

    return {
        "loss": total_loss / total_samples,
        "iou": total_iou / total_foreground_hands,
        "dice": total_dice / total_foreground_hands,
    }