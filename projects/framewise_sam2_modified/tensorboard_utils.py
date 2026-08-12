from __future__ import annotations

from typing import Any
from contextlib import nullcontext

import torch
import argparse


from .losses import iou_target_from_logits, object_targets_from_masks
from .visualization import make_dual_hand_tensorboard_image

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
        point_prompt_fn=None,
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