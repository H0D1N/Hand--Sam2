from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import os
import json


def upsample_logits(logits: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(logits, size=size, mode="bilinear", align_corners=False)


def save_dual_hand_visualization(
        original_image: torch.Tensor,
        left_pred_mask: torch.Tensor,
        right_pred_mask: torch.Tensor,
        left_gt_mask: torch.Tensor,
        right_gt_mask: torch.Tensor,
        save_path: Path,
) -> None:
    """
    将左右手预测和 GT 分别叠加在原图上，并保存为三栏图片。

    左手使用红色，右手使用蓝色：原图 | Ground Truth | 预测结果。
    """
    image = (
        original_image.detach().cpu().numpy()
        .transpose(1, 2, 0)
        .astype(np.uint8)
    )

    left_pred = left_pred_mask.detach().squeeze().cpu().numpy() > 0.5
    right_pred = right_pred_mask.detach().squeeze().cpu().numpy() > 0.5
    left_gt = left_gt_mask.detach().squeeze().cpu().numpy() > 0.5
    right_gt = right_gt_mask.detach().squeeze().cpu().numpy() > 0.5

    left_color = np.array([255, 0, 0], dtype=np.float32)
    right_color = np.array([0, 0, 255], dtype=np.float32)
    alpha = 0.5

    image_gt = image.copy()
    image_gt[left_gt] = (
        image_gt[left_gt] * (1.0 - alpha) + left_color * alpha
    ).astype(np.uint8)
    image_gt[right_gt] = (
        image_gt[right_gt] * (1.0 - alpha) + right_color * alpha
    ).astype(np.uint8)

    image_pred = image.copy()
    image_pred[left_pred] = (
        image_pred[left_pred] * (1.0 - alpha) + left_color * alpha
    ).astype(np.uint8)
    image_pred[right_pred] = (
        image_pred[right_pred] * (1.0 - alpha) + right_color * alpha
    ).astype(np.uint8)

    concat_image = np.concatenate(
        [image, image_gt, image_pred],
        axis=1,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(concat_image).save(save_path)

def configure_finetune_stage(
        model: torch.nn.Module,
) -> None:
    """
    冻结整个模型，只训练左右手 MaskDecoder。
    """

    model.requires_grad_(False)

    model.left_mask_decoder.requires_grad_(True)
    model.right_mask_decoder.requires_grad_(True)

def save_checkpoint(
        output_path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler,
        scheduler,
        epoch: int,
        best_val_iou: float,
        args,
        metrics: dict[str, float],
        history: list[dict[str, float]],
        include_optimizer_state: bool = True,
) -> None:
    """
    保存一个训练 checkpoint。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "best_val_iou": best_val_iou,
        "args": vars(args),
        "metrics": metrics,
        "history": history,
    }

    if include_optimizer_state:
        checkpoint["optimizer_state"] = optimizer.state_dict()
        checkpoint["scaler_state"] = scaler.state_dict()
        checkpoint["scheduler_state"] = scheduler.state_dict()

    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp"
    )

    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, output_path)

def dump_json(
        data: dict,
        output_path: str | Path,
) -> None:
    """
    将训练指标保存为 JSON。
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )

    os.replace(temporary_path, output_path)


def save_training_curves(
        history: list[dict[str, float]],
        output_path: str | Path,
) -> None:
    """保存训练/验证 loss 以及验证 IoU、Dice 曲线。"""
    from matplotlib.figure import Figure

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure = Figure(figsize=(10, 7), dpi=120)

    if not history:
        axis = figure.add_subplot(111)
        axis.text(
            0.5,
            0.5,
            "No training history available",
            horizontalalignment="center",
            verticalalignment="center",
        )
        axis.set_axis_off()
    else:
        epochs = [float(row["epoch"]) for row in history]
        train_loss = [float(row["train_loss"]) for row in history]
        val_loss = [float(row["val_loss"]) for row in history]
        val_iou = [float(row["val_iou"]) for row in history]
        val_dice = [float(row["val_dice"]) for row in history]

        loss_axis = figure.add_subplot(211)
        loss_axis.plot(
            epochs,
            train_loss,
            marker="o",
            label="Train Loss",
        )
        loss_axis.plot(
            epochs,
            val_loss,
            marker="s",
            label="Validation Loss",
        )
        loss_axis.set_title("Training Curves")
        loss_axis.set_ylabel("Loss")
        loss_axis.grid(True, linestyle="--", alpha=0.4)
        loss_axis.legend()

        metric_axis = figure.add_subplot(212)
        metric_axis.plot(
            epochs,
            val_iou,
            marker="o",
            label="Validation IoU",
        )
        metric_axis.plot(
            epochs,
            val_dice,
            marker="s",
            label="Validation Dice",
        )
        metric_axis.set_xlabel("Epoch")
        metric_axis.set_ylabel("Score")
        metric_axis.set_ylim(0.0, 1.05)
        metric_axis.grid(True, linestyle="--", alpha=0.4)
        metric_axis.legend()

    figure.tight_layout()

    temporary_path = output_path.with_name(
        f"{output_path.name}.tmp"
    )
    image_format = output_path.suffix.lstrip(".") or "png"

    figure.savefig(
        temporary_path,
        format=image_format,
        bbox_inches="tight",
    )
    figure.clear()

    os.replace(temporary_path, output_path)
