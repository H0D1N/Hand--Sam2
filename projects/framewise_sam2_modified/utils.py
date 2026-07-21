from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


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
