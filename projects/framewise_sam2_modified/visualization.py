from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

SAM2_MEAN = [0.485, 0.456, 0.406]
SAM2_STD = [0.229, 0.224, 0.225]

def _display_image(img, normalized=False):
    is_uint8 = (img.dtype == torch.uint8)
    img = img.detach().float()

    if normalized:
        mean = img.new_tensor(SAM2_MEAN).view(3, 1, 1)
        std = img.new_tensor(SAM2_STD).view(3, 1, 1)
        img = img * std + mean
    elif is_uint8:
        img = img / 255.0

    return img.clamp(0, 1)


def _overlay(img, left=None, right=None, alpha=0.5):
    result = img.clone()

    for mask, color in (
        (left, (1.0, 0.0, 0.0)),
        (right, (0.0, 0.0, 1.0)),
    ):
        if mask is None:
            continue

        mask = mask.detach().to(img.device).squeeze().gt(0.5).unsqueeze(0)
        color = img.new_tensor(color).view(3, 1, 1)

        result = torch.where(
            mask,
            result * (1.0 - alpha) + color * alpha,
            result,
        )

    return result


def make_dual_hand_visualization(
    img, lp, rp, lg, rg,
    separate=False, normalized=False, height=None,
):
    img = _display_image(img, normalized)

    if separate:
        panels = [
            img,
            _overlay(img, left=lg),
            _overlay(img, left=lp),
            _overlay(img, right=rg),
            _overlay(img, right=rp),
        ]
    else:
        panels = [
            img,
            _overlay(img, lg, rg),
            _overlay(img, lp, rp),
        ]

    result = torch.cat(panels, dim=-1)

    if height is not None and result.shape[-2] != height:
        width = round(result.shape[-1] * height / result.shape[-2])
        result = F.interpolate(
            result.unsqueeze(0),
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    return result


def _save(img, path):
    array = (
        img.clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def save_dual_hand_visualization(
    original_image, left_pred_mask, right_pred_mask,
    left_gt_mask, right_gt_mask, save_path,
):
    result = make_dual_hand_visualization(
        original_image,
        left_pred_mask,
        right_pred_mask,
        left_gt_mask,
        right_gt_mask,
    )
    _save(result, save_path)


def save_dual_hand_five_panel_visualization(
    original_image, left_pred_mask, right_pred_mask,
    left_gt_mask, right_gt_mask, save_path,
):
    result = make_dual_hand_visualization(
        original_image,
        left_pred_mask,
        right_pred_mask,
        left_gt_mask,
        right_gt_mask,
        separate=True,
    )
    _save(result, save_path)


def save_dual_hand_four_panel_visualization(
    original_image, left_pred_mask, right_pred_mask,
    left_gt_mask, right_gt_mask, save_path,
):
    image = _display_image(original_image)
    top = torch.cat([
        _overlay(image, left=left_gt_mask),
        _overlay(image, left=left_pred_mask),
    ], dim=-1)
    bottom = torch.cat([
        _overlay(image, right=right_gt_mask),
        _overlay(image, right=right_pred_mask),
    ], dim=-1)
    _save(torch.cat([top, bottom], dim=-2), save_path)


def make_dual_hand_tensorboard_image(
        normalized_image,
        left_pred_mask,
        right_pred_mask,
        left_gt_mask,
        right_gt_mask,
        output_size=384,
):
    return make_dual_hand_visualization(
        normalized_image,
        left_pred_mask,
        right_pred_mask,
        left_gt_mask,
        right_gt_mask,
        normalized=True,
        height=output_size,
    ).cpu()
