"""多视角双手 Memory 模型的训练可视化。"""

from pathlib import Path

import torch

from projects.framewise_sam2_modified.utils import upsample_logits
from projects.framewise_sam2_modified.visualization import _display_image, _gt_boundary, _overlay, _save


def _make_hand_panel(image, pred_mask, gt_mask, hand, normalized):
    image = _display_image(image, normalized=normalized)
    pred_mask = pred_mask > 0
    panel = _overlay(image, left=pred_mask) if hand == "left" else _overlay(image, right=pred_mask)
    boundary = _gt_boundary(gt_mask, image)
    return torch.where(boundary, panel.new_tensor((1.0, 1.0, 0.0)).view(3, 1, 1), panel)


def save_multiview_visualization(
    batch,
    frame_outputs,
    clip_idx,
    start_idx,
    split,
    epoch,
    args,
    max_images,
):
    """
    保存一个多视角 Clip。

    每张图片的上排为左手，下排为右手，每列对应一个视角。
    训练时保存每轮纠错，验证时只保存最终预测。
    """

    dataset_name = batch["dataset_name"][clip_idx]
    num_views = len(batch["view_names"][clip_idx])
    output_dir = Path(args.output_dir) / "visualizations" / f"{split}_epoch_{epoch + 1}" / dataset_name
    num_saved = 0

    for frame_idx, outputs in enumerate(frame_outputs):
        num_steps = outputs["left"]["multistep_pred_masks_high_res"].size(1)
        step_indices = range(num_steps) if split == "train" else (num_steps - 1,)

        for correction_step in step_indices:
            if num_saved >= max_images:
                return num_saved

            panels = {"left": [], "right": []}

            for view_idx in range(num_views):
                flat_idx = clip_idx * num_views + view_idx

                if split == "train":
                    image = batch["image"][clip_idx, view_idx, frame_idx]
                    left_gt = batch["left_mask"][clip_idx, view_idx, frame_idx]
                    right_gt = batch["right_mask"][clip_idx, view_idx, frame_idx]
                    left_pred = outputs["left"]["multistep_pred_masks_high_res"][flat_idx, correction_step]
                    right_pred = outputs["right"]["multistep_pred_masks_high_res"][flat_idx, correction_step]
                    normalized = True
                else:
                    image = batch["original_image"][clip_idx][view_idx][frame_idx]
                    left_gt = batch["original_left_mask"][clip_idx][view_idx][frame_idx]
                    right_gt = batch["original_right_mask"][clip_idx][view_idx][frame_idx]
                    original_size = left_gt.shape[-2:]
                    left_pred = upsample_logits(
                        outputs["left"]["multistep_pred_masks_high_res"][flat_idx:flat_idx + 1, correction_step:correction_step + 1],
                        original_size,
                    )
                    right_pred = upsample_logits(
                        outputs["right"]["multistep_pred_masks_high_res"][flat_idx:flat_idx + 1, correction_step:correction_step + 1],
                        original_size,
                    )
                    normalized = False

                panels["left"].append(_make_hand_panel(image, left_pred, left_gt, "left", normalized))
                panels["right"].append(_make_hand_panel(image, right_pred, right_gt, "right", normalized))

            left_row = torch.cat(panels["left"], dim=-1)
            right_row = torch.cat(panels["right"], dim=-1)
            multiview_image = torch.cat((left_row, right_row), dim=-2)
            image_idx = start_idx + num_saved
            filename = f"multiview_{image_idx:04d}_frame_{frame_idx:02d}_step_{correction_step:02d}.png"
            _save(multiview_image, output_dir / filename)
            num_saved += 1

    return num_saved
