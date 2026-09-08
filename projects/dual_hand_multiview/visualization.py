"""多视角双手 Memory 模型的训练可视化。"""

from pathlib import Path

import torch
from PIL import Image, ImageDraw

from projects.framewise_sam2_modified.losses import iou_target_from_logits
from projects.framewise_sam2_modified.utils import upsample_logits
from projects.framewise_sam2_modified.visualization import _display_image, _gt_boundary, _labeled_panel, _overlay, _save, _to_pil


def _make_hand_panel(image, pred_logits, gt_mask, hand, view_name, normalized, point_input, initial_points, point_scale, title):
    image = _display_image(image, normalized=normalized).cpu()
    pred_logits = pred_logits.detach().reshape(1, 1, *pred_logits.shape[-2:]).cpu()
    gt_mask = gt_mask.detach().reshape(1, 1, *gt_mask.shape[-2:]).cpu()
    iou = iou_target_from_logits(pred_logits, gt_mask).item()

    panel = _overlay(image, **{hand: pred_logits > 0})
    panel = torch.where(_gt_boundary(gt_mask, image), panel.new_tensor((1.0, 1.0, 0.0)).view(3, 1, 1), panel)
    panel = _to_pil(panel)

    if point_input is not None:
        coords = point_input["point_coords"].detach().reshape(-1, 2).float().cpu()
        labels = point_input["point_labels"].detach().reshape(-1).cpu().tolist()
        if point_scale is not None:
            coords *= coords.new_tensor(point_scale)
        coords = coords.tolist()

        radius, width = max(3, panel.width // 150), max(2, panel.width // 300)
        draw = ImageDraw.Draw(panel)
        point_start = 0
        if initial_points >= 2 and labels[:2] == [2, 3]:
            draw.rectangle((*coords[0], *coords[1]), outline="white", width=width)
            point_start = 2
        for index in range(point_start, len(coords)):
            x, y = coords[index]
            color = "white" if index < initial_points else "lime" if labels[index] == 1 else "magenta"
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline="black", width=width)

    return _labeled_panel(panel, [f"{view_name} | {hand.upper()} | IoU {iou:.3f}", title])


def save_multiview_visualization(batch, frame_outputs, clip_idx, sequence_idx, split, epoch, args):
    """
    保存一个完整的多视角 sequence。

    每张图片的上排为左手，下排为右手，每列对应一个视角。
    训练时保存全部纠错 step，验证时只保存最终 step。
    """

    dataset_name = batch["dataset_name"][clip_idx]
    view_names = batch["view_names"][clip_idx]
    num_views = len(view_names)
    num_frames = len(frame_outputs)
    output_dir = Path(args.output_dir) / "visualizations" / f"{split}_epoch_{epoch + 1}" / dataset_name
    num_saved = 0

    for frame_idx, outputs in enumerate(frame_outputs):
        num_steps = outputs["left"]["multistep_pred_masks_high_res"].size(1)
        step_indices = range(num_steps) if split == "train" else (num_steps - 1,)
        left_points = outputs["left"]["multistep_point_inputs"]
        right_points = outputs["right"]["multistep_point_inputs"]
        left_initial = 0 if left_points[0] is None else left_points[0]["point_labels"].size(1)
        right_initial = 0 if right_points[0] is None else right_points[0]["point_labels"].size(1)

        for correction_step in step_indices:
            if split == "train":
                stage = "NO CORRECTION" if num_steps == 1 else "BEFORE CORRECTION" if correction_step == 0 else f"CORRECTION {correction_step}/{num_steps - 1}"
            else:
                stage = "FINAL"
            title = f"sequence {sequence_idx:04d} | frame {frame_idx + 1}/{num_frames} | {stage}"
            panels = {"left": [], "right": []}

            for view_idx, view_name in enumerate(view_names):
                flat_idx = clip_idx * num_views + view_idx
                if split == "train":
                    image = batch["image"][clip_idx, view_idx, frame_idx]
                    left_gt = batch["left_mask"][clip_idx, view_idx, frame_idx]
                    right_gt = batch["right_mask"][clip_idx, view_idx, frame_idx]
                    left_logits = outputs["left"]["multistep_pred_masks_high_res"][flat_idx, correction_step]
                    right_logits = outputs["right"]["multistep_pred_masks_high_res"][flat_idx, correction_step]
                    normalized, point_scale = True, None
                else:
                    image = batch["original_image"][clip_idx][view_idx][frame_idx]
                    left_gt = batch["original_left_mask"][clip_idx][view_idx][frame_idx]
                    right_gt = batch["original_right_mask"][clip_idx][view_idx][frame_idx]
                    original_size = left_gt.shape[-2:]
                    left_logits = upsample_logits(outputs["left"]["multistep_pred_masks_high_res"][flat_idx:flat_idx + 1, correction_step:correction_step + 1], original_size)
                    right_logits = upsample_logits(outputs["right"]["multistep_pred_masks_high_res"][flat_idx:flat_idx + 1, correction_step:correction_step + 1], original_size)
                    point_scale = original_size[1] / batch["image"].size(-1), original_size[0] / batch["image"].size(-2)
                    normalized = False

                left_point = None if left_points[correction_step] is None else {name: value[flat_idx] for name, value in left_points[correction_step].items()}
                right_point = None if right_points[correction_step] is None else {name: value[flat_idx] for name, value in right_points[correction_step].items()}
                panels["left"].append(_make_hand_panel(image, left_logits, left_gt, "left", view_name, normalized, left_point, left_initial, point_scale, title))
                panels["right"].append(_make_hand_panel(image, right_logits, right_gt, "right", view_name, normalized, right_point, right_initial, point_scale, title))

            left_height = max(panel.height for panel in panels["left"])
            right_height = max(panel.height for panel in panels["right"])
            grid_width = max(sum(panel.width for panel in panels[hand]) for hand in ("left", "right"))
            result = Image.new("RGB", (grid_width, left_height + right_height))
            for hand, y in (("left", 0), ("right", left_height)):
                x = 0
                for panel in panels[hand]:
                    result.paste(panel, (x, y))
                    x += panel.width

            filename = f"sequence_{sequence_idx:04d}_frame_{frame_idx:02d}_step_{correction_step:02d}.png"
            _save(result, output_dir / filename)
            num_saved += 1

    return num_saved
