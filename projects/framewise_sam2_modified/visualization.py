from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

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
    for mask, color in ((left, (1.0, 0.0, 0.0)), (right, (0.0, 0.0, 1.0))):
        if mask is None:
            continue
        mask = mask.detach().to(img.device).squeeze().gt(0.5).unsqueeze(0)
        color = img.new_tensor(color).view(3, 1, 1)
        result = torch.where(mask, result * (1.0 - alpha) + color * alpha, result)
    return result


def _gt_boundary(gt_mask, image):
    """提取 GT 轮廓；低分辨率原图使用更细的轮廓。"""

    gt = gt_mask.detach().to(image.device).float().squeeze().gt(0.5).float()[None, None]
    radius = min(2, max(1, round(min(image.shape[-2:]) / 400)))
    return (F.max_pool2d(gt, 2 * radius + 1, stride=1, padding=radius) + F.max_pool2d(-gt, 2 * radius + 1, stride=1, padding=radius)).squeeze().gt(0)


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


def _to_pil(img):
    array = (
        img.clamp(0, 1)
        .mul(255)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .cpu()
        .numpy()
    )
    return Image.fromarray(array)


def _save(image, path):
    """保存 Tensor 或 PIL 图片。"""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, torch.Tensor):
        image = _to_pil(image)
    image.save(path)


def _metric(value):
    return "N/A" if value is None else f"{value:.3f}"


def _area_percent(area, image_area):
    return 100.0 * area / max(image_area, 1)


def _label_font(panel_width):
    size = max(11, min(24, round(panel_width / 55)))
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _labeled_panel(panel, lines):
    if isinstance(panel, torch.Tensor):
        panel = _to_pil(panel)
    font = _label_font(panel.width)
    probe = ImageDraw.Draw(panel)
    boxes = [probe.textbbox((0, 0), line, font=font) for line in lines]
    line_height = max(box[3] - box[1] for box in boxes)
    padding = max(4, line_height // 3)
    header_height = len(lines) * line_height + (len(lines) + 1) * padding

    result = Image.new("RGB", (panel.width, panel.height + header_height), (20, 20, 20))
    result.paste(panel, (0, header_height))
    draw = ImageDraw.Draw(result)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=(255, 255, 255), font=font)
        y += line_height + padding
    return result


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
    left_gt_mask, right_gt_mask, metrics, save_path,
):
    image = _display_image(original_image)
    panels = []
    for side, pred_mask, gt_mask in (
        ("LEFT", left_pred_mask, left_gt_mask),
        ("RIGHT", right_pred_mask, right_gt_mask),
    ):
        hand = metrics[side.lower()]
        gt_area, pred_area, image_area = hand["gt_area"], hand["pred_area"], hand["image_area"]
        pred_gt_ratio = None if gt_area == 0 else pred_area / gt_area
        pred_gt_text = _metric(pred_gt_ratio)
        panels.extend([
            _labeled_panel(_overlay(image, left=gt_mask) if side == "LEFT" else _overlay(image, right=gt_mask), [
                f"{side} GT | {gt_area:,} px | GT/Image {_area_percent(gt_area, image_area):.3f}% | {hand['gt_state']}",
                " ",
            ]),
            _labeled_panel(_overlay(image, left=pred_mask) if side == "LEFT" else _overlay(image, right=pred_mask), [
                f"{side} PRED | {pred_area:,} px | Pred/Image {_area_percent(pred_area, image_area):.3f}% | Pred/GT {pred_gt_text} | {hand['pred_state']}",
                f"IoU {_metric(hand['region_iou'])} | Precision {_metric(hand['region_precision'])} | Recall {_metric(hand['region_recall'])}",
            ]),
        ])

    panel_width, panel_height = panels[0].size
    result = Image.new("RGB", (2 * panel_width, 2 * panel_height))
    for panel, position in zip(panels, ((0, 0), (panel_width, 0), (0, panel_height), (panel_width, panel_height))):
        result.paste(panel, position)
    _save(result, save_path)


def save_dual_hand_memory_comparison_visualization(
    original_image, left_gt_mask, right_gt_mask,
    left_no_memory_mask, right_no_memory_mask,
    left_memory_mask, right_memory_mask,
    left_no_memory_point, right_no_memory_point, title, save_path,
):
    """保存 Point-prompt No-Memory 与 Memory 的 2x2 定性对照图。"""

    image = _display_image(original_image)
    panels = []
    for label, pred_mask, gt_mask, side, point in (
        ("LEFT | NO MEMORY | POINT", left_no_memory_mask, left_gt_mask, "left", left_no_memory_point),
        ("LEFT | MEMORY", left_memory_mask, left_gt_mask, "left", None),
        ("RIGHT | NO MEMORY | POINT", right_no_memory_mask, right_gt_mask, "right", right_no_memory_point),
        ("RIGHT | MEMORY", right_memory_mask, right_gt_mask, "right", None),
    ):
        panel = _overlay(image, left=pred_mask) if side == "left" else _overlay(image, right=pred_mask)
        panel = torch.where(_gt_boundary(gt_mask, image), panel.new_tensor((1.0, 1.0, 0.0)).view(3, 1, 1), panel)
        panel = _to_pil(panel)
        if point is not None:
            x, y = point.detach().cpu().tolist()
            radius, width = max(3, panel.width // 150), max(2, panel.width // 300)
            ImageDraw.Draw(panel).ellipse((x - radius, y - radius, x + radius, y + radius), fill="white", outline="black", width=width)
        panels.append(_labeled_panel(panel, [label, title]))

    panel_width, panel_height = panels[0].size
    result = Image.new("RGB", (2 * panel_width, 2 * panel_height))
    for panel, position in zip(panels, ((0, 0), (panel_width, 0), (0, panel_height), (panel_width, panel_height))):
        result.paste(panel, position)
    _save(result, save_path)


def save_dual_hand_correction_visualization(
    normalized_image, left_gt_mask, right_gt_mask, left_pred_mask, right_pred_mask,
    left_point_input, right_point_input, left_initial_points, right_initial_points,
    left_iou, right_iou, title, save_path,
):
    """保存当前训练帧某一轮纠错后的左右手预测。"""

    image = _display_image(normalized_image, normalized=True)
    panels = []
    for side, pred, gt_mask, point_input, initial_points, iou in (
        ("LEFT", left_pred_mask, left_gt_mask, left_point_input, left_initial_points, left_iou),
        ("RIGHT", right_pred_mask, right_gt_mask, right_point_input, right_initial_points, right_iou),
    ):
        panel = _overlay(image, left=pred) if side == "LEFT" else _overlay(image, right=pred)
        panel = torch.where(_gt_boundary(gt_mask, image), panel.new_tensor((1.0, 1.0, 0.0)).view(3, 1, 1), panel)
        panel = _to_pil(panel)

        if point_input is not None:
            coords = point_input["point_coords"].detach().reshape(-1, 2).cpu().tolist()
            labels = point_input["point_labels"].detach().reshape(-1).cpu().tolist()
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

        panels.append(_labeled_panel(panel, [f"{side} | IoU {iou:.3f}", title]))

    result = Image.new("RGB", (panels[0].width + panels[1].width, max(panel.height for panel in panels)))
    result.paste(panels[0], (0, 0))
    result.paste(panels[1], (panels[0].width, 0))
    _save(result, save_path)


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
