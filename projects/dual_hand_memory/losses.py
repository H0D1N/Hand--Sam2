import torch

from projects.framewise_sam2_modified.losses import dual_hand_loss

LOSS_NAMES = ("bce", "dice", "iou", "object_score")

def sequence_dual_hand_loss(
    frame_outputs,
    left_masks,
    right_masks,
    bce_weight=1.0,
    dice_weight=1.0,
    iou_weight=0.1,
    object_score_weight=1.0,
):
    """
    按照原版 SAM2 的方式，对序列中所有帧的 Loss 直接求和。

    frame_outputs: forward() 返回的 list[T]
    left_masks:    [B, T, 1, H, W]
    right_masks:   [B, T, 1, H, W]
    """

    num_frames = left_masks.size(1)

    if num_frames == 0:
        raise ValueError("序列不能为空")

    if len(frame_outputs) != num_frames:
        raise ValueError(f"模型输出 {len(frame_outputs)} 帧，GT 包含 {num_frames} 帧")

    loss_kwargs = {
        "bce_weight": bce_weight,
        "dice_weight": dice_weight,
        "iou_weight": iou_weight,
        "object_score_weight": object_score_weight,
    }

    total_loss = 0.0
    loss_details = {
        "left": {name: 0.0 for name in LOSS_NAMES},
        "right": {name: 0.0 for name in LOSS_NAMES},
    }

    for frame_idx, sigle_frame_output in enumerate(frame_outputs):
        frame_loss, frame_details = dual_hand_loss(
            model_output=sigle_frame_output,
            left_masks=left_masks[:, frame_idx],
            right_masks=right_masks[:, frame_idx],
            **loss_kwargs,
        )

        total_loss = total_loss + frame_loss

        for hand in ("left", "right"):
            for name in LOSS_NAMES:
                loss_details[hand][name] += frame_details[hand][name]

    return total_loss, loss_details