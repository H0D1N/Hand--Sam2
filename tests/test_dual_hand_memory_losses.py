import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.losses import sequence_dual_hand_loss
from projects.framewise_sam2_modified.losses import dual_hand_loss


LOSS_NAMES = ("bce", "dice", "iou", "object_score")


def build_test_data():
    batch_size, num_frames, height, width = 2, 3, 4, 4
    left_masks = torch.zeros(batch_size, num_frames, 1, height, width)
    right_masks = torch.zeros_like(left_masks)

    for frame_idx in range(num_frames):
        left_masks[0, frame_idx, 0, frame_idx, 0] = 1
        left_masks[1, frame_idx, 0, frame_idx, 1] = 1
        right_masks[0, frame_idx, 0, frame_idx, 2] = 1
        right_masks[1, frame_idx, 0, frame_idx, 3] = 1

    frame_outputs = []
    trainable_tensors = []
    for frame_idx in range(num_frames):
        frame_output = {}
        for hand, masks in (("left", left_masks), ("right", right_masks)):
            logits = (masks[:, frame_idx] * 4.0 - 2.0).requires_grad_()
            ious = torch.full((batch_size, 1), 0.5, requires_grad=True)
            object_scores = torch.zeros(batch_size, 1, requires_grad=True)
            frame_output[hand] = {
                "high_res_masks": logits,
                "high_res_multimasks": logits,
                "ious": ious,
                "object_score_logits": object_scores,
            }
            trainable_tensors.extend((logits, ious, object_scores))
        frame_outputs.append(frame_output)

    return frame_outputs, left_masks, right_masks, trainable_tensors


def check_sequence_loss_is_frame_sum():
    frame_outputs, left_masks, right_masks, trainable_tensors = build_test_data()

    total_loss, loss_details = sequence_dual_hand_loss(
        frame_outputs=frame_outputs,
        left_masks=left_masks,
        right_masks=right_masks,
    )

    expected_loss = 0.0
    expected_details = {
        "left": {name: 0.0 for name in LOSS_NAMES},
        "right": {name: 0.0 for name in LOSS_NAMES},
    }
    for frame_idx, frame_output in enumerate(frame_outputs):
        frame_loss, frame_details = dual_hand_loss(
            model_output=frame_output,
            left_masks=left_masks[:, frame_idx],
            right_masks=right_masks[:, frame_idx],
        )
        expected_loss += frame_loss
        for hand in ("left", "right"):
            for name in LOSS_NAMES:
                expected_details[hand][name] += frame_details[hand][name]

    assert torch.allclose(total_loss, expected_loss)
    for hand in ("left", "right"):
        for name in LOSS_NAMES:
            assert torch.allclose(loss_details[hand][name], expected_details[hand][name])

    total_loss.backward()
    for tensor in trainable_tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all().item()


def check_invalid_frame_count():
    frame_outputs, left_masks, right_masks, _ = build_test_data()

    try:
        sequence_dual_hand_loss(frame_outputs, left_masks[:, :2], right_masks[:, :2])
    except ValueError as error:
        assert "模型输出 3 帧，GT 包含 2 帧" in str(error)
    else:
        raise AssertionError("模型输出与 GT 帧数不一致时应当报错")


def check_empty_sequence():
    _, left_masks, right_masks, _ = build_test_data()

    try:
        sequence_dual_hand_loss([], left_masks[:, :0], right_masks[:, :0])
    except ValueError as error:
        assert "序列不能为空" in str(error)
    else:
        raise AssertionError("空序列应当报错")


def main():
    check_sequence_loss_is_frame_sum()
    check_invalid_frame_count()
    check_empty_sequence()
    print("SAM2DualHandMemory sequence losses: OK")


if __name__ == "__main__":
    main()
