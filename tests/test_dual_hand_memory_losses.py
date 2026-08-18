import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.losses import DualHandMemoryLoss, LOSS_NAMES
from training.trainer import CORE_LOSS_KEY


def build_test_data():
    batch_size, num_frames, height, width = 2, 2, 4, 4
    left_masks = torch.zeros(batch_size, num_frames, 1, height, width)
    right_masks = torch.zeros_like(left_masks)

    # 每只手各包含一个前景样本和一个空手样本。
    left_masks[0, :, :, :2, :2] = 1
    right_masks[1, :, :, 2:, 2:] = 1

    frame_outputs = []
    trainable_tensors = []
    for frame_idx in range(num_frames):
        frame_output = {}
        for hand in ("left", "right"):
            masks_per_step = []
            ious_per_step = []
            object_scores_per_step = []

            # 第一次预测有三个候选，后续两次纠错各有一个候选。
            for step_idx, num_candidates in enumerate((3, 1, 1)):
                logits = torch.full(
                    (batch_size, num_candidates, height, width),
                    -0.5 + 0.4 * step_idx + 0.1 * frame_idx,
                    requires_grad=True,
                )
                ious = torch.full(
                    (batch_size, num_candidates),
                    0.25 + 0.1 * step_idx,
                    requires_grad=True,
                )
                object_scores = torch.zeros(
                    batch_size,
                    1,
                    requires_grad=True,
                )
                masks_per_step.append(logits)
                ious_per_step.append(ious)
                object_scores_per_step.append(object_scores)
                trainable_tensors.extend((logits, ious, object_scores))

            frame_output[hand] = {
                "multistep_pred_multimasks_high_res": masks_per_step,
                "multistep_pred_ious": ious_per_step,
                "multistep_object_score_logits": object_scores_per_step,
            }
        frame_outputs.append(frame_output)

    return frame_outputs, left_masks, right_masks, trainable_tensors


def check_official_sam2_configuration():
    sam2_loss = DualHandMemoryLoss().sam2_loss

    assert sam2_loss.weight_dict == {
        "loss_mask": 20.0,
        "loss_dice": 1.0,
        "loss_iou": 1.0,
        "loss_class": 1.0,
    }
    assert sam2_loss.supervise_all_iou
    assert sam2_loss.iou_use_l1_loss
    assert sam2_loss.pred_obj_scores
    assert sam2_loss.focal_gamma_obj_score == 0.0
    assert sam2_loss.focal_alpha_obj_score == -1.0


def check_wrapper_matches_original_sam2_loss():
    frame_outputs, left_masks, right_masks, trainable_tensors = build_test_data()
    loss_fn = DualHandMemoryLoss()

    expected = {}
    for hand, target_masks in (
        ("left", left_masks),
        ("right", right_masks),
    ):
        hand_outputs = [output[hand] for output in frame_outputs]
        hand_targets = target_masks.transpose(0, 1).squeeze(2)
        expected[hand] = loss_fn.sam2_loss(hand_outputs, hand_targets)

    actual_loss, actual_details = loss_fn(
        frame_outputs,
        left_masks,
        right_masks,
    )
    expected_loss = (
        expected["left"][CORE_LOSS_KEY]
        + expected["right"][CORE_LOSS_KEY]
    ) / 2.0

    assert torch.allclose(actual_loss, expected_loss)
    for hand in ("left", "right"):
        for name in LOSS_NAMES:
            assert torch.allclose(
                actual_details[hand][name],
                expected[hand][name],
            )

    actual_loss.backward()
    for tensor in trainable_tensors:
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all().item()


def check_empty_hand_only_trains_class_loss():
    frame_outputs, left_masks, right_masks, _ = build_test_data()
    right_masks.zero_()

    _, details = DualHandMemoryLoss()(
        frame_outputs,
        left_masks,
        right_masks,
    )

    assert details["right"]["loss_mask"].item() == 0.0
    assert details["right"]["loss_dice"].item() == 0.0
    assert details["right"]["loss_iou"].item() == 0.0
    assert details["right"]["loss_class"].item() > 0.0


def check_invalid_frame_count():
    frame_outputs, left_masks, right_masks, _ = build_test_data()

    try:
        DualHandMemoryLoss()(
            frame_outputs,
            left_masks[:, :1],
            right_masks[:, :1],
        )
    except ValueError as error:
        assert "模型输出 2 帧，GT 包含 1 帧" in str(error)
    else:
        raise AssertionError("模型输出与 GT 帧数不一致时应当报错")


def check_empty_sequence():
    _, left_masks, right_masks, _ = build_test_data()

    try:
        DualHandMemoryLoss()(
            [],
            left_masks[:, :0],
            right_masks[:, :0],
        )
    except ValueError as error:
        assert "序列不能为空" in str(error)
    else:
        raise AssertionError("空序列应当报错")


def main():
    check_official_sam2_configuration()
    check_wrapper_matches_original_sam2_loss()
    check_empty_hand_only_trains_class_loss()
    check_invalid_frame_count()
    check_empty_sequence()
    print("SAM2DualHandMemory original SAM2 losses: OK")


if __name__ == "__main__":
    main()
