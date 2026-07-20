import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.framewise_sam2_modified.losses import (
    dice_loss_from_logits,
    dual_hand_loss,
    iou_target_from_logits,
    one_hand_loss,
)


def check_dice_loss():
    targets = torch.tensor(
        [[[[1.0, 0.0],
           [0.0, 1.0]]]]
    )

    # 在真实 mask 为 1 的位置输出正数，
    # 在真实 mask 为 0 的位置输出负数。
    good_logits = targets * 20.0 - 10.0

    # 把正确预测完全反过来。
    bad_logits = -good_logits

    good_loss = dice_loss_from_logits(
        good_logits,
        targets,
    )

    bad_loss = dice_loss_from_logits(
        bad_logits,
        targets,
    )

    assert good_loss.ndim == 0
    assert bad_loss.ndim == 0
    assert good_loss.item() < bad_loss.item()


def check_iou_target():
    targets = torch.tensor(
        [[[[1.0, 0.0],
           [0.0, 1.0]]]]
    )

    logits = targets * 20.0 - 10.0

    target_ious = iou_target_from_logits(
        logits,
        targets,
    )

    assert tuple(target_ious.shape) == (1,)

    assert torch.allclose(
        target_ious,
        torch.ones(1),
    )


def check_dual_hand_loss():
    left_masks = torch.tensor(
        [[[[1.0, 0.0],
           [0.0, 0.0]]]]
    )

    right_masks = torch.tensor(
        [[[[0.0, 0.0],
           [0.0, 1.0]]]]
    )

    left_logits = (
        left_masks * 4.0 - 2.0
    ).requires_grad_()

    right_logits = (
        right_masks * 4.0 - 2.0
    ).requires_grad_()

    left_predicted_ious = torch.tensor(
        [[0.5]],
        requires_grad=True,
    )

    right_predicted_ious = torch.tensor(
        [[0.5]],
        requires_grad=True,
    )

    model_output = {
        "left": {
            "high_res_masks": left_logits,
            "ious": left_predicted_ious,
        },
        "right": {
            "high_res_masks": right_logits,
            "ious": right_predicted_ious,
        },
    }

    total_loss, loss_details = dual_hand_loss(
        model_output=model_output,
        left_masks=left_masks,
        right_masks=right_masks,
        bce_weight=1.0,
        dice_weight=1.0,
        iou_weight=0.1,
    )

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss).item()

    assert set(loss_details.keys()) == {
        "left",
        "right",
    }

    assert set(loss_details["left"].keys()) == {
        "bce",
        "dice",
        "iou",
    }

    assert set(loss_details["right"].keys()) == {
        "bce",
        "dice",
        "iou",
    }

    # 分别计算左右分支，用来检查 dual_hand_loss
    # 有没有把 right_masks 错写成 left_masks。
    expected_left_loss, _ = one_hand_loss(
        hand_outputs=model_output["left"],
        target_masks=left_masks,
    )

    expected_right_loss, _ = one_hand_loss(
        hand_outputs=model_output["right"],
        target_masks=right_masks,
    )

    expected_total_loss = (
        expected_left_loss + expected_right_loss
    ) / 2.0

    assert torch.allclose(
        total_loss,
        expected_total_loss,
    )

    # 检查能否反向传播。
    total_loss.backward()

    assert left_logits.grad is not None
    assert right_logits.grad is not None

    assert left_predicted_ious.grad is not None
    assert right_predicted_ious.grad is not None

    assert torch.isfinite(left_logits.grad).all().item()
    assert torch.isfinite(right_logits.grad).all().item()


def main():
    check_dice_loss()
    check_iou_target()
    check_dual_hand_loss()

    print("SAM2Modified dual-hand losses: OK")


if __name__ == "__main__":
    main()