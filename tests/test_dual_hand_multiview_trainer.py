import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_multiview.trainer import run_validation_epoch


class FakeEvaluationModel(nn.Module):
    def forward(self, images, **kwargs):
        batch_size, num_views, num_frames = images.shape[:3]
        flat_size = batch_size * num_views
        logits = torch.ones(flat_size, 1, *images.shape[-2:])
        return [
            {
                hand: {
                    "multistep_object_score_logits": [torch.ones(flat_size, 1)],
                    "pred_masks_high_res": logits,
                }
                for hand in ("left", "right")
            }
            for _ in range(num_frames)
        ]


class FakeEvaluationLoss(nn.Module):
    def forward(self, *args, **kwargs):
        details = {
            hand: {
                "loss_mask": torch.tensor(2.0),
                "loss_dice": torch.tensor(3.0),
                "loss_iou": torch.tensor(4.0),
                "loss_class": torch.tensor(5.0),
            }
            for hand in ("left", "right")
        }
        return torch.tensor(52.0), details


def build_batch():
    images = torch.zeros(1, 2, 2, 3, 4, 4)
    masks = torch.ones(1, 2, 2, 1, 4, 4)
    original_masks = [[
        [masks[0, view, frame] for frame in range(2)]
        for view in range(2)
    ]]
    return {
        "image": images,
        "left_mask": masks,
        "right_mask": masks,
        "original_left_mask": original_masks,
        "original_right_mask": original_masks,
        "dataset_name": ["test"],
    }


def check_validation_reports_loss_components():
    metrics = run_validation_epoch(
        model=FakeEvaluationModel(),
        loss_fn=FakeEvaluationLoss(),
        loader=[build_batch()],
        device=torch.device("cpu"),
        args=SimpleNamespace(
            split="train",
            views_per_encode=1,
            skip_visualizations=True,
            log_interval=1,
        ),
        epoch=0,
        prompt_request=None,
    )["overall"]

    assert metrics["loss"] == 52.0
    assert metrics["loss_mask"] == 2.0
    assert metrics["loss_dice"] == 3.0
    assert metrics["loss_iou"] == 4.0
    assert metrics["loss_class"] == 5.0


def main():
    check_validation_reports_loss_components()
    print("Dual-hand multiview validation: OK")


if __name__ == "__main__":
    main()
