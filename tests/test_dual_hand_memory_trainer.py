import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.trainer import run_training_epoch, run_validation_epoch


class ToySequenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(-0.2))
        self.calls = []

    def forward_sequence(self, images, left_masks, right_masks, prompt_mode):
        self.calls.append((tuple(images.shape), prompt_mode))
        batch_size, num_frames, _, height, width = images.shape
        frame_outputs = []

        for _ in range(num_frames):
            frame_output = {}
            for hand in ("left", "right"):
                logits = self.logit.expand(batch_size, 1, height, width)
                frame_output[hand] = {
                    "high_res_masks": logits,
                    "high_res_multimasks": logits,
                    "ious": self.logit.expand(batch_size, 1),
                    "object_score_logits": self.logit.expand(batch_size, 1),
                }
            frame_outputs.append(frame_output)

        return frame_outputs


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def build_batch():
    images = torch.zeros(1, 2, 3, 4, 4)
    left_masks = torch.zeros(1, 2, 1, 4, 4)
    right_masks = torch.zeros_like(left_masks)
    left_masks[:, :, :, :2, :2] = 1
    right_masks[:, :, :, 2:, 2:] = 1
    return {
        "image": images,
        "left_mask": left_masks,
        "right_mask": right_masks,
        "original_image": [[images[0, frame_idx].clone() for frame_idx in range(2)]],
        "original_left_mask": [[left_masks[0, frame_idx].clone() for frame_idx in range(2)]],
        "original_right_mask": [[right_masks[0, frame_idx].clone() for frame_idx in range(2)]],
        "sample_id": [["sample-0", "sample-1"]],
        "dataset_name": ["test"],
    }


def build_args():
    return SimpleNamespace(
        amp=False,
        grad_accum_steps=2,
        prompt_mode="point",
        bce_weight=1.0,
        dice_weight=1.0,
        iou_weight=0.1,
        object_score_weight=1.0,
        log_interval=10,
        output_dir=Path("outputs/test-dual-hand-memory-trainer"),
        skip_visualizations=True,
    )


def check_training_epoch():
    model = ToySequenceModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    initial_logit = model.logit.detach().clone()

    metrics = run_training_epoch(
        model=model,
        loader=[build_batch(), build_batch(), build_batch()],
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        args=build_args(),
        epoch=0,
    )

    expected_keys = {"loss"}
    for hand in ("left", "right"):
        for name in ("bce", "dice", "iou", "object_score"):
            expected_keys.add(f"{hand}_{name}_loss")

    assert model.training
    assert model.calls == [((1, 2, 3, 4, 4), "point")] * 3
    assert optimizer.step_count == 2
    assert not torch.equal(model.logit.detach(), initial_logit)
    assert set(metrics) == expected_keys
    assert all(torch.isfinite(torch.tensor(value)).item() for value in metrics.values())


def check_empty_loader():
    model = ToySequenceModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    try:
        run_training_epoch(
            model=model, loader=[], optimizer=optimizer, scaler=scaler,
            device=torch.device("cpu"), args=build_args(), epoch=0,
        )
    except ValueError as error:
        assert "训练 DataLoader 中没有 Clip" in str(error)
    else:
        raise AssertionError("空训练 DataLoader 应当报错")


def check_validation_epoch():
    model = ToySequenceModel()
    metrics = run_validation_epoch(
        model=model,
        loader=[build_batch(), build_batch()],
        device=torch.device("cpu"),
        args=build_args(),
        epoch=0,
    )

    assert not model.training
    assert model.calls == [((1, 2, 3, 4, 4), "point")] * 2
    assert set(metrics) == {
        "loss", "iou", "dice", "object_accuracy",
        "object_precision", "object_recall", "object_f1",
    }
    assert all(torch.isfinite(torch.tensor(value)).item() for value in metrics.values())


def check_validation_visualizations():
    calls = []

    def record_visualization(**kwargs):
        calls.append(kwargs)

    with tempfile.TemporaryDirectory() as output_dir:
        args = build_args()
        args.output_dir = Path(output_dir)
        args.skip_visualizations = False

        run_validation_epoch(
            model=ToySequenceModel(),
            loader=[build_batch()],
            device=torch.device("cpu"),
            args=args,
            epoch=1,
            visualization_fn=record_visualization,
        )

        expected_dir = Path(output_dir) / "visualizations" / "val_epoch_2"
        assert [call["save_path"].parent for call in calls] == [expected_dir] * 2
        assert [call["save_path"].name for call in calls] == [
            "sample-0.png", "sample-1.png",
        ]


def check_empty_validation_loader():
    try:
        run_validation_epoch(
            model=ToySequenceModel(), loader=[], device=torch.device("cpu"),
            args=build_args(), epoch=0,
        )
    except ValueError as error:
        assert "验证 DataLoader 中没有 Clip" in str(error)
    else:
        raise AssertionError("空验证 DataLoader 应当报错")


def main():
    check_training_epoch()
    check_empty_loader()
    check_validation_epoch()
    check_validation_visualizations()
    check_empty_validation_loader()
    print("SAM2DualHandMemory training epoch: OK")


if __name__ == "__main__":
    main()
