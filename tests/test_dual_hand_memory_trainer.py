import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.losses import DualHandMemoryLoss, LOSS_NAMES
from projects.dual_hand_memory.trainer import run_training_epoch, run_validation_epoch
from projects.framewise_sam2_modified.visualization import save_dual_hand_memory_comparison_visualization


class ToySequenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(-0.2))
        self.calls = []
        self.single_image_calls = []

    def forward(self, images, left_masks, right_masks, prompt_mode):
        self.calls.append((tuple(images.shape), prompt_mode))
        batch_size, num_frames, _, height, width = images.shape
        frame_outputs = []

        for _ in range(num_frames):
            frame_output = {}
            for hand in ("left", "right"):
                logits = self.logit.expand(batch_size, 1, height, width)
                frame_output[hand] = {
                    "multistep_pred_multimasks_high_res": [logits],
                    "multistep_pred_ious": [
                        self.logit.expand(batch_size, 1)
                    ],
                    "multistep_object_score_logits": [
                        self.logit.expand(batch_size, 1)
                    ],
                    "pred_masks_high_res": logits,
                }
            frame_outputs.append(frame_output)

        return frame_outputs

    def forward_single_image(self, images, mask_inputs=None, multimask_output=False):
        self.single_image_calls.append(tuple(images.shape))
        batch_size, _, height, width = images.shape
        logits = self.logit.expand(batch_size, 1, height, width)
        return {
            hand: {"high_res_masks": logits}
            for hand in ("left", "right")
        }


class CountingSGD(torch.optim.SGD):
    def __init__(self, params, **kwargs):
        super().__init__(params, **kwargs)
        self.step_count = 0

    def step(self, closure=None):
        self.step_count += 1
        return super().step(closure)


def build_batch(batch_size=1):
    images = torch.zeros(batch_size, 2, 3, 4, 4)
    left_masks = torch.zeros(batch_size, 2, 1, 4, 4)
    right_masks = torch.zeros_like(left_masks)
    left_masks[:, :, :, :2, :2] = 1
    right_masks[:, :, :, 2:, 2:] = 1
    return {
        "image": images,
        "left_mask": left_masks,
        "right_mask": right_masks,
        "original_image": [[images[i, frame_idx].clone() for frame_idx in range(2)] for i in range(batch_size)],
        "original_left_mask": [[left_masks[i, frame_idx].clone() for frame_idx in range(2)] for i in range(batch_size)],
        "original_right_mask": [[right_masks[i, frame_idx].clone() for frame_idx in range(2)] for i in range(batch_size)],
        "sample_id": [[f"sample-{i}-{frame_idx}" for frame_idx in range(2)] for i in range(batch_size)],
        "dataset_name": ["test"] * batch_size,
    }


def build_args():
    return SimpleNamespace(
        amp=False,
        grad_accum_steps=2,
        max_grad_norm=0.1,
        prompt_mode="point",
        debug_high_class_loss=False,
        log_interval=10,
        output_dir=Path("outputs/test-dual-hand-memory-trainer"),
        skip_visualizations=True,
    )


def build_loss_fn():
    return DualHandMemoryLoss()


def check_training_epoch():
    model = ToySequenceModel()
    optimizer = CountingSGD(model.parameters(), lr=0.1)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    initial_logit = model.logit.detach().clone()

    metrics = run_training_epoch(
        model=model,
        loss_fn=build_loss_fn(),
        loader=[build_batch(), build_batch(), build_batch()],
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        args=build_args(),
        epoch=0,
    )

    expected_keys = {"loss"}
    for hand in ("left", "right"):
        for name in LOSS_NAMES:
            expected_keys.add(f"{hand}_{name}")

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
            model=model, loss_fn=build_loss_fn(), loader=[],
            optimizer=optimizer, scaler=scaler,
            device=torch.device("cpu"), args=build_args(), epoch=0,
        )
    except ValueError as error:
        assert "训练 DataLoader 中没有 Clip" in str(error)
    else:
        raise AssertionError("空训练 DataLoader 应当报错")


def check_gradient_clipping():
    model = ToySequenceModel()
    optimizer = CountingSGD(model.parameters(), lr=1.0)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    args = build_args()
    args.grad_accum_steps = 1
    args.max_grad_norm = 0.01
    initial_logit = model.logit.detach().clone()

    run_training_epoch(
        model=model,
        loss_fn=build_loss_fn(),
        loader=[build_batch()],
        optimizer=optimizer,
        scaler=scaler,
        device=torch.device("cpu"),
        args=args,
        epoch=0,
    )

    parameter_update = (model.logit.detach() - initial_logit).abs().item()
    assert parameter_update <= args.max_grad_norm + 1e-6


def check_validation_epoch():
    model = ToySequenceModel()
    metrics = run_validation_epoch(
        model=model,
        loss_fn=build_loss_fn(),
        loader=[build_batch(), build_batch()],
        device=torch.device("cpu"),
        args=build_args(),
        epoch=0,
    )

    assert not model.training
    assert model.calls == [((1, 2, 3, 4, 4), "point")] * 2
    assert model.single_image_calls == []
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

        model = ToySequenceModel()
        run_validation_epoch(
            model=model,
            loss_fn=build_loss_fn(),
            loader=[build_batch(batch_size=2)],
            device=torch.device("cpu"),
            args=args,
            epoch=1,
            visualization_fn=record_visualization,
        )

        expected_dir = Path(output_dir) / "visualizations" / "val_epoch_2"
        assert model.single_image_calls == [(1, 3, 4, 4)] * 4
        assert [call["save_path"].parent for call in calls] == [expected_dir / "test"] * 4
        assert [call["save_path"].name for call in calls] == [
            "clip_0000_frame_00.png", "clip_0001_frame_00.png",
            "clip_0000_frame_01.png", "clip_0001_frame_01.png",
        ]
        assert [call["title"] for call in calls] == [
            "clip 0000 | frame 1/2 | COND",
            "clip 0001 | frame 1/2 | COND",
            "clip 0000 | frame 2/2 | MEMORY",
            "clip 0001 | frame 2/2 | MEMORY",
        ]


def check_memory_comparison_png():
    with tempfile.TemporaryDirectory() as output_dir:
        image = torch.zeros(3, 16, 16, dtype=torch.uint8)
        gt = torch.zeros(1, 16, 16)
        pred = torch.zeros_like(gt)
        gt[:, 3:10, 3:10] = 1
        pred[:, 5:12, 5:12] = 1
        save_path = Path(output_dir) / "comparison.png"
        save_dual_hand_memory_comparison_visualization(
            original_image=image, left_gt_mask=gt, right_gt_mask=gt,
            left_no_memory_mask=pred, right_no_memory_mask=pred,
            left_memory_mask=gt, right_memory_mask=gt,
            title="clip 0000 | frame 2/8 | MEMORY", save_path=save_path,
        )
        with Image.open(save_path) as result:
            assert result.width == 32
            assert result.height > 32


def check_empty_validation_loader():
    try:
        run_validation_epoch(
            model=ToySequenceModel(), loss_fn=build_loss_fn(), loader=[],
            device=torch.device("cpu"),
            args=build_args(), epoch=0,
        )
    except ValueError as error:
        assert "验证 DataLoader 中没有 Clip" in str(error)
    else:
        raise AssertionError("空验证 DataLoader 应当报错")


def main():
    check_training_epoch()
    check_empty_loader()
    check_gradient_clipping()
    check_validation_epoch()
    check_validation_visualizations()
    check_memory_comparison_png()
    check_empty_validation_loader()
    print("SAM2DualHandMemory training epoch: OK")


if __name__ == "__main__":
    main()
