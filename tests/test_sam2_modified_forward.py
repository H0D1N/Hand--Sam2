import argparse
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.framewise_sam2_modified import (
    build_sam2_modified_tiny,
)


DEFAULT_CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"

EXPECTED_BRANCH_KEYS = {
    "low_res_multimasks",
    "high_res_multimasks",
    "ious",
    "low_res_masks",
    "high_res_masks",
    "obj_ptr",
    "object_score_logits",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test no-prompt single-image forward in SAM2Modified."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
        help="Path to the official SAM2.1 tiny checkpoint.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device used for forward, such as cpu or cuda:0.",
    )
    return parser.parse_args()


def prepare_device(device_name):
    device = torch.device(device_name)

    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested, but torch.cuda.is_available() is False."
            )

        if device.index is not None:
            torch.cuda.set_device(device)

    return device


def check_tensor(name, tensor, expected_shape, expected_device):
    assert isinstance(tensor, torch.Tensor), name

    actual_shape = tuple(tensor.shape)
    shape_error = f"{name}: expected shape {expected_shape}, " f"but got {actual_shape}"
    assert actual_shape == expected_shape, shape_error

    assert tensor.device.type == expected_device.type, (
        f"{name}: expected device type {expected_device.type}, "
        f"but got {tensor.device.type}"
    )

    if expected_device.index is not None:
        assert tensor.device.index == expected_device.index, (
            f"{name}: expected device index {expected_device.index}, "
            f"but got {tensor.device.index}"
        )

    assert torch.isfinite(tensor).all().item(), f"{name} contains NaN or Inf"
    assert (
        tensor.requires_grad is False
    ), f"{name} should not require gradients in inference mode"


def check_branch_outputs(
    branch_name,
    branch_outputs,
    batch_size,
    image_size,
    hidden_dim,
    device,
):
    assert isinstance(branch_outputs, dict), branch_name
    assert set(branch_outputs.keys()) == EXPECTED_BRANCH_KEYS

    low_res_size = image_size // 4

    expected_shapes = {
        "low_res_multimasks": (
            batch_size,
            1,
            low_res_size,
            low_res_size,
        ),
        "high_res_multimasks": (
            batch_size,
            1,
            image_size,
            image_size,
        ),
        "ious": (batch_size, 1),
        "low_res_masks": (
            batch_size,
            1,
            low_res_size,
            low_res_size,
        ),
        "high_res_masks": (
            batch_size,
            1,
            image_size,
            image_size,
        ),
        "obj_ptr": (batch_size, hidden_dim),
        "object_score_logits": (batch_size, 1),
    }

    for output_name, expected_shape in expected_shapes.items():
        check_tensor(
            name=f"{branch_name}.{output_name}",
            tensor=branch_outputs[output_name],
            expected_shape=expected_shape,
            expected_device=device,
        )


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    device = prepare_device(args.device)

    print(f"Building SAM2Modified on {device}...")

    model = build_sam2_modified_tiny(
        checkpoint_path=checkpoint_path,
        device=device,
        mode="eval",
    )

    assert model.training is False

    batch_size = 1
    torch.manual_seed(0)

    images = torch.randn(
        batch_size,
        3,
        model.image_size,
        model.image_size,
        device=device,
    )

    print("Running no-prompt single-image forward...")

    with torch.inference_mode():
        outputs = model.forward_single_image(
            images=images,
            point_inputs=None,
            mask_inputs=None,
            multimask_output=False,
        )

    assert isinstance(outputs, dict)
    assert set(outputs.keys()) == {"left", "right"}

    check_branch_outputs(
        branch_name="left",
        branch_outputs=outputs["left"],
        batch_size=batch_size,
        image_size=model.image_size,
        hidden_dim=model.hidden_dim,
        device=device,
    )
    check_branch_outputs(
        branch_name="right",
        branch_outputs=outputs["right"],
        batch_size=batch_size,
        image_size=model.image_size,
        hidden_dim=model.hidden_dim,
        device=device,
    )

    print(f"SAM2Modified no-prompt single-image forward: OK on {device}")


if __name__ == "__main__":
    main()
