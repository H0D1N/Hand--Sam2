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
from training.model.sam2_modified import SAM2Modified


DEFAULT_CHECKPOINT_PATH = REPOSITORY_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Test SAM2Modified construction and official checkpoint loading.")
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
        help="Device used to build the model, such as cpu or cuda:0.",
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


def load_official_mask_tokens(checkpoint_path):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if "model" not in checkpoint:
        raise KeyError("The checkpoint does not contain a 'model' state_dict.")

    return checkpoint["model"]["sam_mask_decoder.mask_tokens.weight"]


def check_model_structure(model):
    assert isinstance(model, SAM2Modified)

    assert hasattr(model, "left_mask_decoder")
    assert hasattr(model, "right_mask_decoder")
    assert not hasattr(model, "sam_mask_decoder")

    assert model.left_mask_decoder is not model.right_mask_decoder

    assert not hasattr(model, "sam_model")

    for block in model.image_encoder.trunk.blocks:
        assert not hasattr(block, "adapter")


def check_model_device(model, expected_device):
    actual_device = next(model.parameters()).device

    assert actual_device.type == expected_device.type

    if expected_device.index is not None:
        assert actual_device.index == expected_device.index


def check_decoder_parameters_are_independent(model):
    left_parameters = dict(model.left_mask_decoder.named_parameters())
    right_parameters = dict(model.right_mask_decoder.named_parameters())

    assert left_parameters.keys() == right_parameters.keys()

    for parameter_name in left_parameters:
        left_parameter = left_parameters[parameter_name]
        right_parameter = right_parameters[parameter_name]

        assert torch.equal(
            left_parameter,
            right_parameter,
        ), parameter_name

        assert left_parameter.data_ptr() != right_parameter.data_ptr(), parameter_name


def check_official_mask_tokens_are_loaded(
    model,
    checkpoint_path,
):
    official_mask_tokens = load_official_mask_tokens(checkpoint_path)

    left_mask_tokens = model.left_mask_decoder.mask_tokens.weight.detach().cpu()
    right_mask_tokens = model.right_mask_decoder.mask_tokens.weight.detach().cpu()

    assert torch.equal(
        left_mask_tokens,
        official_mask_tokens,
    )
    assert torch.equal(
        right_mask_tokens,
        official_mask_tokens,
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

    check_model_structure(model)
    check_model_device(model, device)
    check_decoder_parameters_are_independent(model)
    check_official_mask_tokens_are_loaded(
        model,
        checkpoint_path,
    )

    print("SAM2Modified builder checkpoint loading: " f"OK on {device}")


if __name__ == "__main__":
    main()
