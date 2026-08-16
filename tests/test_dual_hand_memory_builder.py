import argparse
import gc
import sys
import tempfile
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.builder import (
    build_sam2_dual_hand_memory_tiny,
    configure_memory_training,
)
from projects.framewise_sam2_modified.builder import (
    build_sam2_modified_tiny,
)
from training.model.sam2_dual_hand_memory import (
    SAM2DualHandMemory,
)


DEFAULT_CHECKPOINT_PATH = (
    REPOSITORY_ROOT / "checkpoints" / "sam2.1_hiera_tiny.pt"
)
TEST_IMAGE_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test official and framewise dual-hand Memory builders."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT_PATH,
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def check_memory_structure(model):
    assert isinstance(model, SAM2DualHandMemory)
    assert not hasattr(model, "memory_attention")
    assert not hasattr(model, "memory_encoder")

    module_pairs = (
        (model.left_memory_attention, model.right_memory_attention),
        (model.left_memory_encoder, model.right_memory_encoder),
    )

    for left_module, right_module in module_pairs:
        assert left_module is not right_module

        left_parameters = dict(left_module.named_parameters())
        right_parameters = dict(right_module.named_parameters())
        assert left_parameters.keys() == right_parameters.keys()

        for name, left_parameter in left_parameters.items():
            right_parameter = right_parameters[name]
            assert torch.equal(left_parameter, right_parameter), name
            assert left_parameter.data_ptr() != right_parameter.data_ptr(), name


def check_memory_training_configuration(model):
    configure_memory_training(model)

    memory_prefixes = (
        "left_memory_attention.",
        "right_memory_attention.",
        "left_memory_encoder.",
        "right_memory_encoder.",
    )
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names
    assert all(name.startswith(memory_prefixes) for name in trainable_names)
    for prefix in memory_prefixes:
        assert any(name.startswith(prefix) for name in trainable_names), prefix

    assert not any("adapter" in name for name in trainable_names)
    assert not any(
        parameter.requires_grad
        for parameter in model.left_mask_decoder.parameters()
    )
    assert not any(
        parameter.requires_grad
        for parameter in model.right_mask_decoder.parameters()
    )


def check_official_checkpoint(checkpoint_path, device):
    model = build_sam2_dual_hand_memory_tiny(
        sam_checkpoint=checkpoint_path,
        image_size=TEST_IMAGE_SIZE,
        device=device,
        mode="eval",
        use_image_adapter=True,
        adapter_dim=16,
    )

    assert model.image_size == TEST_IMAGE_SIZE
    assert model.training is False
    assert next(model.parameters()).device == device
    assert all(
        hasattr(block, "adapter")
        for block in model.image_encoder.trunk.blocks
    )
    check_memory_structure(model)

    del model
    gc.collect()


def create_framewise_checkpoint(checkpoint_path, output_path):
    model = build_sam2_modified_tiny(
        checkpoint_path=checkpoint_path,
        image_size=TEST_IMAGE_SIZE,
        device="cpu",
        mode="eval",
        use_image_adapter=True,
        adapter_dim=32,
        adapter_dropout=0.2,
        adapter_init_scale=1e-2,
    )

    state_dict = model.state_dict()
    changed_keys = {
        "left_mask_decoder.mask_tokens.weight": 0.125,
        "right_mask_decoder.mask_tokens.weight": 0.250,
        "image_encoder.trunk.blocks.0.adapter.scale": 0.375,
        "memory_attention.layers.0.self_attn.q_proj.weight": 0.500,
        "memory_encoder.mask_downsampler.encoder.0.weight": 0.625,
    }

    with torch.no_grad():
        for key, value in changed_keys.items():
            state_dict[key].fill_(value)

    expected_values = {
        key: tensor.detach().clone()
        for key, tensor in state_dict.items()
        if key in changed_keys
    }

    torch.save(
        {
            "model_state": state_dict,
            "args": {
                "image_size": TEST_IMAGE_SIZE,
                "use_image_adapter": True,
                "use_decoder_adapter": False,
                "adapter_dim": 32,
                "adapter_dropout": 0.2,
                "adapter_init_scale": 1e-2,
            },
        },
        output_path,
    )

    del model, state_dict
    gc.collect()
    return expected_values


def check_framewise_checkpoint(checkpoint_path, device):
    with tempfile.TemporaryDirectory() as temporary_directory:
        framewise_checkpoint = Path(temporary_directory) / "best.pt"
        expected_values = create_framewise_checkpoint(
            checkpoint_path=checkpoint_path,
            output_path=framewise_checkpoint,
        )

        model = build_sam2_dual_hand_memory_tiny(
            framewise_checkpoint=framewise_checkpoint,
            image_size=512,
            device=device,
            mode="train",
            use_image_adapter=False,
            adapter_dim=8,
        )

    assert model.image_size == TEST_IMAGE_SIZE
    assert model.training is True
    assert next(model.parameters()).device == device
    assert model.image_encoder.trunk.blocks[0].adapter.down_proj.out_features == 32
    assert model.image_encoder.trunk.blocks[0].adapter.dropout.p == 0.2
    check_memory_structure(model)

    model_state = model.state_dict()
    for key in (
        "left_mask_decoder.mask_tokens.weight",
        "right_mask_decoder.mask_tokens.weight",
        "image_encoder.trunk.blocks.0.adapter.scale",
    ):
        assert torch.equal(model_state[key].cpu(), expected_values[key]), key

    memory_key_pairs = (
        (
            "memory_attention.layers.0.self_attn.q_proj.weight",
            "left_memory_attention.layers.0.self_attn.q_proj.weight",
            "right_memory_attention.layers.0.self_attn.q_proj.weight",
        ),
        (
            "memory_encoder.mask_downsampler.encoder.0.weight",
            "left_memory_encoder.mask_downsampler.encoder.0.weight",
            "right_memory_encoder.mask_downsampler.encoder.0.weight",
        ),
    )

    for source_key, left_key, right_key in memory_key_pairs:
        expected = expected_values[source_key]
        assert torch.equal(model_state[left_key].cpu(), expected), left_key
        assert torch.equal(model_state[right_key].cpu(), expected), right_key

    check_memory_training_configuration(model)


def main():
    args = parse_args()
    checkpoint_path = args.checkpoint.resolve()
    device = torch.device(args.device)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    print("Checking official SAM2 checkpoint initialization...")
    check_official_checkpoint(checkpoint_path, device)

    print("Checking framewise checkpoint initialization...")
    check_framewise_checkpoint(checkpoint_path, device)

    print(f"SAM2DualHandMemory builders: OK on {device}")


if __name__ == "__main__":
    main()
