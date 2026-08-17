import argparse
import gc
import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.builder import (
    build_sam2_dual_hand_memory_tiny,
    configure_memory_training,
)
from training.model.adapter import (
    iter_image_encoder_adapters,
    iter_mask_decoder_adapters,
)
from training.model.sam2_dual_hand_memory import (
    SAM2DualHandMemory,
)


TEST_IMAGE_SIZE = 256


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test official and framewise dual-hand Memory builders."
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument(
        "--sam-checkpoint",
        type=Path,
        help="Official SAM2 checkpoint containing checkpoint['model'].",
    )
    checkpoint_group.add_argument(
        "--framewise-checkpoint",
        type=Path,
        help="Framewise dual-decoder best.pt containing checkpoint['model_state'].",
    )
    parser.add_argument(
        "--finetune-mode",
        choices=("auto", "memory-only", "decoder-memory"),
        default="auto",
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


def check_finetune_configuration(model, finetune_mode):
    configure_memory_training(model, finetune_mode)

    memory_prefixes = (
        "left_memory_attention.",
        "right_memory_attention.",
        "left_memory_encoder.",
        "right_memory_encoder.",
    )
    decoder_prefixes = (
        "left_mask_decoder.",
        "right_mask_decoder.",
    )
    trainable_names = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    assert trainable_names
    for prefix in memory_prefixes:
        assert any(name.startswith(prefix) for name in trainable_names), prefix

    if finetune_mode == "memory-only":
        assert all(name.startswith(memory_prefixes) for name in trainable_names)
        assert not any("adapter" in name for name in trainable_names)
        assert not any(
            parameter.requires_grad
            for parameter in model.left_mask_decoder.parameters()
        )
        assert not any(
            parameter.requires_grad
            for parameter in model.right_mask_decoder.parameters()
        )
        return

    assert finetune_mode == "decoder-memory"
    assert any(name.startswith("left_mask_decoder.") for name in trainable_names)
    assert any(name.startswith("right_mask_decoder.") for name in trainable_names)
    assert all(
        name.startswith(memory_prefixes + decoder_prefixes)
        or (name.startswith("image_encoder.") and ".adapter." in name)
        for name in trainable_names
    )
    assert not any(
        name.startswith("image_encoder.") and ".adapter." not in name
        for name in trainable_names
    )
    assert not any(name.startswith("sam_prompt_encoder.") for name in trainable_names)

    image_adapters = list(iter_image_encoder_adapters(model))
    assert all(
        parameter.requires_grad
        for adapter in image_adapters
        for parameter in adapter.parameters()
    )

    decoder_adapters = list(iter_mask_decoder_adapters(model))
    if decoder_adapters:
        decoder_adapter_parameter_ids = {
            id(parameter)
            for adapter in decoder_adapters
            for parameter in adapter.parameters()
        }
        assert all(
            parameter.requires_grad
            for adapter in decoder_adapters
            for parameter in adapter.parameters()
        )
        for decoder in (model.left_mask_decoder, model.right_mask_decoder):
            assert all(
                parameter.requires_grad
                == (id(parameter) in decoder_adapter_parameter_ids)
                for parameter in decoder.transformer.parameters()
            )
    else:
        assert all(
            parameter.requires_grad
            for decoder in (model.left_mask_decoder, model.right_mask_decoder)
            for parameter in decoder.parameters()
        )


def check_official_checkpoint(checkpoint_path, device, finetune_mode):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    official_state = checkpoint["model"]

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

    model_state = model.state_dict()
    for source_key, expected in official_state.items():
        if source_key.startswith("sam_mask_decoder."):
            suffix = source_key[len("sam_mask_decoder."):]
            target_keys = (
                f"left_mask_decoder.{suffix}",
                f"right_mask_decoder.{suffix}",
            )
        elif source_key.startswith("memory_attention."):
            suffix = source_key[len("memory_attention."):]
            target_keys = (
                f"left_memory_attention.{suffix}",
                f"right_memory_attention.{suffix}",
            )
        elif source_key.startswith("memory_encoder."):
            suffix = source_key[len("memory_encoder."):]
            target_keys = (
                f"left_memory_encoder.{suffix}",
                f"right_memory_encoder.{suffix}",
            )
        else:
            target_keys = (source_key,)

        for target_key in target_keys:
            assert torch.equal(model_state[target_key].cpu(), expected), target_key

    check_finetune_configuration(model, finetune_mode)

    del model, checkpoint, official_state
    gc.collect()


def check_framewise_checkpoint(checkpoint_path, device, finetune_mode):
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    checkpoint_args = checkpoint["args"]
    framewise_state = checkpoint["model_state"]

    assert any(key.startswith("memory_attention.") for key in framewise_state)
    assert any(key.startswith("memory_encoder.") for key in framewise_state)
    assert not any(key.startswith("left_memory_attention.") for key in framewise_state)
    assert not any(key.startswith("right_memory_attention.") for key in framewise_state)

    model = build_sam2_dual_hand_memory_tiny(
        framewise_checkpoint=checkpoint_path,
        image_size=512,
        device=device,
        mode="train",
        use_image_adapter=False,
        adapter_dim=8,
    )

    assert model.image_size == checkpoint_args["image_size"]
    assert model.training is True
    assert next(model.parameters()).device == device
    check_memory_structure(model)

    model_state = model.state_dict()
    for source_key, expected in framewise_state.items():
        if source_key.startswith("memory_attention."):
            suffix = source_key[len("memory_attention."):]
            target_keys = (
                f"left_memory_attention.{suffix}",
                f"right_memory_attention.{suffix}",
            )
        elif source_key.startswith("memory_encoder."):
            suffix = source_key[len("memory_encoder."):]
            target_keys = (
                f"left_memory_encoder.{suffix}",
                f"right_memory_encoder.{suffix}",
            )
        else:
            target_keys = (source_key,)

        for target_key in target_keys:
            assert torch.equal(model_state[target_key].cpu(), expected), target_key

    check_finetune_configuration(model, finetune_mode)

    del model, checkpoint, checkpoint_args, framewise_state
    gc.collect()


def main():
    args = parse_args()
    checkpoint_path = (
        args.sam_checkpoint or args.framewise_checkpoint
    ).resolve()
    device = torch.device(args.device)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    finetune_mode = args.finetune_mode
    if finetune_mode == "auto":
        finetune_mode = (
            "decoder-memory"
            if args.sam_checkpoint is not None
            else "memory-only"
        )

    if args.sam_checkpoint is not None:
        print("Checking official SAM2 checkpoint initialization...")
        check_official_checkpoint(checkpoint_path, device, finetune_mode)
    else:
        print("Checking framewise checkpoint initialization...")
        check_framewise_checkpoint(checkpoint_path, device, finetune_mode)

    print(
        f"SAM2DualHandMemory builder and {finetune_mode} finetuning: "
        f"OK on {device}"
    )


if __name__ == "__main__":
    main()
