import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.framewise_sam2_modified import create_sam2_modified_tiny
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory


def check_independent_modules(left_module, right_module):
    left_parameters = dict(left_module.named_parameters())
    right_parameters = dict(right_module.named_parameters())

    assert left_module is not right_module
    assert left_parameters.keys() == right_parameters.keys()

    for parameter_name, left_parameter in left_parameters.items():
        right_parameter = right_parameters[parameter_name]

        assert torch.equal(left_parameter, right_parameter), parameter_name
        assert left_parameter.data_ptr() != right_parameter.data_ptr(), parameter_name


def check_model_structure(model):
    assert isinstance(model, SAM2DualHandMemory)

    assert not hasattr(model, "memory_attention")
    assert not hasattr(model, "memory_encoder")

    assert hasattr(model, "left_memory_attention")
    assert hasattr(model, "right_memory_attention")
    assert hasattr(model, "left_memory_encoder")
    assert hasattr(model, "right_memory_encoder")

    assert hasattr(model, "left_mask_decoder")
    assert hasattr(model, "right_mask_decoder")
    assert not hasattr(model, "sam_mask_decoder")

    check_independent_modules(
        model.left_memory_attention,
        model.right_memory_attention,
    )
    check_independent_modules(
        model.left_memory_encoder,
        model.right_memory_encoder,
    )


def check_state_dict(model):
    state_keys = model.state_dict().keys()
    expected_counts = {
        "memory_attention.": 0,
        "memory_encoder.": 0,
        "left_memory_attention.": 106,
        "right_memory_attention.": 106,
        "left_memory_encoder.": 40,
        "right_memory_encoder.": 40,
    }

    for prefix, expected_count in expected_counts.items():
        actual_count = sum(key.startswith(prefix) for key in state_keys)
        assert actual_count == expected_count, (
            f"{prefix}: expected {expected_count} keys, got {actual_count}"
        )


def check_single_image_forward(model):
    image_size = model.image_size
    images = torch.randn(1, 3, image_size, image_size)

    with torch.inference_mode():
        outputs = model.forward_single_image(
            images=images,
            mask_inputs=None,
            multimask_output=False,
        )

    assert set(outputs) == {"left", "right"}

    for hand_name in ("left", "right"):
        hand_outputs = outputs[hand_name]
        assert hand_outputs["high_res_masks"].shape == (
            1,
            1,
            image_size,
            image_size,
        )
        assert torch.isfinite(hand_outputs["high_res_masks"]).all().item()


def main():
    torch.manual_seed(0)

    model = create_sam2_modified_tiny(
        image_size=768,
        model_cls=SAM2DualHandMemory,
    )
    model.eval()

    check_model_structure(model)
    check_state_dict(model)
    check_single_image_forward(model)

    print("SAM2DualHandMemory structure and single-image forward: OK")


if __name__ == "__main__":
    main()
