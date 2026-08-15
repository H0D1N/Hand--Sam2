import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.framewise_sam2_modified import create_sam2_modified_tiny
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory


IMAGE_SIZE = 128
OUTPUT_KEYS = {
    "high_res_multimasks",
    "ious",
    "high_res_masks",
    "object_score_logits",
    "maskmem_features",
    "maskmem_pos_enc",
}
PREDICTION_KEYS = {
    "high_res_multimasks",
    "ious",
    "high_res_masks",
    "object_score_logits",
}


def build_test_model():
    model = create_sam2_modified_tiny(
        image_size=IMAGE_SIZE,
        model_cls=SAM2DualHandMemory,
    )
    model.eval()
    return model


def build_test_inputs(num_frames=2):
    images = torch.randn(1, num_frames, 3, IMAGE_SIZE, IMAGE_SIZE)
    left_masks = torch.zeros(1, num_frames, 1, IMAGE_SIZE, IMAGE_SIZE)
    right_masks = torch.zeros_like(left_masks)
    left_masks[:, :, :, 20:60, 15:50] = 1
    right_masks[:, :, :, 50:100, 70:115] = 1
    return images, left_masks, right_masks


def check_output_structure(outputs, num_frames):
    assert isinstance(outputs, list)
    assert len(outputs) == num_frames

    for frame_outputs in outputs:
        assert set(frame_outputs) == {"left", "right"}
        for hand_outputs in frame_outputs.values():
            assert set(hand_outputs) == OUTPUT_KEYS
            assert hand_outputs["high_res_masks"].shape == (
                1,
                1,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )
            assert hand_outputs["maskmem_features"].shape == (1, 64, 8, 8)
            assert len(hand_outputs["maskmem_pos_enc"]) == 1
            assert hand_outputs["maskmem_pos_enc"][0].shape == (1, 64, 8, 8)


def register_memory_counters(model):
    calls = {name: 0 for name in (
        "left_memory_attention",
        "right_memory_attention",
        "left_memory_encoder",
        "right_memory_encoder",
    )}
    handles = []

    for name in calls:
        def count_call(_module, _inputs, _output, module_name=name):
            calls[module_name] += 1

        handles.append(getattr(model, name).register_forward_hook(count_call))

    return calls, handles


def run_with_memory_counters(model, images, left_masks, right_masks):
    calls, handles = register_memory_counters(model)
    try:
        with torch.inference_mode():
            outputs = model.forward_sequence(images, left_masks, right_masks)
    finally:
        for handle in handles:
            handle.remove()
    return outputs, calls


def check_t1_output(model, images, left_masks, right_masks):
    outputs, calls = run_with_memory_counters(
        model,
        images[:, :1],
        left_masks[:, :1],
        right_masks[:, :1],
    )
    check_output_structure(outputs, num_frames=1)
    assert calls == {
        "left_memory_attention": 0,
        "right_memory_attention": 0,
        "left_memory_encoder": 1,
        "right_memory_encoder": 1,
    }


def check_t2_memory_tracking(model, images, left_masks, right_masks):
    first_outputs, calls = run_with_memory_counters(
        model,
        images,
        left_masks,
        right_masks,
    )
    check_output_structure(first_outputs, num_frames=2)
    assert calls == {
        "left_memory_attention": 1,
        "right_memory_attention": 1,
        "left_memory_encoder": 2,
        "right_memory_encoder": 2,
    }

    for hand, masks in (("left", left_masks), ("right", right_masks)):
        expected_logits = masks[:, 0].float() * 20.0 - 10.0
        assert torch.equal(first_outputs[0][hand]["high_res_masks"], expected_logits)

    with torch.inference_mode():
        second_outputs = model.forward_sequence(images, left_masks, right_masks)

    for frame_idx in range(2):
        for hand in ("left", "right"):
            for key in PREDICTION_KEYS:
                assert torch.equal(
                    first_outputs[frame_idx][hand][key],
                    second_outputs[frame_idx][hand][key],
                ), f"Memory bank leaked: frame={frame_idx}, {hand}.{key}"


def check_memory_gradients(model, images, left_masks, right_masks):
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model.forward_sequence(images, left_masks, right_masks)
    tracking_outputs = outputs[1]
    loss = sum(
        tracking_outputs[hand]["high_res_masks"].mean()
        + tracking_outputs[hand]["ious"].mean()
        + tracking_outputs[hand]["object_score_logits"].mean()
        for hand in ("left", "right")
    )
    loss.backward()

    for name in (
        "left_memory_attention",
        "right_memory_attention",
        "left_memory_encoder",
        "right_memory_encoder",
    ):
        gradients = [
            parameter.grad
            for parameter in getattr(model, name).parameters()
            if parameter.grad is not None
        ]
        assert gradients, f"No gradient reached {name}"
        assert any(gradient.abs().sum().item() > 0 for gradient in gradients), (
            f"Only zero gradients reached {name}"
        )


def main():
    torch.manual_seed(0)
    model = build_test_model()
    images, left_masks, right_masks = build_test_inputs()

    check_t1_output(model, images, left_masks, right_masks)
    check_t2_memory_tracking(model, images, left_masks, right_masks)
    check_memory_gradients(model, images, left_masks, right_masks)
    print("SAM2DualHandMemory sequence forward: OK")


if __name__ == "__main__":
    main()
