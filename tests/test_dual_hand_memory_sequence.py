import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.framewise_sam2_modified import create_sam2_modified_tiny
from projects.dual_hand_memory.losses import DualHandMemoryLoss
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory


IMAGE_SIZE = 128
MULTISTEP_OUTPUT_KEYS = {
    "multistep_pred_masks",
    "multistep_pred_masks_high_res",
    "multistep_pred_multimasks",
    "multistep_pred_multimasks_high_res",
    "multistep_pred_ious",
    "multistep_point_inputs",
    "multistep_object_score_logits",
}
OUTPUT_KEYS = MULTISTEP_OUTPUT_KEYS | {
    "pred_masks",
    "pred_masks_high_res",
    "maskmem_features",
    "maskmem_pos_enc",
}
PREDICTION_KEYS = {
    "pred_masks",
    "pred_masks_high_res",
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
            assert hand_outputs["pred_masks"].shape == (
                1,
                1,
                IMAGE_SIZE // 4,
                IMAGE_SIZE // 4,
            )
            assert hand_outputs["pred_masks_high_res"].shape == (
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


def run_with_memory_counters(
    model,
    images,
    left_masks,
    right_masks,
    prompt_mode,
):
    calls, handles = register_memory_counters(model)
    try:
        with torch.inference_mode():
            outputs = model(
                images,
                left_masks,
                right_masks,
                prompt_mode=prompt_mode,
            )
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
        prompt_mode="mask",
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
        prompt_mode="mask",
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
        assert torch.equal(
            first_outputs[0][hand]["pred_masks_high_res"],
            expected_logits,
        )

    with torch.inference_mode():
        second_outputs = model(
            images,
            left_masks,
            right_masks,
            prompt_mode="mask",
        )

    for frame_idx in range(2):
        for hand in ("left", "right"):
            for key in PREDICTION_KEYS:
                assert torch.equal(
                    first_outputs[frame_idx][hand][key],
                    second_outputs[frame_idx][hand][key],
                ), f"Memory bank leaked: frame={frame_idx}, {hand}.{key}"


def register_decoder_counters(model):
    calls = {"left": 0, "right": 0}
    handles = []

    for hand in calls:
        def count_call(_module, _inputs, _output, hand_name=hand):
            calls[hand_name] += 1

        decoder = getattr(model, f"{hand}_mask_decoder")
        handles.append(decoder.register_forward_hook(count_call))

    return calls, handles


def check_correction_point_sampling(model, images, left_masks, right_masks):
    model.eval()
    original_num_correction_points = model.num_correction_pt_per_frame
    model.num_correction_pt_per_frame = 2
    calls, handles = register_decoder_counters(model)

    try:
        with torch.inference_mode():
            outputs = model(
                images,
                left_masks,
                right_masks,
                prompt_mode="point",
            )
    finally:
        model.num_correction_pt_per_frame = original_num_correction_points
        for handle in handles:
            handle.remove()

    # t=0: one initial prediction + two correction clicks; t=1: tracking once.
    assert calls == {"left": 4, "right": 4}

    for hand in ("left", "right"):
        corrected_output = outputs[0][hand]
        assert MULTISTEP_OUTPUT_KEYS <= set(corrected_output)
        assert corrected_output["multistep_pred_masks"].shape == (
            1,
            3,
            IMAGE_SIZE // 4,
            IMAGE_SIZE // 4,
        )
        assert corrected_output["multistep_pred_masks_high_res"].shape == (
            1,
            3,
            IMAGE_SIZE,
            IMAGE_SIZE,
        )
        assert len(corrected_output["multistep_pred_multimasks_high_res"]) == 3
        assert len(corrected_output["multistep_pred_ious"]) == 3
        assert len(corrected_output["multistep_object_score_logits"]) == 3

        point_steps = corrected_output["multistep_point_inputs"]
        assert len(point_steps) == 3
        assert [step["point_coords"].shape for step in point_steps] == [
            (1, 1, 2),
            (1, 2, 2),
            (1, 3, 2),
        ]
        assert [step["point_labels"].shape for step in point_steps] == [
            (1, 1),
            (1, 2),
            (1, 3),
        ]

        assert torch.equal(
            corrected_output["pred_masks"],
            corrected_output["multistep_pred_masks"][:, -1:],
        )
        assert torch.equal(
            corrected_output["pred_masks_high_res"],
            corrected_output["multistep_pred_masks_high_res"][:, -1:],
        )


def check_zero_correction_points(model, images, left_masks, right_masks):
    model.eval()
    original_num_correction_points = model.num_correction_pt_per_frame
    model.num_correction_pt_per_frame = 0
    calls, handles = register_decoder_counters(model)

    try:
        with torch.inference_mode():
            outputs = model(
                images,
                left_masks,
                right_masks,
                prompt_mode="point",
            )
    finally:
        model.num_correction_pt_per_frame = original_num_correction_points
        for handle in handles:
            handle.remove()

    # No correction clicks: each frame only runs its initial prediction.
    assert calls == {"left": 2, "right": 2}

    for hand in ("left", "right"):
        output = outputs[0][hand]
        assert output["multistep_pred_masks"].shape[1] == 1
        assert output["multistep_pred_masks_high_res"].shape[1] == 1
        assert len(output["multistep_pred_multimasks_high_res"]) == 1
        assert len(output["multistep_pred_ious"]) == 1
        assert len(output["multistep_object_score_logits"]) == 1
        assert len(output["multistep_point_inputs"]) == 1
        assert torch.equal(
            output["pred_masks"],
            output["multistep_pred_masks"],
        )
        assert torch.equal(
            output["pred_masks_high_res"],
            output["multistep_pred_masks_high_res"],
        )


def check_corrected_frame_becomes_conditioning_memory(
    model,
    images,
    left_masks,
    right_masks,
):
    model.eval()
    original_num_correction_points = model.num_correction_pt_per_frame
    original_add_as_cond = model.add_all_frames_to_correct_as_cond
    model.num_correction_pt_per_frame = 1
    model.add_all_frames_to_correct_as_cond = True

    try:
        flat_images = images.transpose(0, 1).flatten(0, 1)
        backbone_out = model.forward_image(flat_images)
        backbone_out["batch_size"] = images.size(0)
        backbone_out["num_frames"] = images.size(1)
        backbone_out = model.prepare_prompt_inputs(
            backbone_out,
            left_masks,
            right_masks,
            prompt_mode="point",
        )

        # t=1 is not an initial conditioning frame, but receives a correction.
        assert backbone_out["init_cond_frames"] == [0]
        backbone_out["frames_to_add_correction_pt"] = [0, 1]

        with torch.inference_mode():
            output_dict = model.forward_tracking(backbone_out, return_dict=True)
    finally:
        model.num_correction_pt_per_frame = original_num_correction_points
        model.add_all_frames_to_correct_as_cond = original_add_as_cond

    for hand in ("left", "right"):
        assert 1 in output_dict[hand]["cond_frame_outputs"]
        assert 1 not in output_dict[hand]["non_cond_frame_outputs"]
        assert "obj_ptr" in output_dict[hand]["cond_frame_outputs"][1]


def check_memory_gradients(model, images, left_masks, right_masks):
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(
        images,
        left_masks,
        right_masks,
        prompt_mode="point",
    )
    loss, loss_details = DualHandMemoryLoss()(
        outputs,
        left_masks,
        right_masks,
    )
    assert torch.isfinite(loss).item()
    assert all(
        torch.isfinite(value).item()
        for hand_details in loss_details.values()
        for value in hand_details.values()
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


def check_prompt_modes(model, images, left_masks, right_masks):
    backbone_out = model.forward_image(images[:, 0])
    backbone_out["batch_size"] = images.size(0)
    backbone_out["num_frames"] = images.size(1)

    point_inputs = model.prepare_prompt_inputs(
        backbone_out.copy(), left_masks, right_masks,
        prompt_mode="point",
    )
    assert point_inputs["use_pt_input"]
    assert set(point_inputs["point_inputs_per_frame"]) == {0}
    assert point_inputs["mask_inputs_per_frame"] == {}

    mask_inputs = model.prepare_prompt_inputs(
        backbone_out.copy(), left_masks, right_masks,
        prompt_mode="mask",
    )
    assert not mask_inputs["use_pt_input"]
    assert mask_inputs["point_inputs_per_frame"] == {}
    assert set(mask_inputs["mask_inputs_per_frame"]) == {0}

    backbone_out["num_frames"] = 1
    single_frame_inputs = model.prepare_prompt_inputs(
        backbone_out.copy(), left_masks[:, :1], right_masks[:, :1],
        prompt_mode="mask",
    )
    assert single_frame_inputs["use_pt_input"]
    assert set(single_frame_inputs["point_inputs_per_frame"]) == {0}
    assert single_frame_inputs["mask_inputs_per_frame"] == {}


def main():
    torch.manual_seed(0)
    model = build_test_model()
    images, left_masks, right_masks = build_test_inputs()

    check_prompt_modes(model, images, left_masks, right_masks)
    check_t1_output(model, images, left_masks, right_masks)
    check_t2_memory_tracking(model, images, left_masks, right_masks)
    check_correction_point_sampling(model, images, left_masks, right_masks)
    check_zero_correction_points(model, images, left_masks, right_masks)
    check_corrected_frame_becomes_conditioning_memory(
        model,
        images,
        left_masks,
        right_masks,
    )
    check_memory_gradients(model, images, left_masks, right_masks)
    print("SAM2DualHandMemory sequence forward: OK")


if __name__ == "__main__":
    main()
