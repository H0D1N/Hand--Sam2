from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from inference.dataset import build_loader
from inference.sam2_dual_hand_video_predictor import SAM2DualHandVideoPredictor
from projects.framewise_sam2_modified.builder import create_sam2_modified_tiny
from sam2.modeling.sam2_utils import get_next_point
from test_prediction_dataset import FrameDataset


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(42)
    predictor = create_sam2_modified_tiny(image_size=128, model_cls=SAM2DualHandVideoPredictor).eval()
    # 随机权重的存在性分支可能把整张输出固定为 NO_OBJ_SCORE，避免让数值对比失去意义。
    for decoder in (predictor.left_mask_decoder, predictor.right_mask_decoder):
        decoder.pred_obj_score_head.layers[-1].weight.data.zero_()
        decoder.pred_obj_score_head.layers[-1].bias.data.fill_(10)
    return predictor


def init_video(model, dataset, prompt_mode="mask"):
    args = SimpleNamespace(model="memory", batch_size=1, num_workers=0, device="cpu")
    state = model.init_state(build_loader(dataset, args))
    for hand in ("left", "right"):
        mask = state["first_frame"][f"{hand}_mask"]
        if prompt_mode == "mask":
            model.add_mask(state, hand, mask)
        else:
            coords, labels = get_next_point(gt_masks=mask.bool(), pred_masks=None, method=model.pt_sampling_for_eval)
            model.add_points(state, hand, {"point_coords": coords, "point_labels": labels})
    return state


@pytest.mark.parametrize("num_frames", [1, 3])
def test_lazy_read_shared_backbone_and_isolated_state(model, num_frames):
    dataset = FrameDataset(num_frames)
    state = init_video(model, dataset)
    assert dataset.reads == [0]
    other = init_video(model, FrameDataset(1))
    with patch.object(model, "forward", side_effect=AssertionError("不能调用整段 forward")), \
         patch.object(model, "forward_image", wraps=model.forward_image) as encode, \
         patch.object(model, "_prepare_backbone_features", wraps=model._prepare_backbone_features) as prepare, \
         patch.object(model, "track_step", wraps=model.track_step) as track:
        seen = []
        for frame_idx, outputs, info in model.propagate_in_video(state):
            seen.append(frame_idx)
            assert dataset.reads == list(range(frame_idx + 1))
            assert info == {"image_path": f"video/rgb/{frame_idx * 3}.jpg", "original_size": (131, 133)}
            assert "first_frame" not in state and state["prompts"] == {}
            assert all(out["pred_masks_high_res"].shape == (1, 1, 128, 128) for out in outputs.values())
            if frame_idx == 0:
                for hand in ("left", "right"):
                    torch.testing.assert_close(outputs[hand]["pred_masks_high_res"] > 0, other["first_frame"][f"{hand}_mask"].bool())
        assert seen == list(range(num_frames))
        assert encode.call_count == prepare.call_count == num_frames
        assert all(call.args[0].shape == (1, 3, 128, 128) for call in encode.call_args_list)
        for call in track.call_args_list:
            kwargs = call.kwargs
            assert kwargs["gt_masks"] is None and kwargs["frames_to_add_correction_pt"] == []
            assert kwargs["num_frames"] == num_frames and kwargs["run_mem_encoder"]
            if kwargs["frame_idx"] > 0:
                assert kwargs["mask_inputs"] is None and kwargs["point_inputs"] is None
    assert other["frame_idx"] == 0
    assert all(not history["cond_frame_outputs"] for history in other["output_dict"].values())
    with pytest.raises(ValueError, match="传播前"):
        model.add_mask(state, "left", torch.zeros(1, 1, 128, 128))


@pytest.mark.parametrize("prompt_mode", ["mask", "point"])
def test_matches_sequence_forward_and_ignores_later_gt(model, prompt_mode):
    dataset = FrameDataset(3)
    frames = [dataset[i] for i in range(len(dataset))]
    images = torch.stack([frame["image"] for frame in frames])[None]
    left = torch.stack([frame["left_mask"] for frame in frames])[None]
    right = torch.stack([frame["right_mask"] for frame in frames])[None]
    with torch.inference_mode():
        reference = model(images, left, right, prompt_mode, correction_frame_indices=[])
    actual = list(model.propagate_in_video(init_video(model, dataset, prompt_mode)))
    dataset.change_later_gt = True
    changed = list(model.propagate_in_video(init_video(model, dataset, prompt_mode)))
    for frame_idx, output, _ in actual:
        for hand in ("left", "right"):
            logits = output[hand]["pred_masks_high_res"]
            torch.testing.assert_close(logits, reference[frame_idx][hand]["pred_masks_high_res"], rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(logits, changed[frame_idx][1][hand]["pred_masks_high_res"], rtol=0, atol=0)
    assert actual[-1][1]["left"]["pred_masks_high_res"].std() > 0


@pytest.mark.parametrize("stride,num_maskmem,use_ptrs", [(1, 7, True), (4, 7, True), (3, 2, True), (1, 1, True), (1, 7, False), (1, 0, True)])
def test_pruned_history_matches_full_history(model, monkeypatch, stride, num_maskmem, use_ptrs):
    monkeypatch.setattr(model, "memory_temporal_stride_for_eval", stride)
    monkeypatch.setattr(model, "num_maskmem", num_maskmem)
    monkeypatch.setattr(model, "use_obj_ptrs_in_encoder", use_ptrs)
    spatial_window = 0 if num_maskmem <= 1 else 1 + (num_maskmem - 2) * stride
    num_frames = max(spatial_window, model.max_obj_ptrs_in_encoder) + 4
    pointer_window = min(num_frames, model.max_obj_ptrs_in_encoder) - 1 if num_maskmem > 0 and use_ptrs else 0
    dataset = FrameDataset(num_frames)
    with patch.object(model, "_prune_memory"):
        reference = list(model.propagate_in_video(init_video(model, dataset)))
    state = init_video(model, dataset)
    for frame_idx, output, _ in model.propagate_in_video(state):
        for hand in ("left", "right"):
            torch.testing.assert_close(output[hand]["pred_masks_high_res"], reference[frame_idx][1][hand]["pred_masks_high_res"], rtol=0, atol=0)
            history = state["output_dict"][hand]
            assert list(history["cond_frame_outputs"]) == [0]
            ordinary = history["non_cond_frame_outputs"]
            assert len(ordinary) <= max(spatial_window, pointer_window)
            assert sum("maskmem_features" in out for out in ordinary.values()) == min(frame_idx, spatial_window)
            assert sum("obj_ptr" in out for out in ordinary.values()) == min(frame_idx, pointer_window)
            for out in [*history["cond_frame_outputs"].values(), *ordinary.values()]:
                assert set(out) <= {"maskmem_features", "maskmem_pos_enc", "obj_ptr"}
                assert ("maskmem_features" in out) == ("maskmem_pos_enc" in out)
