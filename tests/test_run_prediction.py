import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from PIL import Image

from inference import run_prediction as prediction
from inference.builder import build_model
from inference.sam2_dual_hand_video_predictor import SAM2DualHandVideoPredictor
from projects.framewise_sam2_modified.builder import create_sam2_modified_tiny, inject_sam2_modified_adapters
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory
from training.model.sam2_modified import SAM2Modified
from test_long_video_evaluation import FakeMemoryModel
from test_prediction_dataset import FrameDataset


@pytest.mark.parametrize(
    "kind,batch_size",
    [("sam2", 2), ("framewise", 2), ("memory", 1), ("multiview", 1)],
)
def test_cli_defaults(monkeypatch, kind, batch_size):
    monkeypatch.setattr(
        sys,
        "argv",
        ["prediction", "predict", "--model", kind, "--model-checkpoint", "model.pt"],
    )
    args = prediction.parse_args()
    assert args.batch_size == batch_size and args.prediction_dir_name == f"{kind}_prediction"
    assert args.prompt_mode == "mask" and args.strategy == "baseline"
    assert args.prompt_interval == 200
    assert not hasattr(args, "clip_length") and not hasattr(args, "clip_stride")
    assert args.use_point_prompt is (kind == "sam2")


def test_sam2_checkpoint_uses_original_checkpoint_loader():
    sentinel = SimpleNamespace(image_size=768)
    args = SimpleNamespace(
        model="sam2",
        model_checkpoint="sam2.pt",
        device="cpu",
        image_size=768,
    )
    with patch("inference.builder.build_sam2_modified_tiny", return_value=sentinel) as load:
        assert build_model(args) is sentinel
    load.assert_called_once_with(
        checkpoint_path="sam2.pt",
        device="cpu",
        mode="eval",
        image_size=768,
    )


@pytest.mark.parametrize(
    "kind,prompt_mode",
    [("sam2", "point"), ("framewise", "none"), ("memory", "mask"), ("multiview", "mask")],
)
def test_single_entry_parses_evaluation_outputs(kind, prompt_mode):
    args = prediction.parse_args([
        "evaluate",
        "--model",
        kind,
        "--model-checkpoint",
        f"{kind}.pt",
        "--output-dir",
        "results",
        "--save-predictions",
        "--plot-curves",
    ])
    assert args.command == "evaluate"
    assert args.prompt_mode == prompt_mode
    assert args.save_predictions and args.plot_curves


@pytest.mark.parametrize("extra", [["--batch-size", "2"], ["--batch-size", "0"], ["--clip-length", "8"], ["--clip-stride", "8"], ["--dataset", "dexycb"]])
def test_cli_rejects_invalid_memory_args(monkeypatch, extra):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prediction",
            "predict",
            "--model",
            "memory",
            "--model-checkpoint",
            "model.pt",
            *extra,
        ],
    )
    with pytest.raises(SystemExit) as error:
        prediction.parse_args()
    assert error.value.code == 2


@pytest.mark.parametrize("kind", ["framewise", "memory"])
def test_strict_checkpoint_and_framewise_outputs(tmp_path, kind):
    saved_args = dict(image_size=128, use_image_adapter=True, use_decoder_adapter=True,
                      adapter_dim=8, adapter_dropout=0.1, adapter_init_scale=1e-3)
    original = create_sam2_modified_tiny(image_size=128, model_cls=SAM2Modified if kind == "framewise" else SAM2DualHandMemory).eval()
    inject_sam2_modified_adapters(original, **{key: value for key, value in saved_args.items() if key != "image_size"})
    original.eval()
    checkpoint = {"args": saved_args, "model_state": original.state_dict()}
    path = tmp_path / "model.pt"
    torch.save(checkpoint, path)
    args = SimpleNamespace(model=kind, model_checkpoint=path, device="cpu")
    loaded = build_model(args)
    assert type(loaded) is (SAM2Modified if kind == "framewise" else SAM2DualHandVideoPredictor)
    assert not loaded.training and args.image_size == 128
    loaded_weights = loaded.state_dict()
    assert loaded_weights.keys() == checkpoint["model_state"].keys()
    assert all(torch.equal(value, loaded_weights[key]) for key, value in checkpoint["model_state"].items())
    if kind == "framewise":
        args.batch_size, args.num_workers, args.amp, args.log_interval = 2, 0, False, 50
        args.prediction_dir_name = "framewise_prediction"
        dataset = FrameDataset(3)
        for use_points in (False, True):
            args.use_point_prompt = use_points
            with patch.object(prediction, "save_prediction") as save:
                assert prediction.run_prediction(loaded, dataset, args) == 3
            with torch.inference_mode():
                reference = []
                for batch in prediction.build_loader(dataset, args):
                    output = original.forward_single_image(
                        images=batch["image"],
                        left_point_inputs=prediction.build_center_point_prompt(batch["left_mask"]) if use_points else None,
                        right_point_inputs=prediction.build_center_point_prompt(batch["right_mask"]) if use_points else None,
                        mask_inputs=None, multimask_output=False,
                    )
                    reference.extend((output["left"]["high_res_masks"][i:i + 1], output["right"]["high_res_masks"][i:i + 1]) for i in range(batch["image"].size(0)))
            assert save.call_count == 3
            for index, call in enumerate(save.call_args_list):
                assert call.kwargs["image_path"] == f"video/rgb/{index * 3}.jpg"
                assert call.kwargs["original_size"] == (131, 133)
                torch.testing.assert_close(call.kwargs["left_logits"], reference[index][0], rtol=0, atol=0)
                torch.testing.assert_close(call.kwargs["right_logits"], reference[index][1], rtol=0, atol=0)
    del loaded, loaded_weights
    checkpoint["model_state"]["unexpected_parameter"] = torch.zeros(1)
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        build_model(args)
    checkpoint["model_state"].pop("unexpected_parameter")
    checkpoint["model_state"].pop(next(iter(checkpoint["model_state"])))
    torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        build_model(args)


@pytest.mark.parametrize("prompt_mode", ["mask", "point"])
def test_memory_entry_uses_the_shared_baseline_runner(prompt_mode):
    class MemoryFrames(FrameDataset):
        def __getitem__(self, index):
            frame = super().__getitem__(index)
            frame["original_left_mask"] = frame["left_mask"].clone()
            frame["original_right_mask"] = frame["right_mask"].clone()
            frame["dataset_root"] = self.prefix
            return frame

    dataset = MemoryFrames(3, image_size=8, prefix="dataset")
    dataset.streams = {
        "dataset/sequence/cam-a": {
            "dataset_name": "dataset",
            "sample_indices": [0, 1, 2],
            "frame_numbers": [0, 1, 2],
        },
    }
    model = FakeMemoryModel()
    args = SimpleNamespace(
        model="memory",
        device="cpu",
        amp=False,
        strategy="baseline",
        prompt_mode=prompt_mode,
        prompt_interval=200,
        iou_threshold=0.5,
        correction_points=3,
        max_condition_frames=4,
        log_interval=50,
        prediction_dir_name="memory_prediction",
        output_dir=None,
    )
    with patch.object(prediction, "save_prediction") as save:
        assert prediction.run_prediction(model, dataset, args) == 3

    assert save.call_count == 3
    for hand in ("left", "right"):
        calls = [call for call in model.calls if call["hand"] == hand]
        assert [call["prompted"] for call in calls] == [True, False, False]
        assert not any(call["corrected"] for call in calls)


def test_multiview_entry_uses_fixed_prompt_strategy():
    class MultiViewFrames(FrameDataset):
        def __getitem__(self, index):
            frame = super().__getitem__(index)
            frame["original_left_mask"] = frame["left_mask"].clone()
            frame["original_right_mask"] = frame["right_mask"].clone()
            frame["dataset_root"] = self.prefix
            return frame

    dataset = MultiViewFrames(4, image_size=8, prefix="dataset")
    dataset.streams = {
        "dataset/sequence/cam-a": {
            "dataset_name": "dataset",
            "sample_indices": [0, 1],
            "frame_numbers": [0, 1],
        },
        "dataset/sequence/cam-b": {
            "dataset_name": "dataset",
            "sample_indices": [2, 3],
            "frame_numbers": [0, 1],
        },
    }
    model = FakeMemoryModel()
    model.set_multiview_fusion_enabled = lambda enabled: setattr(
        model, "multiview_fusion_enabled", enabled
    )
    args = SimpleNamespace(
        model="multiview",
        device="cpu",
        amp=False,
        num_views=2,
        strategy="fixed",
        prompt_mode="mask",
        prompt_interval=1,
        iou_threshold=0.5,
        correction_points=3,
        max_condition_frames=4,
        disable_multiview_fusion=False,
        log_interval=50,
        prediction_dir_name="multiview_prediction",
        output_dir=None,
    )

    with patch.object(prediction, "save_prediction") as save:
        assert prediction.run_prediction(model, dataset, args) == 4

    assert save.call_count == 4
    assert model.multiview_fusion_enabled
    frame_zero = [call for call in model.calls if call["frame_idx"] == 0]
    frame_one = [call for call in model.calls if call["frame_idx"] == 1]
    assert all(call["prompted"] and not call["corrected"] for call in frame_zero)
    assert all(call["prompted"] and not call["corrected"] for call in frame_one)


@pytest.mark.parametrize("folder", ["rgb_undistort", "RGB", "color", "images", "camera"])
def test_png_labels_overlap_and_output_path(tmp_path, folder):
    source = tmp_path / folder / "frame.jpg"
    expected_path = (tmp_path / folder if folder == "camera" else tmp_path) / "prediction" / "frame.png"
    assert prediction.prediction_path(str(source), "prediction") == expected_path
    prediction.save_prediction(
        image_path=str(source), original_size=(2, 3), prediction_dir_name="prediction",
        left_logits=torch.tensor([[[[-1, 2, 1], [-1, 1, 3]]]]),
        right_logits=torch.tensor([[[[-2, -1, 2], [2, 1, 2]]]]),
    )
    with Image.open(expected_path) as result:
        assert result.mode == "L"
        np.testing.assert_array_equal(np.array(result), [[0, 2, 1], [1, 1, 1]])
