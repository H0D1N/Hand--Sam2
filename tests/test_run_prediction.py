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
from test_prediction_dataset import FrameDataset


@pytest.mark.parametrize("kind,batch_size", [("framewise", 2), ("memory", 1)])
def test_cli_defaults(monkeypatch, kind, batch_size):
    monkeypatch.setattr(sys, "argv", ["prediction", "--model", kind, "--model-checkpoint", "model.pt"])
    args = prediction.parse_args()
    assert args.batch_size == batch_size and args.prediction_dir_name == f"{kind}_prediction"
    assert args.prompt_mode == "mask" and not hasattr(args, "clip_length") and not hasattr(args, "clip_stride")


@pytest.mark.parametrize("extra", [["--batch-size", "2"], ["--batch-size", "0"], ["--clip-length", "8"], ["--clip-stride", "8"], ["--dataset", "dexycb"]])
def test_cli_rejects_invalid_memory_args(monkeypatch, extra):
    monkeypatch.setattr(sys, "argv", ["prediction", "--model", "memory", "--model-checkpoint", "model.pt", *extra])
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


@pytest.mark.parametrize("prompt_mode", ["mask", "auto", "point"])
def test_memory_entry_saves_all_stream_frames(tmp_path, prompt_mode):
    model = create_sam2_modified_tiny(image_size=128, model_cls=SAM2DualHandVideoPredictor).eval()
    dataset = FrameDataset(4, prefix=str(tmp_path / "video"))
    dataset.streams = {"first": {"sample_indices": [2, 0, 3]}, "second": {"sample_indices": [1]}}
    args = SimpleNamespace(model="memory", batch_size=1, num_workers=0, device="cpu", amp=False,
                           prompt_mode=prompt_mode, log_interval=2, prediction_dir_name="memory_prediction")
    with patch.object(model, "track_step", wraps=model.track_step) as track, \
         patch.object(prediction, "save_prediction", wraps=prediction.save_prediction) as save:
        assert prediction.run_prediction(model, dataset, args) == 4
    assert dataset.reads == [2, 0, 3, 1]
    assert [call.kwargs["image_path"] for call in save.call_args_list] == [f"{dataset.prefix}/rgb/{i * 3}.jpg" for i in [2, 0, 3, 1]]
    assert [call.kwargs["frame_idx"] for call in track.call_args_list] == [0, 0, 1, 1, 2, 2, 0, 0]
    for call in track.call_args_list:
        kwargs = call.kwargs
        assert kwargs["gt_masks"] is None and kwargs["frames_to_add_correction_pt"] == []
        if kwargs["frame_idx"] == 0:
            if prompt_mode == "point":
                assert kwargs["mask_inputs"] is None and kwargs["point_inputs"]["point_coords"].shape == (1, 1, 2)
            else:
                assert kwargs["point_inputs"] is None and kwargs["mask_inputs"].shape == (1, 1, 128, 128)
        else:
            assert kwargs["point_inputs"] is None and kwargs["mask_inputs"] is None
    for index in range(4):
        with Image.open(tmp_path / "video" / "memory_prediction" / f"{index * 3}.png") as result:
            assert result.size == (133, 131) and result.mode == "L"
            assert set(np.unique(result)) <= {0, 1, 2}


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
        np.testing.assert_array_equal(np.array(result), [[0, 1, 2], [2, 1, 1]])
