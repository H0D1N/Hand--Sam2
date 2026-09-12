import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from evaluation.dataset import (
    LongVideoDataset,
    LongVideoSequence,
    NaturalValMultiServerDataset,
    natural_sort_key,
)
from evaluation.streaming import EvaluationPolicy, LongVideoEvaluator
from evaluation.run_evaluation import MetricAccumulator, build_policies


IMAGE_SIZE = 8


class FakeFrameDataset:
    def __init__(self):
        self.streams = {
            "dataset/sequence/cam-a": {
                "dataset_name": "dataset",
                "sample_indices": [0, 1, 2, 3, 4],
                "frame_numbers": [1, 2, 4, 5, 6],
            },
            "dataset/sequence/cam-b": {
                "dataset_name": "dataset",
                "sample_indices": [5, 6, 7, 8, 9],
                "frame_numbers": [1, 2, 3, 4, 5],
            },
        }

    def __getitem__(self, index):
        mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        mask[:, 2:6, 2:6] = 1
        return {
            "image": torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE),
            "left_mask": mask.clone(),
            "right_mask": mask.clone(),
            "original_left_mask": mask.clone(),
            "original_right_mask": mask.clone(),
            "image_path": f"{index}.png",
        }


class FakeMemoryModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.image_size = IMAGE_SIZE
        self.num_maskmem = 3
        self.memory_temporal_stride_for_eval = 1
        self.max_obj_ptrs_in_encoder = 3
        self.use_obj_ptrs_in_encoder = True
        self.calls = []

    def forward_image(self, images):
        features = [images[:, :1]]
        return {
            "left_high_res_features": features,
            "right_high_res_features": features,
        }

    def _prepare_backbone_features(self, backbone_out):
        num_views = backbone_out["left_high_res_features"][0].shape[0]
        feature = torch.zeros(1, num_views, 1)
        return backbone_out, [feature], [feature], [(1, 1)]

    def track_step(
        self,
        *,
        hand,
        frame_idx,
        mask_inputs,
        run_mem_encoder,
        frames_to_add_correction_pt,
        gt_masks,
        **kwargs,
    ):
        corrected = frame_idx in frames_to_add_correction_pt
        self.calls.append({
            "hand": hand,
            "frame_idx": frame_idx,
            "prompted": mask_inputs is not None,
            "corrected": corrected,
            "run_mem_encoder": run_mem_encoder,
            "num_views": kwargs.get("num_views"),
        })
        if mask_inputs is not None or corrected:
            logits = gt_masks * 20.0 - 10.0
        else:
            logits = torch.full_like(gt_masks, -10.0)
            if logits.shape[0] > 1:
                logits[0] = gt_masks[0] * 20.0 - 10.0
        return {
            "pred_masks_high_res": logits,
            "maskmem_features": torch.zeros(gt_masks.shape[0], 1, 1, 1),
            "maskmem_pos_enc": [torch.zeros(gt_masks.shape[0], 1, 1, 1)],
            "obj_ptr": torch.zeros(gt_masks.shape[0], 1),
        }


class ThreeFrameDataset:
    def __init__(self, num_views=1):
        self.num_views = num_views

    def load_frame(self, sequence, frame_index):
        mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        mask[:, 2:6, 2:6] = 1
        return {
            "image": torch.zeros(self.num_views, 3, IMAGE_SIZE, IMAGE_SIZE),
            "left_mask": mask.unsqueeze(0).repeat(self.num_views, 1, 1, 1),
            "right_mask": mask.unsqueeze(0).repeat(self.num_views, 1, 1, 1),
            "original_left_mask": [mask.clone() for _ in range(self.num_views)],
            "original_right_mask": [mask.clone() for _ in range(self.num_views)],
            "image_path": [
                f"view-{view_index}-{frame_index}.png"
                for view_index in range(self.num_views)
            ],
        }


def build_three_frame_sequence():
    return LongVideoSequence(
        dataset_name="dataset",
        sequence_id="dataset/sequence",
        segment_index=0,
        view_names=("cam-a",),
        frame_numbers=(1, 2, 3),
        sample_indices=((0, 1, 2),),
    )


def check_long_video_alignment_and_gap_splitting():
    names = ["06", "07", "08", "09", "010"]
    assert sorted(names, key=natural_sort_key)[-3:] == ["08", "09", "010"]

    single_view = LongVideoDataset(
        FakeFrameDataset(), num_views=1, min_sequence_length=2
    )
    assert [sequence.frame_numbers for sequence in single_view.sequences] == [
        (1, 2),
        (4, 5, 6),
        (1, 2, 3, 4, 5),
    ]

    multiview = LongVideoDataset(
        FakeFrameDataset(), num_views=2, min_sequence_length=2
    )
    assert [sequence.frame_numbers for sequence in multiview.sequences] == [
        (1, 2),
        (4, 5),
    ]
    first_frame = multiview.load_frame(multiview.sequences[0], 0)
    assert first_frame["image"].shape == (2, 3, IMAGE_SIZE, IMAGE_SIZE)


def check_natural_validation_sequence_selection():
    with tempfile.TemporaryDirectory() as temporary_directory:
        root = Path(temporary_directory)
        config = {
            "dataset_name": "dataset",
            "data_root": "sources/dataset",
            "sequence_glob": "*",
            "view_glob": "cam-*",
            "rgb_dir": "{view}/rgb",
            "mask_dir": "{view}/mask",
            "image_extensions": [".png"],
            "mask_extensions": [".png"],
            "mask_values": {"left": [1], "right": [2]},
        }
        (root / "datasets.json").write_text(
            json.dumps({"datasets": [config]}), encoding="utf-8"
        )
        for sequence_name in ("06", "07", "08", "09", "010"):
            rgb_dir = root / "sources/dataset" / sequence_name / "cam-a/rgb"
            mask_dir = root / "sources/dataset" / sequence_name / "cam-a/mask"
            rgb_dir.mkdir(parents=True)
            mask_dir.mkdir(parents=True)
            Image.fromarray(np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)).save(
                rgb_dir / "1.png"
            )
            Image.fromarray(np.ones((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)).save(
                mask_dir / "1.png"
            )

        dataset = NaturalValMultiServerDataset(
            dataset_root=root,
            test_seq_count=3,
            image_size=IMAGE_SIZE,
            dataset_names=["dataset"],
        )
        assert list(dataset.streams) == [
            "dataset/08/cam-a",
            "dataset/09/cam-a",
            "dataset/010/cam-a",
        ]
        assert dataset[0]["original_left_mask"].all()


def check_adaptive_correction_is_decided_before_memory_commit():
    model = FakeMemoryModel()
    evaluator = LongVideoEvaluator(
        model=model,
        model_type="memory",
        device="cpu",
        policy=EvaluationPolicy(
            strategy="adaptive",
            iou_threshold=0.5,
            correction_points=1,
        ),
    )
    rows = evaluator.evaluate_sequence(
        ThreeFrameDataset(), build_three_frame_sequence()
    )

    frame_one_rows = [row for row in rows if row["frame_index"] == 1]
    assert all(row["corrected"] for row in frame_one_rows)
    assert all(row["iou_before"] < 0.01 for row in frame_one_rows)
    assert all(row["iou_after"] == 1.0 for row in frame_one_rows)

    frame_one_calls = [call for call in model.calls if call["frame_idx"] == 1]
    for hand in ("left", "right"):
        hand_calls = [call for call in frame_one_calls if call["hand"] == hand]
        assert len(hand_calls) == 2
        assert not hand_calls[0]["run_mem_encoder"]
        assert hand_calls[1]["run_mem_encoder"]
        assert hand_calls[1]["corrected"]


def check_fixed_prompts_repeat_the_initial_prompt():
    evaluator = LongVideoEvaluator(
        model=FakeMemoryModel(),
        model_type="memory",
        device="cpu",
        policy=EvaluationPolicy(
            strategy="fixed",
            prompt_mode="mask",
            prompt_interval=2,
        ),
    )
    rows = evaluator.evaluate_sequence(
        ThreeFrameDataset(), build_three_frame_sequence()
    )
    left_rows = [row for row in rows if row["hand"] == "left"]
    assert [row["prompt_type"] for row in left_rows] == ["mask", "none", "mask"]
    assert [row["is_conditioning"] for row in left_rows] == [True, False, True]


def check_multiview_adaptive_trigger_uses_all_views():
    model = FakeMemoryModel()
    evaluator = LongVideoEvaluator(
        model=model,
        model_type="multiview",
        device="cpu",
        policy=EvaluationPolicy(strategy="adaptive", iou_threshold=0.5),
    )
    sequence = LongVideoSequence(
        dataset_name="dataset",
        sequence_id="dataset/sequence",
        segment_index=0,
        view_names=("cam-a", "cam-b"),
        frame_numbers=(1, 2, 3),
        sample_indices=((0, 1, 2), (3, 4, 5)),
    )
    rows = evaluator.evaluate_sequence(ThreeFrameDataset(num_views=2), sequence)

    frame_one_rows = [row for row in rows if row["frame_index"] == 1]
    assert len(frame_one_rows) == 4
    assert all(row["corrected"] for row in frame_one_rows)
    assert any(row["iou_before"] < 0.01 for row in frame_one_rows)
    assert any(row["iou_before"] == 1.0 for row in frame_one_rows)
    assert all(row["iou_after"] == 1.0 for row in frame_one_rows)
    assert all(
        call["num_views"] == 2
        for call in model.calls
        if call["frame_idx"] == 1
    )
    metrics = MetricAccumulator(metric_start_frame=1)
    for row in rows:
        metrics.update(row)
    result = metrics.result()
    assert result["total_timesteps"] == 3
    assert result["total_hand_view_frames"] == 12
    assert result["ordinary_prompt_hand_views"] == 4
    assert result["correction_hand_events"] == 4
    assert result["correction_clicks"] == 8
    metrics = MetricAccumulator(metric_start_frame=1)
    for row in rows:
        metrics.update(row)
    result = metrics.result()
    assert result["ordinary_prompt_timesteps"] == 1
    assert result["ordinary_prompt_hand_views"] == 4
    assert result["correction_hand_events"] == 4
    assert result["correction_clicks"] == 8


def check_strategy_parameter_sweep():
    policies = build_policies(SimpleNamespace(
        strategies=("baseline", "fixed", "adaptive"),
        prompt_mode="mask",
        fixed_intervals=(20, 80, 20),
        adaptive_thresholds=(0.3, 0.5, 0.3),
        correction_points=2,
        max_condition_frames=4,
    ))
    assert [policy.configuration for policy in policies] == [
        "baseline",
        "fixed_interval_20",
        "fixed_interval_80",
        "adaptive_iou_0p3",
        "adaptive_iou_0p5",
    ]
    assert all(policy.prompt_mode == "mask" for policy in policies)


def main():
    check_long_video_alignment_and_gap_splitting()
    check_natural_validation_sequence_selection()
    check_adaptive_correction_is_decided_before_memory_commit()
    check_fixed_prompts_repeat_the_initial_prompt()
    check_multiview_adaptive_trigger_uses_all_views()
    check_strategy_parameter_sweep()
    print("Long-video evaluation: OK")


if __name__ == "__main__":
    main()
