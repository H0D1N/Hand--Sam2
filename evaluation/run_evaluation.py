"""运行 memory / multiview 的长视频提示策略评估。"""

from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path

import torch

from evaluation.dataset import LongVideoDataset, build_full_gt_frame_dataset
from evaluation.model_loader import load_evaluation_model
from evaluation.streaming import EvaluationPolicy, LongVideoEvaluator
from projects.framewise_sam2_modified.utils import (
    configure_runtime,
    dump_json,
    set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = (
    "xingyi_4-5090_oak150-100output",
    "wuwen_4-5090_release-0623-compressed",
    "tencent_4-5090_7.5",
)

CSV_FIELDS = (
    "model",
    "strategy",
    "configuration",
    "prompt_mode",
    "prompt_interval",
    "iou_threshold",
    "correction_points",
    "dataset",
    "sequence_id",
    "evaluation_id",
    "segment_index",
    "view",
    "frame_index",
    "frame_number",
    "hand",
    "prompt_type",
    "is_conditioning",
    "corrected",
    "iou_before",
    "iou_after",
    "gt_present",
    "image_path",
)

POLICY_FIELDS = (
    "model",
    "configuration",
    "strategy",
    "prompt_mode",
    "prompt_interval",
    "iou_threshold",
    "correction_points",
)

SUMMARY_FIELDS = POLICY_FIELDS + (
    "metric_start_frame",
    "total_timesteps",
    "total_hand_view_frames",
    "evaluated_hand_views",
    "foreground_hand_views",
    "mean_iou_before",
    "mean_iou_after",
    "mean_iou_gain",
    "foreground_mean_iou_before",
    "foreground_mean_iou_after",
    "ordinary_prompt_timesteps",
    "ordinary_prompt_hand_views",
    "ordinary_prompts_per_1000_hand_view_frames",
    "correction_hand_events",
    "correction_clicks",
    "correction_clicks_per_1000_hand_view_frames",
)

TEMPORAL_FIELDS = POLICY_FIELDS + (
    "frame_index",
    "contributing_sequences",
    "sequence_coverage",
    "evaluated_hand_views",
    "foreground_hand_views",
    "mean_iou_before",
    "mean_iou_after",
    "foreground_mean_iou_before",
    "foreground_mean_iou_after",
    "ordinary_prompt_timesteps",
    "ordinary_prompt_hand_views",
    "correction_hand_events",
    "correction_clicks",
)


class MetricAccumulator:
    def __init__(self, metric_start_frame: int):
        self.metric_start_frame = metric_start_frame
        self.before_sum = 0.0
        self.after_sum = 0.0
        self.count = 0
        self.foreground_before_sum = 0.0
        self.foreground_after_sum = 0.0
        self.foreground_count = 0
        self.timesteps = set()
        self.total_hand_view_frames = 0
        self.prompt_timesteps = set()
        self.ordinary_prompt_hand_views = 0
        self.correction_hand_events = set()
        self.correction_clicks = 0
        self.sequences = set()

    def update(self, row: dict) -> None:
        timestep = (row["evaluation_id"], row["frame_index"])
        self.timesteps.add(timestep)
        self.total_hand_view_frames += 1
        self.sequences.add(row["evaluation_id"])
        if row["prompt_type"] in {"mask", "point"}:
            self.prompt_timesteps.add(timestep)
            if row["prompt_type"] == "mask" or row["gt_present"]:
                self.ordinary_prompt_hand_views += 1
        if row["corrected"]:
            self.correction_hand_events.add((*timestep, row["hand"]))
            self.correction_clicks += row["correction_points"]

        if row["frame_index"] < self.metric_start_frame:
            return
        self.before_sum += row["iou_before"]
        self.after_sum += row["iou_after"]
        self.count += 1
        if row["gt_present"]:
            self.foreground_before_sum += row["iou_before"]
            self.foreground_after_sum += row["iou_after"]
            self.foreground_count += 1

    def result(self) -> dict:
        mean_before = self.before_sum / self.count if self.count else None
        mean_after = self.after_sum / self.count if self.count else None
        foreground_before = (
            self.foreground_before_sum / self.foreground_count
            if self.foreground_count else None
        )
        foreground_after = (
            self.foreground_after_sum / self.foreground_count
            if self.foreground_count else None
        )
        return {
            "metric_start_frame": self.metric_start_frame,
            "total_timesteps": len(self.timesteps),
            "total_hand_view_frames": self.total_hand_view_frames,
            "evaluated_hand_views": self.count,
            "foreground_hand_views": self.foreground_count,
            "mean_iou_before": mean_before,
            "mean_iou_after": mean_after,
            "mean_iou_gain": (
                mean_after - mean_before
                if mean_before is not None and mean_after is not None else None
            ),
            "foreground_mean_iou_before": foreground_before,
            "foreground_mean_iou_after": foreground_after,
            "ordinary_prompt_timesteps": len(self.prompt_timesteps),
            "ordinary_prompt_hand_views": self.ordinary_prompt_hand_views,
            "ordinary_prompts_per_1000_hand_view_frames": (
                1000.0 * self.ordinary_prompt_hand_views / self.total_hand_view_frames
                if self.total_hand_view_frames else None
            ),
            "correction_hand_events": len(self.correction_hand_events),
            "correction_clicks": self.correction_clicks,
            "correction_clicks_per_1000_hand_view_frames": (
                1000.0 * self.correction_clicks / self.total_hand_view_frames
                if self.total_hand_view_frames else None
            ),
        }


def policy_metadata(model_type: str, policy: EvaluationPolicy) -> dict:
    return {
        "model": model_type,
        "configuration": policy.configuration,
        "strategy": policy.strategy,
        "prompt_mode": policy.prompt_mode,
        "prompt_interval": (
            policy.prompt_interval if policy.strategy == "fixed" else None
        ),
        "iou_threshold": (
            policy.iou_threshold if policy.strategy == "adaptive" else None
        ),
        "correction_points": (
            policy.correction_points if policy.strategy == "adaptive" else None
        ),
    }


def build_policies(args) -> list[EvaluationPolicy]:
    policies = []
    seen_configurations = set()
    for strategy in args.strategies:
        if strategy == "baseline":
            candidates = [EvaluationPolicy(
                strategy="baseline",
                prompt_mode=args.prompt_mode,
                correction_points=args.correction_points,
                max_condition_frames=args.max_condition_frames,
            )]
        elif strategy == "fixed":
            candidates = [
                EvaluationPolicy(
                    strategy="fixed",
                    prompt_mode=args.prompt_mode,
                    prompt_interval=interval,
                    correction_points=args.correction_points,
                    max_condition_frames=args.max_condition_frames,
                )
                for interval in args.fixed_intervals
            ]
        else:
            candidates = [
                EvaluationPolicy(
                    strategy="adaptive",
                    prompt_mode=args.prompt_mode,
                    iou_threshold=threshold,
                    correction_points=args.correction_points,
                    max_condition_frames=args.max_condition_frames,
                )
                for threshold in args.adaptive_thresholds
            ]

        for policy in candidates:
            if policy.configuration not in seen_configurations:
                policies.append(policy)
                seen_configurations.add(policy.configuration)
    return policies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate prompt policies causally on full-GT long videos."
    )
    parser.add_argument("--model", choices=("memory", "multiview"), required=True)
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("baseline", "fixed", "adaptive"),
        default=("baseline",),
    )
    parser.add_argument("--prompt-mode", choices=("mask", "point"), default="mask")
    parser.add_argument("--fixed-intervals", type=int, nargs="+", default=(80,))
    parser.add_argument(
        "--adaptive-thresholds", type=float, nargs="+", default=(0.5,)
    )
    parser.add_argument("--correction-points", type=int, default=1)
    parser.add_argument("--max-condition-frames", type=int, default=4)

    parser.add_argument(
        "--dataset",
        dest="dataset_mode",
        choices=("multiserver", "dexycb", "mixed"),
        default="multiserver",
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--test-seq-count", type=int, default=3)
    parser.add_argument("--dex-ycb-root", type=Path)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--min-sequence-length", type=int, default=2)

    parser.add_argument("--metric-start-frame", type=int, default=1)
    parser.add_argument("--max-sequences", type=int)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.dataset_mode in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error("--dataset dexycb/mixed 需要提供 --dex-ycb-root")
    if args.model == "multiview" and args.num_views < 2:
        parser.error("multiview 模型要求 --num-views 至少为 2")
    if any(interval < 1 for interval in args.fixed_intervals):
        parser.error("--fixed-intervals 必须全部大于 0")
    if any(
        not 0.0 <= threshold <= 1.0
        for threshold in args.adaptive_thresholds
    ):
        parser.error("--adaptive-thresholds 必须全部位于 [0, 1]")
    if args.correction_points < 1:
        parser.error("--correction-points 必须大于 0")
    if args.max_condition_frames < 1:
        parser.error("--max-condition-frames 必须大于 0")
    if args.min_sequence_length < 1:
        parser.error("--min-sequence-length 必须大于 0")
    if args.metric_start_frame < 0:
        parser.error("--metric-start-frame 不能小于 0")
    if args.max_sequences is not None and args.max_sequences < 1:
        parser.error("--max-sequences 必须大于 0")
    if args.log_interval < 1:
        parser.error("--log-interval 必须大于 0")
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)

    model = load_evaluation_model(args.model, args.model_checkpoint, device)
    frame_dataset = build_full_gt_frame_dataset(
        dataset_mode=args.dataset_mode,
        image_size=model.image_size,
        dataset_root=args.dataset_root,
        dataset_names=args.dataset_names,
        test_seq_count=args.test_seq_count,
        dex_ycb_root=args.dex_ycb_root,
    )
    dataset = LongVideoDataset(
        frame_dataset,
        num_views=1 if args.model == "memory" else args.num_views,
        min_sequence_length=args.min_sequence_length,
    )
    sequences = dataset.sequences
    if args.max_sequences is not None:
        sequences = sequences[:args.max_sequences]
    if not sequences:
        raise ValueError("没有找到满足条件的连续长视频序列")

    logging.info(
        "Long-video dataset | model=%s | sequences=%d | frames=%d",
        args.model,
        len(sequences),
        sum(sequence.num_frames for sequence in sequences),
    )

    policies = build_policies(args)
    if not policies:
        raise ValueError("没有生成任何评估配置")

    overall = {}
    per_sequence = {}
    summary_rows = []
    temporal_rows = []
    csv_path = args.output_dir / "per_frame_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for policy in policies:
            evaluator = LongVideoEvaluator(
                model=model,
                model_type=args.model,
                device=device,
                policy=policy,
                amp=args.amp,
            )
            strategy_metrics = MetricAccumulator(args.metric_start_frame)
            strategy_sequences = {}
            temporal_metrics = {}

            for sequence_index, sequence in enumerate(sequences, start=1):
                rows = evaluator.evaluate_sequence(dataset, sequence)
                sequence_metrics = MetricAccumulator(args.metric_start_frame)
                for row in rows:
                    writer.writerow(row)
                    strategy_metrics.update(row)
                    sequence_metrics.update(row)
                    temporal_metrics.setdefault(
                        row["frame_index"], MetricAccumulator(metric_start_frame=0)
                    ).update(row)
                strategy_sequences[sequence.evaluation_id] = sequence_metrics.result()

                if sequence_index % args.log_interval == 0 or sequence_index == len(sequences):
                    logging.info(
                        "%s | sequence %d/%d | %s | frames=%d | mean_iou=%.4f",
                        policy.configuration,
                        sequence_index,
                        len(sequences),
                        sequence.evaluation_id,
                        sequence.num_frames,
                        sequence_metrics.result()["mean_iou_after"] or 0.0,
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            result = strategy_metrics.result()
            overall[policy.configuration] = result
            per_sequence[policy.configuration] = strategy_sequences
            metadata = policy_metadata(args.model, policy)
            summary_rows.append({**metadata, **result})
            for frame_index, frame_metrics in sorted(temporal_metrics.items()):
                frame_result = frame_metrics.result()
                temporal_rows.append({
                    **metadata,
                    "frame_index": frame_index,
                    "contributing_sequences": len(frame_metrics.sequences),
                    "sequence_coverage": len(frame_metrics.sequences) / len(sequences),
                    **{
                        key: frame_result[key]
                        for key in TEMPORAL_FIELDS
                        if key in frame_result
                    },
                })

    summary_csv_path = args.output_dir / "configuration_summary.csv"
    with summary_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(summary_rows)

    temporal_csv_path = args.output_dir / "temporal_metrics.csv"
    with temporal_csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TEMPORAL_FIELDS)
        writer.writeheader()
        writer.writerows(temporal_rows)

    summary = {
        "config": {
            "model": args.model,
            "model_checkpoint": str(args.model_checkpoint),
            "configurations": [policy.configuration for policy in policies],
            "prompt_mode": args.prompt_mode,
            "fixed_intervals": list(args.fixed_intervals),
            "adaptive_thresholds": list(args.adaptive_thresholds),
            "correction_points": args.correction_points,
            "max_condition_frames": args.max_condition_frames,
            "metric_start_frame": args.metric_start_frame,
            "num_views": 1 if args.model == "memory" else args.num_views,
            "num_sequences": len(sequences),
        },
        "overall": overall,
        "per_sequence": per_sequence,
    }
    dump_json(summary, args.output_dir / "summary.json")
    logging.info("Saved per-frame metrics: %s", csv_path)
    logging.info("Saved configuration summary: %s", summary_csv_path)
    logging.info("Saved temporal metrics: %s", temporal_csv_path)
    logging.info("Saved summary: %s", args.output_dir / "summary.json")


if __name__ == "__main__":
    main()
