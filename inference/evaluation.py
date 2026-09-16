"""运行 memory / multiview 的长视频提示策略评估。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import logging
from pathlib import Path

import torch

from inference.long_video_dataset import LongVideoDataset, build_full_gt_frame_dataset
from inference.builder import load_trained_model
from inference.output import save_prediction
from inference.streaming import (
    EvaluationPolicy,
    FramewiseLongVideoEvaluator,
    LongVideoEvaluator,
)
from projects.framewise_sam2_modified.builder import build_sam2_modified_tiny
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
    "correction_clicks",
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
            self.correction_clicks += row["correction_clicks"]

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


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        choices=("sam2", "framewise", "memory", "multiview"),
        required=True,
    )
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="仅原始 SAM2 checkpoint 需要。",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=("baseline", "fixed", "adaptive"),
        default=("baseline",),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("none", "mask", "point"),
        help="默认：SAM2=point，Framewise=none，Memory/MultiView=mask。",
    )
    parser.add_argument("--fixed-intervals", type=int, nargs="+", default=(80,))
    parser.add_argument(
        "--adaptive-thresholds", type=float, nargs="+", default=(0.5,)
    )
    parser.add_argument(
        "--correction-points",
        type=int,
        default=10,
        help="单次 adaptive 纠错允许的最大点击轮数",
    )
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
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="每处理多少帧输出一次进度",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="同时把每个评估配置的最终预测保存为 PNG。",
    )
    parser.add_argument(
        "--plot-curves",
        action="store_true",
        help="评估完成后立即生成 IoU PNG 曲线。",
    )
    parser.add_argument("--min-sequence-coverage", type=float, default=0.5)
    parser.add_argument("--plot-dpi", type=int, default=200)


def validate_evaluation_args(args, parser: argparse.ArgumentParser) -> None:
    if args.dataset_mode in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error("--dataset dexycb/mixed 需要提供 --dex-ycb-root")
    if args.model == "multiview" and args.num_views < 2:
        parser.error("multiview 模型要求 --num-views 至少为 2")
    if args.prompt_mode is None:
        args.prompt_mode = {
            "sam2": "point",
            "framewise": "none",
            "memory": "mask",
            "multiview": "mask",
        }[args.model]
    if args.model == "sam2" and args.prompt_mode != "point":
        parser.error("sam2 评估需要 --prompt-mode point")
    if args.model == "framewise" and args.prompt_mode not in {"none", "point"}:
        parser.error("framewise 评估只支持 --prompt-mode none/point")
    if args.model in {"memory", "multiview"} and args.prompt_mode == "none":
        parser.error("memory/multiview 评估需要 mask 或 point Prompt")
    if args.model in {"sam2", "framewise"} and tuple(args.strategies) != (
        "baseline",
    ):
        parser.error("sam2/framewise 只支持 baseline 逐帧评估")
    if args.image_size <= 0 or args.image_size % 16 != 0:
        parser.error("--image-size 必须为 16 的正整数倍")
    if any(interval < 1 for interval in args.fixed_intervals):
        parser.error("--fixed-intervals 必须全部大于 0")
    if any(
        not 0.0 <= threshold <= 1.0
        for threshold in args.adaptive_thresholds
    ):
        parser.error("--adaptive-thresholds 必须全部位于 [0, 1]")
    if args.correction_points < 1:
        parser.error("--correction-points 最大点击轮数必须大于 0")
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
    if not 0.0 <= args.min_sequence_coverage <= 1.0:
        parser.error("--min-sequence-coverage 必须位于 [0, 1]")
    if args.plot_dpi < 1:
        parser.error("--plot-dpi 必须大于 0")


def run_evaluation(args) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)

    if args.model == "sam2":
        model = build_sam2_modified_tiny(
            checkpoint_path=args.model_checkpoint,
            device=device,
            mode="eval",
            image_size=args.image_size,
        )
    else:
        model = load_trained_model(args.model, args.model_checkpoint, device)
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
        num_views=args.num_views if args.model == "multiview" else 1,
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

    view_sets_by_sequence = defaultdict(set)
    for sequence in sequences:
        view_sets_by_sequence[sequence.sequence_id].add(sequence.view_names)

    overall = {}
    per_sequence = {}
    summary_rows = []
    temporal_rows = []
    saved_predictions = 0
    csv_path = args.output_dir / "per_frame_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()

        for policy in policies:
            if args.model in {"sam2", "framewise"}:
                evaluator = FramewiseLongVideoEvaluator(
                    model=model,
                    model_type=args.model,
                    device=device,
                    prompt_mode=args.prompt_mode,
                    amp=args.amp,
                    log_interval=args.log_interval,
                )
            else:
                evaluator = LongVideoEvaluator(
                    model=model,
                    model_type=args.model,
                    device=device,
                    policy=policy,
                    amp=args.amp,
                    log_interval=args.log_interval,
                )
            strategy_metrics = MetricAccumulator(args.metric_start_frame)
            strategy_sequences = {}
            temporal_metrics = {}

            for sequence_index, sequence in enumerate(sequences, start=1):
                prediction_callback = None
                if args.save_predictions:
                    prediction_dir_name = "prediction"
                    if len(view_sets_by_sequence[sequence.sequence_id]) > 1:
                        prediction_dir_name += "/" + "+".join(sequence.view_names)

                    def prediction_callback(*, frame, predictions, **_):
                        nonlocal saved_predictions
                        for view_index in range(len(sequence.view_names)):
                            save_prediction(
                                image_path=frame["image_path"][view_index],
                                original_size=frame["original_size"][view_index],
                                left_logits=predictions["left"][
                                    view_index:view_index + 1
                                ],
                                right_logits=predictions["right"][
                                    view_index:view_index + 1
                                ],
                                prediction_dir_name=prediction_dir_name,
                                output_dir=(
                                    args.output_dir
                                    / "predictions"
                                    / policy.configuration
                                ),
                                dataset_root=frame["dataset_root"][view_index],
                            )
                            saved_predictions += 1

                rows = evaluator.evaluate_sequence(
                    dataset,
                    sequence,
                    prediction_callback=prediction_callback,
                )
                sequence_metrics = MetricAccumulator(args.metric_start_frame)
                for row in rows:
                    writer.writerow(row)
                    strategy_metrics.update(row)
                    sequence_metrics.update(row)
                    temporal_metrics.setdefault(
                        row["frame_index"], MetricAccumulator(metric_start_frame=0)
                    ).update(row)
                strategy_sequences[sequence.evaluation_id] = sequence_metrics.result()

                logging.info(
                    "%s | sequence %d/%d complete | %s | frames=%d | mean_iou=%.4f",
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
            "num_views": args.num_views if args.model == "multiview" else 1,
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
    if args.save_predictions:
        logging.info(
            "Saved predictions: %d images under %s",
            saved_predictions,
            args.output_dir / "predictions",
        )
    if args.plot_curves:
        from inference.plotting import plot_results

        plot_results(
            args.output_dir,
            min_sequence_coverage=args.min_sequence_coverage,
            dpi=args.plot_dpi,
        )
