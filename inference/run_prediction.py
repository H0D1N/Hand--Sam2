"""推理/评估唯一 CLI：定义参数、校验参数并分发执行。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
from contextlib import nullcontext
from pathlib import Path

import torch

from inference.builder import build_model
from inference.dataset import build_frame_dataset, build_loader
from inference.evaluation import run_evaluation
from inference.long_video_dataset import (
    LongVideoDataset,
    build_full_gt_frame_dataset,
)
from inference.output import prediction_path, save_prediction
from inference.streaming import EvaluationPolicy, LongVideoEvaluator
from projects.framewise_sam2_modified.dataset import build_center_point_prompt
from projects.framewise_sam2_modified.utils import configure_runtime


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = (
    "xingyi_4-5090_oak150-100output",
    "wuwen_4-5090_release-0623-compressed",
    "tencent_4-5090_7.5",
)
MODELS = ("sam2", "framewise", "memory", "multiview")
DATASETS = ("multiserver", "dexycb", "mixed")
STRATEGIES = ("baseline", "fixed", "adaptive")
SINGLE_FRAME_MODELS = {"sam2", "framewise"}
LONG_VIDEO_MODELS = {"memory", "multiview"}
DEFAULT_DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    model = parser.add_argument_group("model and runtime")
    model.add_argument("--model", choices=MODELS, required=True, help="checkpoint 类型：原始 SAM2、单帧、单视角 Memory 或 MultiView。")
    model.add_argument("--model-checkpoint", type=Path, required=True, help="模型 checkpoint 路径。")
    model.add_argument("--image-size", type=int, default=768, help="原始 SAM2 的输入尺寸；训练 checkpoint 会覆盖此值。")
    model.add_argument("--device", default=DEFAULT_DEVICE, help="例如 cuda:0 或 cpu。")
    model.add_argument("--amp", action="store_true", help="在 CUDA 上启用 bfloat16 autocast。")
    model.add_argument("--disable-tf32", action="store_true", help="关闭 CUDA TF32。")
    model.add_argument("--log-interval", type=int, default=50, help="每处理 N 帧输出一次进度。")

    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset", choices=DATASETS, default="multiserver", help="使用 MultiServer、DexYCB 或二者合并的数据。")
    data.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT, help="MultiServer datasets.json 所在目录。")
    data.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES, help="从 datasets.json 选择的数据集名称。")
    data.add_argument("--test-seq-count", type=int, default=3, help="每个 MultiServer 数据集按自然顺序选取最后 N 个序列。")
    data.add_argument("--dex-ycb-root", type=Path, help="DexYCB 根目录。")
    data.add_argument("--dex-ycb-setup", default="s0", help="DexYCB setup，默认 s0。")
    data.add_argument("--num-views", type=int, default=2, help="MultiView 同步推理使用的视角数。")


def add_prediction_arguments(parser: argparse.ArgumentParser) -> None:
    add_shared_arguments(parser)
    output = parser.add_argument_group("prediction output")
    output.add_argument("--output-dir", type=Path, help="预测输出根目录；不传时写回各相机目录。")
    output.add_argument("--prediction-dir-name", help="预测 mask 子目录名；默认 <model>_prediction。")
    output.add_argument("--batch-size", type=int, help="单帧模型默认 2；长视频模型固定为 1。")
    output.add_argument("--num-workers", type=int, default=4, help="DataLoader worker 数。")

    framewise = parser.add_argument_group("framewise")
    framewise.add_argument("--use-point-prompt", action="store_true", help="Framewise 每帧使用 GT 中心点；原始 SAM2 始终使用该提示。")

    prompt = parser.add_argument_group("prompt strategy")
    prompt.add_argument("--strategy", choices=STRATEGIES, default="baseline", help="Memory/MultiView 策略：baseline 仅首帧；fixed 定期提示；adaptive 按 IoU 纠错。")
    prompt.add_argument("--prompt-mode", choices=("mask", "point"), default="mask", help="Memory/MultiView 首帧及 fixed 提示使用 GT mask 或 GT 中心点。")
    prompt.add_argument("--prompt-interval", type=int, default=200, help="fixed：从每段第 0 帧开始，每 N 帧重新提示。")
    prompt.add_argument("--iou-threshold", type=float, default=0.5, help="adaptive：当前预测低于该 IoU 时开始连续点纠错。")
    prompt.add_argument("--correction-points", type=int, default=10, help="adaptive：每帧每只手的最大点击数；达到 IoU 阈值会提前停止。")
    prompt.add_argument("--max-condition-frames", type=int, default=4, help="Memory 中最多保留的提示/纠错条件帧数。")

    multiview = parser.add_argument_group("multiview")
    multiview.add_argument("--disable-multiview-fusion", action="store_true", help="关闭 MultiView 跨视角融合。")


def add_evaluation_arguments(parser: argparse.ArgumentParser) -> None:
    add_shared_arguments(parser)
    output = parser.add_argument_group("evaluation output")
    output.add_argument("--output-dir", type=Path, required=True, help="CSV、JSON、曲线和可选预测图的输出目录。")
    output.add_argument("--save-predictions", action="store_true", help="同时保存每种配置的预测 mask PNG。")
    output.add_argument("--plot-curves", action="store_true", help="评估完成后生成 IoU PNG 曲线。")
    output.add_argument("--plot-dpi", type=int, default=200, help="曲线 PNG 的 DPI。")

    prompt = parser.add_argument_group("prompt strategy sweep")
    prompt.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=("baseline",), help="要评估的策略；可同时传入 baseline fixed adaptive。")
    prompt.add_argument("--prompt-mode", choices=("none", "mask", "point"), help="普通提示类型；默认 SAM2=point、Framewise=none、长视频模型=mask。")
    prompt.add_argument("--fixed-intervals", type=int, nargs="+", default=(80,), help="fixed 扫描的提示间隔，例如 20 40 80 160。")
    prompt.add_argument("--adaptive-thresholds", type=float, nargs="+", default=(0.5,), help="adaptive 扫描的 IoU 阈值，例如 0.3 0.5 0.7。")
    prompt.add_argument("--correction-points", type=int, default=10, help="adaptive 每帧每只手的最大点击数；达到阈值会提前停止。")
    prompt.add_argument("--max-condition-frames", type=int, default=4, help="Memory 中最多保留的提示/纠错条件帧数。")

    metrics = parser.add_argument_group("evaluation range and metrics")
    metrics.add_argument("--min-sequence-length", type=int, default=2, help="忽略短于 N 帧的连续视频段。")
    metrics.add_argument("--max-sequences", type=int, help="只评估前 N 条长视频，用于 smoke test。")
    metrics.add_argument("--metric-start-frame", type=int, default=1, help="汇总指标从相对第 N 帧开始，默认排除首帧提示。")
    metrics.add_argument("--min-sequence-coverage", type=float, default=0.5, help="时序曲线保留点所需的最小视频覆盖比例。")
    metrics.add_argument("--seed", type=int, default=42, help="随机种子。")


def validate_shared_args(args, parser: argparse.ArgumentParser) -> None:
    if args.dataset in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error(f"--dataset {args.dataset} 需要 --dex-ycb-root")
    if args.image_size <= 0 or args.image_size % 16 != 0:
        parser.error("--image-size 必须为 16 的正整数倍")
    if args.test_seq_count < 1:
        parser.error("--test-seq-count 必须大于 0")
    if args.log_interval < 1:
        parser.error("--log-interval 必须大于 0")
    if args.model == "multiview" and args.num_views < 2:
        parser.error("multiview 要求 --num-views 至少为 2")


def validate_prediction_args(args, parser: argparse.ArgumentParser) -> None:
    validate_shared_args(args, parser)
    args.prediction_dir_name = args.prediction_dir_name or f"{args.model}_prediction"
    if args.batch_size is None:
        args.batch_size = 1 if args.model in LONG_VIDEO_MODELS else 2
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.model in LONG_VIDEO_MODELS and args.batch_size != 1:
        parser.error("memory/multiview 仅支持 --batch-size 1")

    if args.model == "sam2":
        args.use_point_prompt = True
    if args.model in SINGLE_FRAME_MODELS and args.strategy != "baseline":
        parser.error("sam2/framewise 只支持 baseline 逐帧推理")
    if args.prompt_interval < 1:
        parser.error("--prompt-interval 必须大于 0")
    if not 0.0 <= args.iou_threshold <= 1.0:
        parser.error("--iou-threshold 必须位于 [0, 1]")
    if args.correction_points < 1:
        parser.error("--correction-points 必须大于 0")
    if args.max_condition_frames < 1:
        parser.error("--max-condition-frames 必须大于 0")
    args.mask_frame_indices = (0,)
    args.mask_frame_interval = (
        args.prompt_interval if args.strategy == "fixed" else None
    )


def validate_evaluation_args(args, parser: argparse.ArgumentParser) -> None:
    validate_shared_args(args, parser)
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
    if args.model in LONG_VIDEO_MODELS and args.prompt_mode == "none":
        parser.error("memory/multiview 评估需要 --prompt-mode mask/point")
    if args.model in SINGLE_FRAME_MODELS and tuple(args.strategies) != ("baseline",):
        parser.error("sam2/framewise 只支持 baseline 逐帧评估")

    if any(interval < 1 for interval in args.fixed_intervals):
        parser.error("--fixed-intervals 必须全部大于 0")
    if any(not 0.0 <= value <= 1.0 for value in args.adaptive_thresholds):
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
    if not 0.0 <= args.min_sequence_coverage <= 1.0:
        parser.error("--min-sequence-coverage 必须位于 [0, 1]")
    if args.plot_dpi < 1:
        parser.error("--plot-dpi 必须大于 0")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="双手分割图片输出与长视频 IoU 评估"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prediction_parser = commands.add_parser(
        "predict", help="用四类 checkpoint 生成预测 mask PNG"
    )
    add_prediction_arguments(prediction_parser)
    evaluation_parser = commands.add_parser(
        "evaluate", help="在完整 GT 长视频上输出 IoU/CSV/曲线，可选保存 PNG"
    )
    add_evaluation_arguments(evaluation_parser)

    args = parser.parse_args(argv)
    if args.command == "predict":
        validate_prediction_args(args, prediction_parser)
    else:
        validate_evaluation_args(args, evaluation_parser)
    return args


def _autocast(args: argparse.Namespace):
    enabled = torch.device(args.device).type == "cuda" and args.amp
    return torch.amp.autocast("cuda") if enabled else nullcontext()


@torch.inference_mode()
def predict_framewise(model, loader, args) -> int:
    count = 0
    device = torch.device(args.device)
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)
        with _autocast(args):
            outputs = model.forward_single_image(
                images=images,
                left_point_inputs=build_center_point_prompt(left_masks) if args.use_point_prompt else None,
                right_point_inputs=build_center_point_prompt(right_masks) if args.use_point_prompt else None,
                mask_inputs=None,
                multimask_output=False,
            )
        for index in range(images.size(0)):
            save_prediction(
                image_path=batch["image_path"][index],
                original_size=batch["original_size"][index],
                left_logits=outputs["left"]["high_res_masks"][index:index + 1],
                right_logits=outputs["right"]["high_res_masks"][index:index + 1],
                prediction_dir_name=args.prediction_dir_name,
                output_dir=getattr(args, "output_dir", None),
                dataset_root=batch["dataset_root"][index],
            )
            count += 1
        if step % args.log_interval == 0 or step == len(loader):
            logging.info("framewise | %d/%d batches | %d images", step, len(loader), count)
    return count


@torch.inference_mode()
def predict_long_video(model, frame_dataset, args) -> int:
    """Memory/MultiView 共用 baseline、fixed、adaptive 流式推理。"""
    num_views = args.num_views if args.model == "multiview" else 1
    dataset = LongVideoDataset(
        frame_dataset,
        num_views=num_views,
        min_sequence_length=1,
    )
    if not dataset.sequences:
        raise ValueError(
            f"没有找到至少包含 {num_views} 个同步视角的视频序列"
        )

    if args.model == "multiview":
        model.set_multiview_fusion_enabled(
            not args.disable_multiview_fusion
        )
    policy = EvaluationPolicy(
        strategy=args.strategy,
        prompt_mode=args.prompt_mode,
        prompt_interval=args.prompt_interval,
        iou_threshold=args.iou_threshold,
        correction_points=args.correction_points,
        max_condition_frames=args.max_condition_frames,
    )
    evaluator = LongVideoEvaluator(
        model=model,
        model_type=args.model,
        device=args.device,
        policy=policy,
        amp=args.amp,
        log_interval=args.log_interval,
    )
    view_sets_by_sequence = defaultdict(set)
    for sequence in dataset.sequences:
        view_sets_by_sequence[sequence.sequence_id].add(sequence.view_names)

    logging.info(
        "%s | strategy=%s | prompt_mode=%s | prompt_interval=%d | "
        "iou_threshold=%.3f | correction_points_cap=%d | sequences=%d",
        args.model,
        args.strategy,
        args.prompt_mode,
        args.prompt_interval,
        args.iou_threshold,
        args.correction_points,
        len(dataset.sequences),
    )

    count = 0
    for sequence_index, sequence in enumerate(dataset.sequences, start=1):
        prediction_dir_name = args.prediction_dir_name
        if (
            args.model == "multiview"
            and len(view_sets_by_sequence[sequence.sequence_id]) > 1
        ):
            prediction_dir_name = (
                f"{prediction_dir_name}/" + "+".join(sequence.view_names)
            )

        def save_frame_predictions(*, frame, predictions, **_):
            nonlocal count
            for view_index in range(len(sequence.view_names)):
                save_prediction(
                    image_path=frame["image_path"][view_index],
                    original_size=frame["original_size"][view_index],
                    left_logits=predictions["left"][view_index:view_index + 1],
                    right_logits=predictions["right"][view_index:view_index + 1],
                    prediction_dir_name=prediction_dir_name,
                    output_dir=getattr(args, "output_dir", None),
                    dataset_root=frame["dataset_root"][view_index],
                )
                count += 1

        evaluator.evaluate_sequence(
            dataset,
            sequence,
            prediction_callback=save_frame_predictions,
            collect_metrics=False,
        )
        logging.info(
            "%s | sequence %d/%d complete | %s | frames=%d | "
            "predictions=%d",
            args.model,
            sequence_index,
            len(dataset.sequences),
            sequence.evaluation_id,
            sequence.num_frames,
            count,
        )
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    return count


def run_prediction(model, dataset, args) -> int:
    """统一推理入口；Memory/MultiView 按长视频逐帧传播。"""
    if args.model in {"sam2", "framewise"}:
        return predict_framewise(model, build_loader(dataset, args), args)
    return predict_long_video(model, dataset, args)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    if args.command == "evaluate":
        run_evaluation(args)
        return

    configure_runtime(
        torch.device(args.device), use_tf32=not args.disable_tf32
    )
    model = build_model(args)
    if args.model in {"memory", "multiview"} and args.strategy == "adaptive":
        dataset = build_full_gt_frame_dataset(
            dataset_mode=args.dataset,
            image_size=args.image_size,
            dataset_root=args.dataset_root,
            dataset_names=args.dataset_names,
            test_seq_count=args.test_seq_count,
            dex_ycb_root=args.dex_ycb_root,
            dex_ycb_setup=args.dex_ycb_setup,
        )
    else:
        dataset = build_frame_dataset(args)
    count = run_prediction(model, dataset, args)
    logging.info("Prediction complete | %d images", count)


if __name__ == "__main__":
    main()
