"""统一运行 SAM2、Framewise、Memory 和 MultiView 双手分割推理。"""

from __future__ import annotations

import argparse
from collections import defaultdict
import logging
from contextlib import nullcontext
from pathlib import Path

import torch

from inference.builder import build_model
from inference.dataset import build_frame_dataset, build_loader
from inference.evaluation import (
    add_evaluation_arguments,
    run_evaluation,
    validate_evaluation_args,
)
from inference.long_video_dataset import LongVideoDataset
from inference.output import prediction_path, save_prediction
from inference.streaming import EvaluationPolicy, LongVideoEvaluator
from projects.framewise_sam2_modified.dataset import build_center_point_prompt
from projects.framewise_sam2_modified.utils import configure_runtime
from sam2.modeling.sam2_utils import get_next_point
from training.model.sam2_multiview_dual_hand_memory import PromptRequest


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = (
    "xingyi_4-5090_oak150-100output",
    "wuwen_4-5090_release-0623-compressed",
    "tencent_4-5090_7.5",
)


def add_prediction_arguments(parser: argparse.ArgumentParser) -> None:
    common = parser.add_argument_group("common")
    common.add_argument(
        "--model",
        choices=("sam2", "framewise", "memory", "multiview"),
        required=True,
    )
    common.add_argument("--model-checkpoint", type=Path, required=True)
    common.add_argument("--prediction-dir-name", help="每个图像目录对应的推理文件夹名；默认 <model>_prediction。")
    common.add_argument("--output-dir", type=Path, help="输出根目录；默认写回原图对应的相机目录。")
    common.add_argument("--batch-size", type=int)
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    common.add_argument("--amp", action="store_true")
    common.add_argument("--disable-tf32", action="store_true")
    common.add_argument("--log-interval", type=int, default=50)
    common.add_argument(
        "--image-size",
        type=int,
        default=768,
        help="仅原始 SAM2 checkpoint 需要；训练 checkpoint 会读取自身配置。",
    )

    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset", choices=("multiserver", "dexycb", "mixed"), default="multiserver")
    data.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    data.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    data.add_argument("--test-seq-count", type=int, default=3)
    data.add_argument("--dex-ycb-root", type=Path)
    data.add_argument("--dex-ycb-setup", default="s0")

    framewise = parser.add_argument_group("framewise")
    framewise.add_argument("--use-point-prompt", action="store_true")

    memory = parser.add_argument_group("memory")
    memory.add_argument("--prompt-mode", choices=("mask", "point", "auto"), default="mask")

    multiview = parser.add_argument_group("multiview PromptPlan")
    multiview.add_argument("--num-views", type=int, default=2)
    multiview.add_argument(
        "--prompt-frame-indices",
        type=int,
        nargs="+",
        default=(0,),
        help="使用 mask/point Prompt 的相对帧下标。",
    )
    multiview.add_argument(
        "--correction-frame-indices",
        type=int,
        nargs="*",
        default=(),
        help="追加 GT 误差点纠错的相对帧下标。",
    )
    multiview.add_argument(
        "--correction-points",
        type=int,
        default=10,
        help="每个纠错帧、每只手追加的纠错点数。",
    )
    multiview.add_argument("--max-condition-frames", type=int, default=4)
    multiview.add_argument(
        "--add-correction-frames-as-cond",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    multiview.add_argument("--disable-multiview-fusion", action="store_true")


def validate_prediction_args(args, parser: argparse.ArgumentParser) -> None:
    args.prediction_dir_name = args.prediction_dir_name or f"{args.model}_prediction"
    if args.batch_size is None:
        args.batch_size = 1 if args.model in {"memory", "multiview"} else 2
    if args.batch_size < 1 or (
        args.model in {"memory", "multiview"} and args.batch_size != 1
    ):
        parser.error(
            "batch-size 必须为正整数，memory/multiview 推理仅支持 "
            "--batch-size 1"
        )
    if args.dataset in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error(f"--dataset {args.dataset} 需要 --dex-ycb-root")
    if args.image_size <= 0 or args.image_size % 16 != 0:
        parser.error("--image-size 必须为 16 的正整数倍")
    if args.model == "sam2":
        args.use_point_prompt = True
    if args.model == "multiview":
        if args.num_views < 2:
            parser.error("multiview 推理要求 --num-views 至少为 2")
        if args.prompt_mode == "auto":
            parser.error("multiview 长视频推理需要明确的 mask 或 point PromptPlan")
        if any(index < 0 for index in args.prompt_frame_indices):
            parser.error("--prompt-frame-indices 不能包含负数")
        if any(index < 0 for index in args.correction_frame_indices):
            parser.error("--correction-frame-indices 不能包含负数")
        if 0 not in args.prompt_frame_indices:
            parser.error("multiview 因果推理的 PromptPlan 必须包含第 0 帧")
        if args.correction_points < 1:
            parser.error("--correction-points 必须大于 0")
        if args.max_condition_frames < 1:
            parser.error("--max-condition-frames 必须大于 0")
        args.prompt_frame_indices = tuple(dict.fromkeys(args.prompt_frame_indices))
        args.correction_frame_indices = tuple(
            dict.fromkeys(args.correction_frame_indices)
        )
        args.mask_frame_indices = tuple(sorted(
            set(args.prompt_frame_indices) | set(args.correction_frame_indices)
        ))
    else:
        args.mask_frame_indices = (0,)


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
def predict_memory(model, dataset, args) -> int:
    count = 0
    for stream_id, stream in dataset.streams.items():
        loader = build_loader(dataset, args, sample_indices=stream["sample_indices"])
        state = model.init_state(loader)
        for hand in ("left", "right"):
            mask = state["first_frame"][f"{hand}_mask"]
            if args.prompt_mode == "point":
                coords, labels = get_next_point(gt_masks=mask.bool(), pred_masks=None, method=model.pt_sampling_for_eval)
                model.add_points(state, hand, {"point_coords": coords, "point_labels": labels})
                del coords, labels
            else:
                model.add_mask(state, hand, mask)
        del mask
        with _autocast(args):
            for frame_idx, outputs, frame_info in model.propagate_in_video(state):
                save_prediction(
                    image_path=frame_info["image_path"], original_size=frame_info["original_size"],
                    left_logits=outputs["left"]["pred_masks_high_res"],
                    right_logits=outputs["right"]["pred_masks_high_res"],
                    prediction_dir_name=args.prediction_dir_name,
                    output_dir=getattr(args, "output_dir", None),
                    dataset_root=frame_info["dataset_root"],
                )
                count += 1
                if (frame_idx + 1) % args.log_interval == 0 or frame_idx + 1 == state["num_frames"]:
                    logging.info("memory | %s | %d/%d frames | %d images", stream_id, frame_idx + 1, state["num_frames"], count)
                del outputs
        del state, loader
    return count


def build_multiview_prompt_request(args) -> PromptRequest:
    """把推理参数组成训练模型使用的同一种 PromptRequest。"""
    return PromptRequest(
        mode=args.prompt_mode,
        start_frame_idx=0,
        prompt_frame_indices=tuple(args.prompt_frame_indices),
        correction_frame_indices=tuple(args.correction_frame_indices),
        num_correction_points_per_frame=args.correction_points,
        add_correction_frames_as_cond=args.add_correction_frames_as_cond,
    )


@torch.inference_mode()
def predict_multiview(model, frame_dataset, args) -> int:
    request = build_multiview_prompt_request(args)
    dataset = LongVideoDataset(
        frame_dataset,
        num_views=args.num_views,
        min_sequence_length=1,
    )
    if not dataset.sequences:
        raise ValueError(
            f"没有找到至少包含 {args.num_views} 个同步视角的视频序列"
        )

    model.set_multiview_fusion_enabled(not args.disable_multiview_fusion)
    policy = EvaluationPolicy(
        strategy="explicit",
        prompt_mode=request.mode,
        correction_points=request.num_correction_points_per_frame,
        max_condition_frames=args.max_condition_frames,
        prompt_frame_indices=request.prompt_frame_indices,
        correction_frame_indices=request.correction_frame_indices or (),
        add_correction_frames_as_cond=request.add_correction_frames_as_cond,
    )
    evaluator = LongVideoEvaluator(
        model=model,
        model_type="multiview",
        device=args.device,
        policy=policy,
        amp=args.amp,
        log_interval=args.log_interval,
    )

    view_sets_by_sequence = defaultdict(set)
    for sequence in dataset.sequences:
        view_sets_by_sequence[sequence.sequence_id].add(sequence.view_names)

    logging.info(
        "MultiView PromptPlan | prompt_mode=%s | prompt_frames=%s | "
        "correction_frames=%s | correction_points=%d | sequences=%d",
        request.mode,
        list(request.prompt_frame_indices),
        list(request.correction_frame_indices or ()),
        request.num_correction_points_per_frame,
        len(dataset.sequences),
    )

    count = 0
    for sequence_index, sequence in enumerate(dataset.sequences, start=1):
        prediction_dir_name = args.prediction_dir_name
        if len(view_sets_by_sequence[sequence.sequence_id]) > 1:
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
            "multiview | sequence %d/%d complete | %s | frames=%d | "
            "predictions=%d",
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
    if args.model == "memory":
        return predict_memory(model, dataset, args)
    return predict_multiview(model, dataset, args)


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
    dataset = build_frame_dataset(args)
    count = run_prediction(model, dataset, args)
    logging.info("Prediction complete | %d images", count)


if __name__ == "__main__":
    main()
