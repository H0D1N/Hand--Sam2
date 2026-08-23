"""扫描双手 Memory tracking 的最终输出，并复用通用 hardcase 分类。"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from statistics import fmean, median

import torch
from torch.utils.data import DataLoader

from projects.dual_hand_memory.builder import build_sam2_dual_hand_memory_tiny
from projects.dual_hand_memory.dataset import ConsecutiveClipDataset, collate_clip_batch
from projects.framewise_sam2_modified.dataset import (
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
)
from projects.framewise_sam2_modified.utils import (
    configure_runtime,
    dump_json,
    set_seed,
    upsample_logits,
)
from projects.hardcases.common import (
    analyze_frame,
    analyze_transition,
    save_frame_case,
    save_temporal_case,
)


def parse_args() -> argparse.Namespace:
    """解析最小运行参数；模型和数据配置优先从训练 checkpoint 恢复。"""

    parser = argparse.ArgumentParser(description="Find hard cases in Memory tracking outputs.")
    parser.add_argument("--model-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("multiserver", "dexycb", "mixed"), default=None)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dex-ycb-root", type=Path)
    parser.add_argument("--dex-ycb-setup", default="s0")
    parser.add_argument("--dataset-names", nargs="+", default=None)
    parser.add_argument("--test-seq-count", type=int, default=None)
    parser.add_argument("--clip-length", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """按 checkpoint 中的训练配置重建模型，再严格加载完整 Memory 权重。"""

    checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]

    # 命令行参数可以覆盖 checkpoint 中的数据配置。
    args.dataset = args.dataset or saved_args["dataset_mode"]
    args.dataset_root = args.dataset_root or saved_args["dataset_root"]
    args.dex_ycb_root = args.dex_ycb_root or saved_args["dex_ycb_root"]
    args.dataset_names = args.dataset_names or saved_args["dataset_names"]
    args.test_seq_count = args.test_seq_count or saved_args["test_seq_count"]
    args.clip_length = args.clip_length or saved_args["clip_length"]

    model = build_sam2_dual_hand_memory_tiny(
        sam_checkpoint=saved_args["sam_checkpoint"],
        framewise_checkpoint=saved_args["framewise_checkpoint"],
        device=device,
        mode="eval",
        image_size=saved_args["image_size"],
        use_image_adapter=saved_args["use_image_adapter"],
        use_decoder_adapter=saved_args["use_decoder_adapter"],
        adapter_dim=saved_args["adapter_dim"],
        adapter_dropout=saved_args["adapter_dropout"],
        adapter_init_scale=saved_args["adapter_init_scale"],
        num_init_cond_frames_for_train=saved_args["num_init_cond_frames_for_train"],
        num_frames_to_correct_for_train=saved_args["num_frames_to_correct_for_train"],
        add_all_frames_to_correct_as_cond=saved_args["add_all_frames_to_correct_as_cond"],
        num_correction_pt_per_frame=saved_args["num_correction_pt_per_frame"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    args.image_size = model.image_size
    logging.info(
        "Loaded Memory checkpoint | %s | epoch=%s | clip_length=%d",
        args.model_checkpoint,
        checkpoint.get("epoch", "unknown"),
        args.clip_length,
    )
    return model.eval()


def build_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    """按 framewise 的方式选择数据集，再包装成非重叠连续 clips。"""

    datasets = []
    if args.dataset in {"multiserver", "mixed"}:
        datasets.append(MultiServerDualHandDataset(
            dataset_root=args.dataset_root, split="val", test_seq_count=args.test_seq_count,
            image_size=args.image_size, use_augmentation=False,
            dataset_names=args.dataset_names,
        ))
    if args.dataset in {"dexycb", "mixed"}:
        datasets.append(DexYCBDataset(
            dataset_root=args.dex_ycb_root, split="val", setup=args.dex_ycb_setup,
            image_size=args.image_size, use_augmentation=False,
        ))

    frame_dataset = datasets[0] if len(datasets) == 1 else CombinedStreamDataset(datasets)
    clip_dataset = ConsecutiveClipDataset(
        frame_dataset, clip_length=args.clip_length, clip_stride=args.clip_length,
    )
    loader_kwargs = dict(
        dataset=clip_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
        collate_fn=collate_clip_batch,
    )
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)

    logging.info(
        "Validation | dataset=%s | clips=%d | clip_length=%d | tracking_frames=%d",
        args.dataset, len(clip_dataset), args.clip_length,
        len(clip_dataset) * (args.clip_length - 1),
    )
    return DataLoader(**loader_kwargs)


@torch.inference_mode()
def run_analysis(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    """
    运行 clip 推理，只统计首个 GT 条件帧之后的最终预测。

    Memory forward 返回 ``list[T]``。每个时间步只读取
    ``pred_masks_high_res``，不读取或比较任何 ``multistep_*`` 输出。
    """

    frame_cases, temporal_cases = [], []
    stats = {
        name: {
            "analyzed_frames": 0, "conditioning_frames": 0,
            "valid_hand_ious": [], "hardcase_frames": 0,
            "category_counts": Counter(), "reason_counts": Counter(),
            "temporal_comparisons": 0, "temporal_jumps": 0,
        }
        for name in ("overall", "multiserver", "dexycb")
    }
    use_amp = device.type == "cuda" and args.amp

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)

        # mask 模式只在 clip 首帧输入 GT；后续帧依靠 Memory tracking。
        context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with context:
            frame_outputs = model(
                images=images,
                left_masks=left_masks,
                right_masks=right_masks,
                prompt_mode="mask",
            )

        batch_size = images.size(0)

        for sample_index in range(batch_size):
            # 每个 clip 独立维护 previous，不跨 clip 比较 temporal jump。
            dataset_group = (
                "dexycb"
                if batch["dataset_name"][sample_index].lower() == "dexycb"
                else "multiserver"
            )
            current_stats = (stats["overall"], stats[dataset_group])
            previous = None
            for frame_index, outputs in enumerate(frame_outputs):
                left_gt = (
                    batch["original_left_mask"][sample_index][frame_index]
                    .unsqueeze(0).to(device)
                )
                right_gt = (
                    batch["original_right_mask"][sample_index][frame_index]
                    .unsqueeze(0).to(device)
                )
                original_size = left_gt.shape[-2:]

                # pred_masks_high_res 是 Memory/纠错流程给出的最终 logits。
                left_logits = upsample_logits(
                    outputs["left"]["pred_masks_high_res"][sample_index:sample_index + 1],
                    original_size,
                )
                right_logits = upsample_logits(
                    outputs["right"]["pred_masks_high_res"][sample_index:sample_index + 1],
                    original_size,
                )
                # common.py 使用单帧 batch 接口；这里仅转换 clip 元数据的索引方式。
                frame_batch = {
                    "dataset_name": [batch["dataset_name"][sample_index]],
                    "sample_id": [batch["sample_id"][sample_index][frame_index]],
                    "image_path": [batch["image_path"][sample_index][frame_index]],
                    "stream_id": [batch["stream_id"][sample_index]],
                    "source_frame_number": [batch["frame_numbers"][sample_index][frame_index]],
                    "original_image": [batch["original_image"][sample_index][frame_index]],
                }
                record, current = analyze_frame(
                    frame_batch, 0, left_logits, right_logits, left_gt, right_gt,
                )

                # 第 0 帧直接输入 GT mask，只用于建立 Memory，不进入结果统计。
                if frame_index == 0:
                    for values in current_stats:
                        values["conditioning_frames"] += 1
                    previous = current
                    continue

                frame_ious = [
                    current["metrics"][side]["region_iou"]
                    for side in ("left", "right")
                    if current["metrics"][side]["gt_state"] == "valid"
                ]
                categories = set()
                reasons = []
                if record is not None:
                    frame_cases.append(record)
                    categories = {
                        issue["category"]
                        for issue in record["issues"]
                    }
                    reasons = [
                        f"{issue['category']}/{issue['reason']}"
                        for issue in record["issues"]
                    ]
                    save_frame_case(record, current, args.output_dir)

                event = analyze_transition(previous, current)
                if event is not None:
                    temporal_cases.append(event)
                    save_temporal_case(event, previous, current, args.output_dir)

                for values in current_stats:
                    values["analyzed_frames"] += 1
                    values["valid_hand_ious"].extend(frame_ious)
                    values["hardcase_frames"] += record is not None
                    values["category_counts"].update(categories)
                    values["reason_counts"].update(reasons)
                    values["temporal_comparisons"] += 1
                    values["temporal_jumps"] += event is not None
                previous = current

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            logging.info(
                "Scan clips=%d/%d | frames=%d | hardcases=%d | temporal=%d",
                min(step * args.batch_size, len(loader.dataset)),
                len(loader.dataset),
                stats["overall"]["analyzed_frames"],
                stats["overall"]["hardcase_frames"],
                stats["overall"]["temporal_jumps"],
            )

    # 三组使用完全相同的统计口径，方便比较总体和两个数据源。
    summary = {}
    for name, values in stats.items():
        analyzed_frames = values["analyzed_frames"]
        valid_hand_ious = values["valid_hand_ious"]
        temporal_comparisons = values["temporal_comparisons"]
        summary[name] = {
            "analyzed_frames": analyzed_frames,
            "conditioning_frames": values["conditioning_frames"],
            "evaluated_hands": len(valid_hand_ious),
            "hardcase_frames": values["hardcase_frames"],
            "hardcase_rate": round(values["hardcase_frames"] / analyzed_frames, 4) if analyzed_frames else None,
            "mean_iou": round(fmean(valid_hand_ious), 4) if valid_hand_ious else None,
            "median_iou": round(median(valid_hand_ious), 4) if valid_hand_ious else None,
            "category_counts": dict(values["category_counts"]),
            "category_rates": {
                category: round(count / analyzed_frames, 4)
                for category, count in values["category_counts"].items()
            } if analyzed_frames else {},
            "reason_counts": dict(values["reason_counts"]),
            "temporal_comparisons": temporal_comparisons,
            "temporal_jumps": values["temporal_jumps"],
            "temporal_jump_rate": round(values["temporal_jumps"] / temporal_comparisons, 4) if temporal_comparisons else None,
        }

    dump_json({
        "dataset": args.dataset,
        "model_checkpoint": str(args.model_checkpoint),
        "clip_length": args.clip_length,
        "frame_cases": frame_cases,
        "temporal_cases": temporal_cases,
    }, args.output_dir / "hardcases.json")
    dump_json(summary, args.output_dir / "summary.json")


def main() -> None:
    """组装运行环境、模型、validation loader 和分析循环。"""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)
    model = load_model(args, device)
    loader = build_loader(args, device)
    run_analysis(model, loader, device, args)
    logging.info("Memory hardcase analysis complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
