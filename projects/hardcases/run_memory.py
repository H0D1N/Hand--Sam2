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
from projects.framewise_sam2_modified.dataset import MultiServerDualHandDataset
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
    parser.add_argument("--dataset-root", type=Path)
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

    # 命令行只用于覆盖服务器路径或本次扫描的 clip 长度。
    args.dataset_root = args.dataset_root or saved_args["dataset_root"]
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
    category_counts, reason_counts = Counter(), Counter()
    valid_hand_ious = []
    analyzed_frames, conditioning_frames, temporal_comparisons = 0, 0, 0
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
                    conditioning_frames += 1
                    previous = current
                    continue

                analyzed_frames += 1
                valid_hand_ious.extend(
                    current["metrics"][side]["region_iou"]
                    for side in ("left", "right")
                    if current["metrics"][side]["gt_state"] == "valid"
                )
                if record is not None:
                    frame_cases.append(record)
                    category_counts.update({
                        issue["category"]
                        for issue in record["issues"]
                    })
                    reason_counts.update(
                        f"{issue['category']}/{issue['reason']}"
                        for issue in record["issues"]
                    )
                    save_frame_case(record, current, args.output_dir)

                temporal_comparisons += 1
                event = analyze_transition(previous, current)
                if event is not None:
                    temporal_cases.append(event)
                    save_temporal_case(event, previous, current, args.output_dir)
                previous = current

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            logging.info(
                "Scan clips=%d/%d | frames=%d | hardcases=%d | temporal=%d",
                min(step * args.batch_size, len(loader.dataset)),
                len(loader.dataset),
                analyzed_frames,
                len(frame_cases),
                len(temporal_cases),
            )

    # 汇总口径与 run_framewise.py 一致，额外记录被排除的条件帧数量。
    category_rates = ({
        category: round(count / analyzed_frames, 4)
        for category, count in category_counts.items()
    } if analyzed_frames else {})
    dump_json({
        "dataset": "multiserver",
        "model_checkpoint": str(args.model_checkpoint),
        "clip_length": args.clip_length,
        "frame_cases": frame_cases,
        "temporal_cases": temporal_cases,
    }, args.output_dir / "hardcases.json")
    dump_json({
        "analyzed_frames": analyzed_frames,
        "conditioning_frames": conditioning_frames,
        "evaluated_hands": len(valid_hand_ious),
        "hardcase_frames": len(frame_cases),
        "hardcase_rate": (
            round(len(frame_cases) / analyzed_frames, 4)
            if analyzed_frames else None
        ),
        "mean_iou": round(fmean(valid_hand_ious), 4) if valid_hand_ious else None,
        "median_iou": round(median(valid_hand_ious), 4) if valid_hand_ious else None,
        "category_counts": dict(category_counts),
        "category_rates": category_rates,
        "reason_counts": dict(reason_counts),
        "temporal_comparisons": temporal_comparisons,
        "temporal_jumps": len(temporal_cases),
        "temporal_jump_rate": (
            round(len(temporal_cases) / temporal_comparisons, 4)
            if temporal_comparisons else None
        ),
    }, args.output_dir / "summary.json")


def main() -> None:
    """组装现有 MultiServer frame dataset、非重叠 clips、模型和分析循环。"""

    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)
    model = load_model(args, device)

    frame_dataset = MultiServerDualHandDataset(
        dataset_root=args.dataset_root, split="val", test_seq_count=args.test_seq_count,
        image_size=args.image_size, use_augmentation=False,
        dataset_names=args.dataset_names,
    )
    clip_dataset = ConsecutiveClipDataset(
        frame_dataset, clip_length=args.clip_length, clip_stride=args.clip_length,
    )

    loader_kwargs = {
        "dataset": clip_dataset,
        "batch_size": args.batch_size,
        "shuffle": False,
        "drop_last": False,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_clip_batch,
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2
    loader = DataLoader(**loader_kwargs)

    logging.info(
        "Validation | clips=%d | clip_length=%d | tracking_frames=%d",
        len(clip_dataset),
        args.clip_length,
        len(clip_dataset) * (args.clip_length - 1),
    )
    run_analysis(model, loader, device, args)
    logging.info("Memory hardcase analysis complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
