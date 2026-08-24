"""在 validation dataset 上推理并分类保存双手分割 hard cases。"""

from __future__ import annotations

import argparse
import logging
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from statistics import fmean, median
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset

from projects.hardcases.common import (
    analyze_frame,
    analyze_transition,
    save_frame_case,
    save_temporal_case,
)
from projects.framewise_sam2_modified.builder import build_sam2_modified_tiny
from projects.framewise_sam2_modified.dataset import (
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
    build_center_point_prompt,
    collate_batch,
)
from projects.framewise_sam2_modified.utils import configure_runtime, dump_json, set_seed, upsample_logits


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = (
    "xingyi_4-5090_oak150-100output",
    "wuwen_4-5090_release-0623-compressed",
    "tencent_4-5090_7.5",
)

MODEL_CONFIG_KEYS = (
    "image_size", "use_image_adapter", "use_decoder_adapter", "adapter_dim",
    "adapter_dropout", "adapter_init_scale", "multimask_output",
)
FRAME_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find hard cases on validation data.")

    parser.add_argument("--dataset", choices=["multiserver", "dexycb", "mixed"], default="mixed")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dex-ycb-root", "--dex_ycb_root", dest="dex_ycb_root", type=Path)
    parser.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--test-seq-count", type=int, default=3)
    parser.add_argument("--dex-ycb-setup", default="s0")
    parser.add_argument("--image-size", type=int, default=768)

    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=REPO_ROOT / "checkpoints/sam2.1_hiera_tiny.pt")
    parser.add_argument("--model-checkpoint", type=Path)
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument("--use-image-adapter", action="store_true")
    parser.add_argument("--use-decoder-adapter", action="store_true")
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-init-scale", type=float, default=1e-3)

    parser.add_argument("--use-point-prompt", action="store_true")

    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    args = parser.parse_args()

    if args.dataset in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error(f"--dataset {args.dataset} 需要 --dex-ycb-root")
    if args.test_seq_count <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        parser.error("test-seq-count/batch-size 必须为正数，num-workers 不能为负数")
    return args


class StreamOrderedDataset(Dataset):
    """把 dataset.streams 展平成逐 stream、逐帧的验证顺序。"""

    def __init__(self, dataset: Dataset) -> None:
        self.dataset = dataset
        self.records = [
            (sample_index, stream_id)
            for stream_id, stream in dataset.streams.items()
            for sample_index in stream["sample_indices"]
        ]
        if len(self.records) != len(dataset):
            raise ValueError("dataset.streams 未覆盖全部 validation 样本")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, stream_id = self.records[index]
        item = dict(self.dataset[sample_index])
        item.update(
            stream_id=stream_id,
            source_frame_number=_source_frame_number(item["image_path"]),
        )
        return item


def hardcase_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = collate_batch(batch)
    result["stream_id"] = [item["stream_id"] for item in batch]
    result["source_frame_number"] = [item["source_frame_number"] for item in batch]
    return result


def build_validation_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    datasets = []
    if args.dataset in {"multiserver", "mixed"}:
        datasets.append(MultiServerDualHandDataset(
            dataset_root=args.dataset_root, split="val", test_seq_count=args.test_seq_count,
            image_size=args.image_size, use_augmentation=False, dataset_names=args.dataset_names,
        ))
    if args.dataset in {"dexycb", "mixed"}:
        datasets.append(DexYCBDataset(
            dataset_root=args.dex_ycb_root, split="val", setup=args.dex_ycb_setup,
            image_size=args.image_size, use_augmentation=False,
        ))

    base_dataset = datasets[0] if len(datasets) == 1 else CombinedStreamDataset(datasets)
    dataset = StreamOrderedDataset(base_dataset)
    kwargs = dict(
        dataset=dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
        pin_memory=device.type == "cuda", collate_fn=hardcase_collate, drop_last=False,
    )
    if args.num_workers > 0:
        kwargs.update(persistent_workers=True, prefetch_factor=max(2, args.prefetch_factor))

    logging.info("Validation | mode=%s | streams=%d | samples=%d", args.dataset, len(base_dataset.streams), len(dataset))
    return DataLoader(**kwargs)


def load_model(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint = None
    if args.model_checkpoint is not None:
        if not args.model_checkpoint.is_file():
            raise FileNotFoundError(f"训练 checkpoint 不存在: {args.model_checkpoint}")
        checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
        if "model_state" not in checkpoint:
            raise KeyError("训练 checkpoint 中不存在 model_state")
        saved_args = checkpoint.get("args", {})
        for name in MODEL_CONFIG_KEYS:
            if name in saved_args:
                setattr(args, name, saved_args[name])

    model = build_sam2_modified_tiny(
        checkpoint_path=args.sam_checkpoint, device=args.device, mode="eval", image_size=args.image_size,
        use_image_adapter=args.use_image_adapter, use_decoder_adapter=args.use_decoder_adapter,
        adapter_dim=args.adapter_dim, adapter_dropout=args.adapter_dropout,
        adapter_init_scale=args.adapter_init_scale,
    )
    if checkpoint is not None:
        state = checkpoint["model_state"]
        if not any(k.startswith("left_mask_decoder.") for k in state):
            raise KeyError("训练 checkpoint 缺少 left_mask_decoder")
        if not any(k.startswith("right_mask_decoder.") for k in state):
            raise KeyError("训练 checkpoint 缺少 right_mask_decoder")
        model.load_state_dict(state, strict=True)
        logging.info("Loaded dual-decoder checkpoint: %s", args.model_checkpoint)

    if args.channels_last and torch.device(args.device).type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    return model.eval()


def _source_frame_number(image_path: str) -> int | None:
    match = FRAME_NUMBER_PATTERN.search(Path(image_path).stem)
    return int(match.group(1)) if match else None


@torch.inference_mode()
def run_analysis(model: torch.nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> None:
    frame_cases, temporal_cases, previous = [], [], None
    stats = {
        name: {
            "analyzed_frames": 0, "valid_hand_ious": [], "hardcase_frames": 0,
            "category_counts": Counter(), "reason_counts": Counter(),
            "temporal_comparisons": 0, "temporal_jumps": 0,
        }
        for name in ("overall", "multiserver", "dexycb")
    }
    point_prompt = build_center_point_prompt if args.use_point_prompt else None
    use_amp = device.type == "cuda" and args.amp

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)
        if args.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        context = torch.amp.autocast("cuda") if use_amp else nullcontext()
        with context:
            outputs = model.forward_single_image(
                images=images,
                left_point_inputs=point_prompt(left_masks) if point_prompt else None,
                right_point_inputs=point_prompt(right_masks) if point_prompt else None,
                mask_inputs=None, multimask_output=args.multimask_output,
            )

        batch_size = images.size(0)
        for index in range(batch_size):
            dataset_group = "dexycb" if batch["dataset_name"][index].lower() == "dexycb" else "multiserver"
            current_stats = (stats["overall"], stats[dataset_group])
            left_gt = batch["original_left_mask"][index].unsqueeze(0).to(device)
            right_gt = batch["original_right_mask"][index].unsqueeze(0).to(device)
            size = left_gt.shape[-2:]
            left_logits = upsample_logits(outputs["left"]["high_res_masks"][index:index + 1], size)
            right_logits = upsample_logits(outputs["right"]["high_res_masks"][index:index + 1], size)
            record, current = analyze_frame(
                batch, index, left_logits, right_logits, left_gt, right_gt,
            )
            frame_ious = [current["metrics"][side]["region_iou"] for side in ("left", "right") if current["metrics"][side]["gt_state"] == "valid"]
            categories, reasons = set(), []
            if record is not None:
                frame_cases.append(record)
                categories = {issue["category"] for issue in record["issues"]}
                reasons = [f"{issue['category']}/{issue['reason']}" for issue in record["issues"]]
                save_frame_case(record, current, args.output_dir)

            is_temporal_comparison = (
                previous is not None
                and previous["stream_id"] == current["stream_id"]
                and previous["source_frame_number"] is not None
                and current["source_frame_number"]
                == previous["source_frame_number"] + 1
            )
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
                values["temporal_comparisons"] += is_temporal_comparison
                values["temporal_jumps"] += event is not None
            previous = current

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            logging.info(
                "Scan samples=%d/%d | frame=%d | temporal=%d | categories=%s",
                stats["overall"]["analyzed_frames"], len(loader.dataset),
                stats["overall"]["hardcase_frames"], stats["overall"]["temporal_jumps"],
                dict(stats["overall"]["category_counts"]),
            )

    summary = {}
    for name, values in stats.items():
        analyzed_frames = values["analyzed_frames"]
        valid_hand_ious = values["valid_hand_ious"]
        temporal_comparisons = values["temporal_comparisons"]
        summary[name] = {
            "analyzed_frames": analyzed_frames,
            "evaluated_hands": len(valid_hand_ious),
            "hardcase_frames": values["hardcase_frames"],
            "hardcase_rate": round(values["hardcase_frames"] / analyzed_frames, 4) if analyzed_frames else None,
            "mean_iou": round(fmean(valid_hand_ious), 4) if valid_hand_ious else None,
            "median_iou": round(median(valid_hand_ious), 4) if valid_hand_ious else None,
            "category_counts": dict(values["category_counts"]),
            "category_rates": {category: round(count / analyzed_frames, 4) for category, count in values["category_counts"].items()} if analyzed_frames else {},
            "reason_counts": dict(values["reason_counts"]),
            "temporal_comparisons": temporal_comparisons,
            "temporal_jumps": values["temporal_jumps"],
            "temporal_jump_rate": round(values["temporal_jumps"] / temporal_comparisons, 4) if temporal_comparisons else None,
        }

    dump_json({
        "dataset": args.dataset,
        "use_point_prompt": args.use_point_prompt,
        "model_checkpoint": str(args.model_checkpoint) if args.model_checkpoint else None,
        "frame_cases": frame_cases,
        "temporal_cases": temporal_cases,
    }, args.output_dir / "hardcases.json")
    dump_json(summary, args.output_dir / "summary.json")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32, channels_last=args.channels_last)
    model = load_model(args)  # checkpoint 中的模型配置会同步到 args.image_size
    loader = build_validation_loader(args, device)
    run_analysis(model, loader, device, args)
    logging.info("Hardcase analysis complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
