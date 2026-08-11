"""在 validation dataset 上推理并分类保存双手分割 hard cases。"""

from __future__ import annotations

import argparse
import logging
import math
import re
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .builder import build_sam2_modified_tiny
from .dataset import (
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
    build_center_point_prompt,
    collate_batch,
)
from .losses import iou_target_from_logits, object_targets_from_masks
from .utils import configure_runtime, dump_json, set_seed, upsample_logits
from .visualization import save_dual_hand_five_panel_visualization


REPO_ROOT = Path(__file__).resolve().parents[2]

# Basic decisions
MASK_LOGIT_THRESHOLD = 0.0
GT_MASK_THRESHOLD = 0.5
OBJECT_PROBABILITY_THRESHOLD = 0.5
LOW_REGION_IOU_THRESHOLD = 0.50

# Boundary error
BOUNDARY_MIN_REGION_IOU = 0.50
BOUNDARY_F1_THRESHOLD = 0.65
BOUNDARY_TOLERANCE_RATIO = 0.005

# Spatial inconsistency
COMPONENT_MIN_AREA_PIXELS = 16
COMPONENT_MIN_AREA_RATIO = 0.005
LARGEST_COMPONENT_RATIO_THRESHOLD = 0.80
LARGEST_COMPONENT_DROP_THRESHOLD = 0.15
CROSS_HAND_LEAKAGE_THRESHOLD = 0.20

# Temporal jump
TEMPORAL_GT_IOU_MIN = 0.60
TEMPORAL_PRED_IOU_MAX = 0.30
TEMPORAL_EXCESS_CHANGE_THRESHOLD = 0.35
TEMPORAL_COVERAGE_JUMP_THRESHOLD = 0.40
TEMPORAL_QUALITY_JUMP_THRESHOLD = 0.35

# Catastrophic failure
CATASTROPHIC_IOU_THRESHOLD = 0.10
ABSENT_MASK_AREA_THRESHOLD = 0.005

MODEL_CONFIG_KEYS = (
    "image_size", "use_image_adapter", "use_decoder_adapter", "adapter_dim",
    "adapter_dropout", "adapter_init_scale", "multimask_output",
)
FRAME_NUMBER_PATTERN = re.compile(r"(\d+)(?!.*\d)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find hard cases on validation data.")

    parser.add_argument("--dataset", choices=["multiserver", "dexycb", "mixed"], default="multiserver")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--dex-ycb-root", "--dex_ycb_root", dest="dex_ycb_root", type=Path)
    parser.add_argument("--dataset-names", nargs="+", default=None)
    parser.add_argument("--test-seq-count", type=int, default=2)
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

    if args.dataset in {"multiserver", "mixed"} and args.dataset_root is None:
        parser.error(f"--dataset {args.dataset} 需要 --dataset-root")
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
            (sample_index, stream_id, position)
            for stream_id, stream in dataset.streams.items()
            for position, sample_index in enumerate(stream["sample_indices"])
        ]
        if len(self.records) != len(dataset):
            raise ValueError("dataset.streams 未覆盖全部 validation 样本")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_index, stream_id, position = self.records[index]
        item = dict(self.dataset[sample_index])
        item.update(
            stream_id=stream_id,
            frame_position=position,
            source_frame_number=_source_frame_number(item["image_path"]),
        )
        return item


def hardcase_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    result = collate_batch(batch)
    result["stream_id"] = [item["stream_id"] for item in batch]
    result["frame_position"] = [item["frame_position"] for item in batch]
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


def _bool_mask(tensor: torch.Tensor, threshold: float) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().squeeze().gt(threshold).cpu().numpy())


def _source_frame_number(image_path: str) -> int | None:
    match = FRAME_NUMBER_PATTERN.search(Path(image_path).stem)
    return int(match.group(1)) if match else None


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(intersection / union)


def _boundary_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_edge = pred & (cv2.erode(pred.astype(np.uint8), kernel, borderValue=0) == 0)
    gt_edge = gt & (cv2.erode(gt.astype(np.uint8), kernel, borderValue=0) == 0)
    pred_count, gt_count = int(pred_edge.sum()), int(gt_edge.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0

    radius = max(1, round(math.hypot(*gt.shape) * BOUNDARY_TOLERANCE_RATIO))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_band = cv2.dilate(pred_edge.astype(np.uint8), kernel).astype(bool)
    gt_band = cv2.dilate(gt_edge.astype(np.uint8), kernel).astype(bool)
    precision = float((pred_edge & gt_band).sum() / pred_count) if pred_count else 0.0
    recall = float((gt_edge & pred_band).sum() / gt_count) if gt_count else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def _component_metrics(mask: np.ndarray, gt_area: int) -> tuple[int, float]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    min_area = max(COMPONENT_MIN_AREA_PIXELS, round(gt_area * COMPONENT_MIN_AREA_RATIO))
    areas = areas[areas >= min_area]
    largest_ratio = int(areas.max()) / max(int(mask.sum()), 1) if areas.size else 0.0
    return int(areas.size), largest_ratio


def analyze_frame(
    batch: dict[str, Any], index: int, left_logits: torch.Tensor, right_logits: torch.Tensor,
    left_gt: torch.Tensor, right_gt: torch.Tensor,
    left_object_logit: torch.Tensor, right_object_logit: torch.Tensor,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    logits = {"left": left_logits, "right": right_logits}
    gt_tensors = {"left": left_gt, "right": right_gt}
    object_logits = {"left": left_object_logit, "right": right_object_logit}
    pred = {side: _bool_mask(logits[side], MASK_LOGIT_THRESHOLD) for side in ("left", "right")}
    gt = {side: _bool_mask(gt_tensors[side], GT_MASK_THRESHOLD) for side in ("left", "right")}
    labels, reasons, metrics = set(), [], {}

    for side in ("left", "right"):
        gt_present = bool(object_targets_from_masks(gt_tensors[side])[0].item())
        object_probability = float(torch.sigmoid(object_logits[side]).item())
        pred_present = object_probability >= OBJECT_PROBABILITY_THRESHOLD
        region_iou = float(iou_target_from_logits(logits[side], gt_tensors[side]).item())
        gt_area, pred_area = int(gt[side].sum()), int(pred[side].sum())
        precision, recall, boundary_f1 = _boundary_metrics(pred[side], gt[side])
        gt_components, gt_largest = _component_metrics(gt[side], gt_area)
        pred_components, pred_largest = _component_metrics(pred[side], gt_area)

        if gt_present and region_iou >= BOUNDARY_MIN_REGION_IOU and boundary_f1 < BOUNDARY_F1_THRESHOLD:
            labels.add("boundary_error")
            reasons.append(f"{side}_boundary_error")

        fragmented = gt_present and pred_components > gt_components and (
            pred_largest < LARGEST_COMPONENT_RATIO_THRESHOLD
            or gt_largest - pred_largest > LARGEST_COMPONENT_DROP_THRESHOLD
        )
        if fragmented:
            labels.add("spatial_inconsistency/fragmented_mask")
            reasons.append(f"{side}_fragmented_mask")

        image_area = gt[side].size
        catastrophic = (
            gt_present and (region_iou < CATASTROPHIC_IOU_THRESHOLD or not pred_present)
        ) or (
            not gt_present and (pred_present or pred_area / image_area >= ABSENT_MASK_AREA_THRESHOLD)
        )
        if catastrophic:
            labels.add("catastrophic_failure")
            reasons.append(f"{side}_catastrophic_failure")

        metrics[side] = {
            "gt_present": gt_present, "pred_present": pred_present,
            "object_probability": object_probability, "region_iou": region_iou if gt_present else None,
            "gt_area": gt_area, "pred_area": pred_area,
            "boundary_precision": precision if gt_present else None,
            "boundary_recall": recall if gt_present else None,
            "boundary_f1": boundary_f1 if gt_present else None,
            "gt_components": gt_components, "pred_components": pred_components,
            "gt_largest_component_ratio": gt_largest,
            "pred_largest_component_ratio": pred_largest,
        }

    for side, other in (("left", "right"), ("right", "left")):
        leakage = float((pred[side] & gt[other]).sum() / max(int(pred[side].sum()), 1))
        metrics[side]["cross_hand_leakage"] = leakage
        if leakage >= CROSS_HAND_LEAKAGE_THRESHOLD:
            labels.add("spatial_inconsistency/cross_hand_leakage")
            reasons.append(f"{side}_cross_hand_leakage")

    low_iou_sides = [
        side for side in ("left", "right")
        if metrics[side]["gt_present"] and metrics[side]["region_iou"] < LOW_REGION_IOU_THRESHOLD
    ]
    if low_iou_sides:
        labels.add("low_region_iou")
        reasons.extend(f"{side}_low_region_iou" for side in low_iou_sides)

    state = {
        "dataset_name": batch["dataset_name"][index], "sample_id": batch["sample_id"][index],
        "image_path": batch["image_path"][index], "stream_id": batch["stream_id"][index],
        "frame_position": batch["frame_position"][index],
        "source_frame_number": batch["source_frame_number"][index],
        "image": batch["original_image"][index],
        "pred": pred, "gt": gt, "metrics": metrics,
    }
    if not labels:
        return None, state
    return {
        "dataset_name": state["dataset_name"], "sample_id": state["sample_id"],
        "image_path": state["image_path"], "stream_id": state["stream_id"],
        "frame_position": state["frame_position"],
        "source_frame_number": state["source_frame_number"], "labels": sorted(labels),
        "reasons": reasons, "hands": metrics,
    }, state


def analyze_transition(previous: dict[str, Any] | None, current: dict[str, Any]) -> dict[str, Any] | None:
    if previous is None or previous["stream_id"] != current["stream_id"]:
        return None
    previous_number = previous["source_frame_number"]
    current_number = current["source_frame_number"]
    if previous_number is None or current_number != previous_number + 1:
        return None

    hands, score = {}, 0.0
    for side in ("left", "right"):
        prev_m, curr_m = previous["metrics"][side], current["metrics"][side]
        if not prev_m["gt_present"] or not curr_m["gt_present"]:
            continue
        gt_iou = _mask_iou(previous["gt"][side], current["gt"][side])
        pred_iou = _mask_iou(previous["pred"][side], current["pred"][side])
        prev_coverage = prev_m["pred_area"] / max(prev_m["gt_area"], 1)
        curr_coverage = curr_m["pred_area"] / max(curr_m["gt_area"], 1)
        coverage_jump = abs(curr_coverage - prev_coverage)
        quality_jump = abs(curr_m["region_iou"] - prev_m["region_iou"])
        excess_change = gt_iou - pred_iou
        object_flip = prev_m["pred_present"] != curr_m["pred_present"]
        is_jump = gt_iou >= TEMPORAL_GT_IOU_MIN and (
            pred_iou <= TEMPORAL_PRED_IOU_MAX or excess_change >= TEMPORAL_EXCESS_CHANGE_THRESHOLD
            or coverage_jump >= TEMPORAL_COVERAGE_JUMP_THRESHOLD
            or quality_jump >= TEMPORAL_QUALITY_JUMP_THRESHOLD or object_flip
        )
        if is_jump:
            hands[side] = {
                "gt_temporal_iou": gt_iou, "pred_temporal_iou": pred_iou,
                "excess_change": excess_change, "previous_coverage": prev_coverage,
                "current_coverage": curr_coverage, "coverage_jump": coverage_jump,
                "quality_jump": quality_jump, "object_presence_flip": object_flip,
            }
            score = max(score, 1 - pred_iou, excess_change, coverage_jump, quality_jump, float(object_flip))

    if not hands:
        return None
    return {
        "dataset_name": current["dataset_name"], "stream_id": current["stream_id"],
        "previous_sample_id": previous["sample_id"], "current_sample_id": current["sample_id"],
        "previous_source_frame_number": previous_number,
        "current_source_frame_number": current_number,
        "previous_frame_position": previous["frame_position"],
        "current_frame_position": current["frame_position"], "score": score, "hands": hands,
    }


def _safe_name(value: str) -> str:
    return value.replace("/", "__").replace("\\", "__")


def _save_five_panel(state: dict[str, Any], path: Path) -> None:
    save_dual_hand_five_panel_visualization(
        original_image=state["image"],
        left_pred_mask=torch.from_numpy(state["pred"]["left"]),
        right_pred_mask=torch.from_numpy(state["pred"]["right"]),
        left_gt_mask=torch.from_numpy(state["gt"]["left"]),
        right_gt_mask=torch.from_numpy(state["gt"]["right"]),
        save_path=path,
    )


def save_frame_case(record: dict[str, Any], state: dict[str, Any], output_dir: Path) -> None:
    name = f"{_safe_name(record['sample_id'])}.png"
    for label in record["labels"]:
        _save_five_panel(state, output_dir / label / _safe_name(record["dataset_name"]) / name)


def save_temporal_case(
    event: dict[str, Any], previous: dict[str, Any], current: dict[str, Any], output_dir: Path,
) -> None:
    path = (
        output_dir / "temporal_jump" / _safe_name(event["dataset_name"])
        / _safe_name(event["stream_id"])
        / f"{event['previous_frame_position']:06d}_to_{event['current_frame_position']:06d}"
    )
    _save_five_panel(previous, path / "previous.png")
    _save_five_panel(current, path / "current.png")
    dump_json(event, path / "metrics.json")


@torch.inference_mode()
def run_analysis(model: torch.nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> None:
    frame_cases, temporal_cases, previous = [], [], None
    counts, hardcase_ids = Counter(), set()
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
        left_object = outputs["left"]["object_score_logits"].reshape(batch_size, -1)[:, 0]
        right_object = outputs["right"]["object_score_logits"].reshape(batch_size, -1)[:, 0]
        for index in range(batch_size):
            left_gt = batch["original_left_mask"][index].unsqueeze(0).to(device)
            right_gt = batch["original_right_mask"][index].unsqueeze(0).to(device)
            size = left_gt.shape[-2:]
            left_logits = upsample_logits(outputs["left"]["high_res_masks"][index:index + 1], size)
            right_logits = upsample_logits(outputs["right"]["high_res_masks"][index:index + 1], size)
            record, current = analyze_frame(
                batch, index, left_logits, right_logits, left_gt, right_gt,
                left_object[index], right_object[index],
            )
            if record is not None:
                frame_cases.append(record)
                counts.update(record["labels"])
                hardcase_ids.add((record["dataset_name"], record["sample_id"]))
                save_frame_case(record, current, args.output_dir)

            event = analyze_transition(previous, current)
            if event is not None:
                temporal_cases.append(event)
                counts["temporal_jump"] += 1
                hardcase_ids.update({
                    (event["dataset_name"], event["previous_sample_id"]),
                    (event["dataset_name"], event["current_sample_id"]),
                })
                save_temporal_case(event, previous, current, args.output_dir)
            previous = current

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            logging.info("Scan %d/%d | frame=%d | temporal=%d", step, len(loader), len(frame_cases), len(temporal_cases))

    dump_json({
        "dataset": args.dataset, "dataset_names": args.dataset_names,
        "test_seq_count": args.test_seq_count, "dex_ycb_setup": args.dex_ycb_setup,
        "use_point_prompt": args.use_point_prompt,
        "model_checkpoint": str(args.model_checkpoint) if args.model_checkpoint else None,
        "frame_hardcases": frame_cases, "temporal_jumps": temporal_cases,
    }, args.output_dir / "hardcases.json")
    dump_json({
        "validation_samples": len(loader.dataset), "hardcase_frames": len(hardcase_ids),
        "frame_hardcases": len(frame_cases), "temporal_jumps": len(temporal_cases),
        "category_counts": dict(counts),
    }, args.output_dir / "summary.json")


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
