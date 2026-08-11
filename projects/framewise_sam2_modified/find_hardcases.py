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
from .utils import configure_runtime, dump_json, set_seed, upsample_logits
from .visualization import save_dual_hand_four_panel_visualization


REPO_ROOT = Path(__file__).resolve().parents[2]

# Basic decisions
MASK_LOGIT_THRESHOLD = 0.0
GT_MASK_THRESHOLD = 0.5

# GT / prediction states
GT_MIN_AREA_PIXELS = 64
GT_MIN_AREA_RATIO = 0.0001
PRED_MIN_AREA_PIXELS = 16
PRED_MIN_GT_AREA_RATIO = 0.01
EMPTY_GT_PRED_MIN_AREA_RATIO = 0.0005

# Region error
REGION_IOU_THRESHOLD = 0.70
REGION_PRECISION_THRESHOLD = 0.80
REGION_RECALL_THRESHOLD = 0.80

# Boundary error
BOUNDARY_F1_THRESHOLD = 0.65
BOUNDARY_MATCH_TOLERANCE_RATIO = 0.002
BOUNDARY_ERROR_BAND_RATIO = 0.01
BOUNDARY_ERROR_IN_BAND_THRESHOLD = 0.80

# Spatial inconsistency
COMPONENT_MIN_AREA_PIXELS = 16
COMPONENT_MIN_AREA_RATIO = 0.005
LARGEST_COMPONENT_RATIO_THRESHOLD = 0.80
LARGEST_COMPONENT_DROP_THRESHOLD = 0.15
CROSS_HAND_LEAKAGE_THRESHOLD = 0.10

# Temporal jump
TEMPORAL_GT_IOU_MIN = 0.75
TEMPORAL_EXCESS_CHANGE_THRESHOLD = 0.30
TEMPORAL_AREA_EXCESS_THRESHOLD = 0.30

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


def _region_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    intersection = int(np.logical_and(pred, gt).sum())
    pred_area, gt_area = int(pred.sum()), int(gt.sum())
    union = pred_area + gt_area - intersection
    iou = 1.0 if union == 0 else intersection / union
    precision = 1.0 if pred_area == 0 and gt_area == 0 else intersection / max(pred_area, 1)
    recall = 1.0 if gt_area == 0 and pred_area == 0 else intersection / max(gt_area, 1)
    return float(iou), float(precision), float(recall)


def _gt_state(gt_area: int, image_area: int) -> tuple[str, int]:
    min_area = max(GT_MIN_AREA_PIXELS, round(image_area * GT_MIN_AREA_RATIO))
    if gt_area == 0:
        return "empty", min_area
    if gt_area < min_area:
        return "tiny", min_area
    return "valid", min_area


def _prediction_state(pred_area: int, gt_area: int, image_area: int, gt_state: str) -> tuple[str, int]:
    if gt_state == "valid":
        min_area = max(PRED_MIN_AREA_PIXELS, round(gt_area * PRED_MIN_GT_AREA_RATIO))
    else:
        min_area = max(PRED_MIN_AREA_PIXELS, round(image_area * EMPTY_GT_PRED_MIN_AREA_RATIO))
    if pred_area == 0:
        return "empty", min_area
    if pred_area < min_area:
        return "tiny", min_area
    return "valid", min_area


def _boundary_radius(mask: np.ndarray, ratio: float) -> int:
    return max(1, round(math.hypot(*mask.shape) * ratio))


def _boundary_metrics(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    kernel = np.ones((3, 3), dtype=np.uint8)
    pred_edge = pred & (cv2.erode(pred.astype(np.uint8), kernel, borderValue=0) == 0)
    gt_edge = gt & (cv2.erode(gt.astype(np.uint8), kernel, borderValue=0) == 0)
    pred_count, gt_count = int(pred_edge.sum()), int(gt_edge.sum())
    if pred_count == 0 and gt_count == 0:
        return 1.0, 1.0, 1.0

    radius = _boundary_radius(gt, BOUNDARY_MATCH_TOLERANCE_RATIO)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    pred_band = cv2.dilate(pred_edge.astype(np.uint8), kernel).astype(bool)
    gt_band = cv2.dilate(gt_edge.astype(np.uint8), kernel).astype(bool)
    precision = float((pred_edge & gt_band).sum() / pred_count) if pred_count else 0.0
    recall = float((gt_edge & pred_band).sum() / gt_count) if gt_count else 0.0
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)
    return precision, recall, f1


def _boundary_error_ratio(pred: np.ndarray, gt: np.ndarray) -> float:
    error = np.logical_xor(pred, gt)
    error_area = int(error.sum())
    if error_area == 0:
        return 1.0

    radius = _boundary_radius(gt, BOUNDARY_ERROR_BAND_RATIO)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    gt_u8 = gt.astype(np.uint8)
    outer = cv2.dilate(gt_u8, kernel).astype(bool)
    inner = cv2.erode(gt_u8, kernel, borderValue=0).astype(bool)
    boundary_band = np.logical_and(outer, np.logical_not(inner))
    return float(np.logical_and(error, boundary_band).sum() / error_area)


def _minimum_component_area(reference_area: int) -> int:
    return max(COMPONENT_MIN_AREA_PIXELS, round(reference_area * COMPONENT_MIN_AREA_RATIO))


def _component_metrics(mask: np.ndarray, gt_area: int) -> tuple[int, float]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= _minimum_component_area(gt_area)]
    largest_ratio = int(areas.max()) / max(int(mask.sum()), 1) if areas.size else 0.0
    return int(areas.size), largest_ratio


def analyze_frame(
    batch: dict[str, Any], index: int, left_logits: torch.Tensor, right_logits: torch.Tensor,
    left_gt: torch.Tensor, right_gt: torch.Tensor,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    logits = {"left": left_logits, "right": right_logits}
    gt_tensors = {"left": left_gt, "right": right_gt}
    pred = {side: _bool_mask(logits[side], MASK_LOGIT_THRESHOLD) for side in ("left", "right")}
    gt = {side: _bool_mask(gt_tensors[side], GT_MASK_THRESHOLD) for side in ("left", "right")}
    issues: list[dict[str, Any]] = []
    hand_categories = {"left": set(), "right": set()}
    metrics: dict[str, dict[str, Any]] = {}

    def add_issue(category: str, side: str, reason: str, **details: Any) -> None:
        issue = {"category": category, "hand": side, "reason": reason}
        issue.update(details)
        issues.append(issue)
        hand_categories[side].add(category)

    for side in ("left", "right"):
        gt_area, pred_area = int(gt[side].sum()), int(pred[side].sum())
        image_area = gt[side].size
        gt_state, gt_min_area = _gt_state(gt_area, image_area)
        pred_state, pred_min_area = _prediction_state(pred_area, gt_area, image_area, gt_state)
        side_metrics = {
            "gt_state": gt_state, "pred_state": pred_state, "ignored": gt_state == "tiny",
            "gt_area": gt_area, "pred_area": pred_area, "image_area": image_area,
            "gt_min_area": gt_min_area, "pred_min_area": pred_min_area,
            "region_iou": None, "region_precision": None, "region_recall": None,
            "boundary_precision": None, "boundary_recall": None, "boundary_f1": None,
            "boundary_error_ratio": None,
            "gt_components": None, "pred_components": None,
            "gt_largest_component_ratio": None, "pred_largest_component_ratio": None,
            "wrong_hand_ratio": 0.0, "wrong_hand_area": 0,
        }
        metrics[side] = side_metrics

        if gt_state == "tiny":
            continue

        if gt_state == "empty" and pred_state == "valid":
            add_issue("region_error", side, "mask_without_gt", needs_gt_review=True)
            continue
        if gt_state == "empty":
            continue
        if pred_state != "valid":
            add_issue("region_error", side, "no_mask")
            continue

        region_iou, region_precision, region_recall = _region_metrics(pred[side], gt[side])
        side_metrics.update(
            region_iou=region_iou,
            region_precision=region_precision,
            region_recall=region_recall,
        )
        if (
            region_iou < REGION_IOU_THRESHOLD
            or region_precision < REGION_PRECISION_THRESHOLD
            or region_recall < REGION_RECALL_THRESHOLD
        ):
            contained = (
                region_precision >= REGION_PRECISION_THRESHOLD
                and region_recall < REGION_RECALL_THRESHOLD
            ) or (
                region_recall >= REGION_RECALL_THRESHOLD
                and region_precision < REGION_PRECISION_THRESHOLD
            )
            add_issue("region_error", side, "size_mismatch" if contained else "poor_overlap")

        gt_components, gt_largest = _component_metrics(gt[side], gt_area)
        pred_components, pred_largest = _component_metrics(pred[side], gt_area)
        side_metrics.update(
            gt_components=gt_components,
            pred_components=pred_components,
            gt_largest_component_ratio=gt_largest,
            pred_largest_component_ratio=pred_largest,
        )
        fragmented = pred_components > gt_components and (
            pred_largest < LARGEST_COMPONENT_RATIO_THRESHOLD
            or gt_largest - pred_largest > LARGEST_COMPONENT_DROP_THRESHOLD
        )
        if fragmented:
            add_issue("spatial_error", side, "fragmented_mask")

    for side, other in (("left", "right"), ("right", "left")):
        side_metrics, other_metrics = metrics[side], metrics[other]
        if side_metrics["gt_state"] == "tiny" or side_metrics["pred_state"] != "valid":
            continue
        if other_metrics["gt_state"] != "valid":
            continue
        wrong_area = int(np.logical_and(pred[side], gt[other]).sum())
        wrong_ratio = wrong_area / max(side_metrics["pred_area"], 1)
        side_metrics["wrong_hand_area"] = wrong_area
        side_metrics["wrong_hand_ratio"] = wrong_ratio
        if (
            wrong_area >= _minimum_component_area(other_metrics["gt_area"])
            and wrong_ratio >= CROSS_HAND_LEAKAGE_THRESHOLD
        ):
            add_issue("spatial_error", side, "wrong_hand", target_hand=other)

    for side in ("left", "right"):
        side_metrics = metrics[side]
        if side_metrics["gt_state"] != "valid" or side_metrics["pred_state"] != "valid":
            continue
        if hand_categories[side] & {"region_error", "spatial_error"}:
            continue
        boundary_precision, boundary_recall, boundary_f1 = _boundary_metrics(pred[side], gt[side])
        boundary_error_ratio = _boundary_error_ratio(pred[side], gt[side])
        side_metrics.update(
            boundary_precision=boundary_precision,
            boundary_recall=boundary_recall,
            boundary_f1=boundary_f1,
            boundary_error_ratio=boundary_error_ratio,
        )
        if (
            boundary_f1 < BOUNDARY_F1_THRESHOLD
            and boundary_error_ratio >= BOUNDARY_ERROR_IN_BAND_THRESHOLD
        ):
            add_issue("boundary_error", side, "boundary_error")

    state = {
        "dataset_name": batch["dataset_name"][index], "sample_id": batch["sample_id"][index],
        "image_path": batch["image_path"][index], "stream_id": batch["stream_id"][index],
        "frame_position": batch["frame_position"][index],
        "source_frame_number": batch["source_frame_number"][index],
        "image": batch["original_image"][index],
        "pred": pred, "gt": gt, "metrics": metrics,
    }
    if not issues:
        return None, state
    categories = sorted({issue["category"] for issue in issues})
    return {
        "dataset_name": state["dataset_name"], "sample_id": state["sample_id"],
        "image_path": state["image_path"], "stream_id": state["stream_id"],
        "frame_position": state["frame_position"],
        "source_frame_number": state["source_frame_number"],
        "categories": categories, "issues": issues, "hands": metrics,
    }, state


def _normalized_area_change(previous_area: int, current_area: int) -> float:
    return abs(current_area - previous_area) / max(previous_area, current_area, 1)


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
        if prev_m["gt_state"] != "valid" or curr_m["gt_state"] != "valid":
            continue
        gt_iou = _mask_iou(previous["gt"][side], current["gt"][side])
        if gt_iou < TEMPORAL_GT_IOU_MIN:
            continue

        pred_iou = _mask_iou(previous["pred"][side], current["pred"][side])
        excess_change = gt_iou - pred_iou
        gt_area_change = _normalized_area_change(prev_m["gt_area"], curr_m["gt_area"])
        pred_area_change = _normalized_area_change(prev_m["pred_area"], curr_m["pred_area"])
        area_excess = pred_area_change - gt_area_change
        is_jump = (
            excess_change >= TEMPORAL_EXCESS_CHANGE_THRESHOLD
            or area_excess >= TEMPORAL_AREA_EXCESS_THRESHOLD
        )
        if is_jump:
            hands[side] = {
                "gt_temporal_iou": gt_iou, "pred_temporal_iou": pred_iou,
                "excess_change": excess_change,
                "previous_gt_area": prev_m["gt_area"], "current_gt_area": curr_m["gt_area"],
                "previous_pred_area": prev_m["pred_area"], "current_pred_area": curr_m["pred_area"],
                "gt_area_change": gt_area_change, "pred_area_change": pred_area_change,
                "area_excess": area_excess,
                "previous_region_iou": prev_m["region_iou"],
                "current_region_iou": curr_m["region_iou"],
            }
            score = max(score, excess_change, area_excess)

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


def _save_four_panel(state: dict[str, Any], path: Path) -> None:
    save_dual_hand_four_panel_visualization(
        original_image=state["image"],
        left_pred_mask=torch.from_numpy(state["pred"]["left"]),
        right_pred_mask=torch.from_numpy(state["pred"]["right"]),
        left_gt_mask=torch.from_numpy(state["gt"]["left"]),
        right_gt_mask=torch.from_numpy(state["gt"]["right"]),
        save_path=path,
    )


def save_frame_case(record: dict[str, Any], state: dict[str, Any], output_dir: Path) -> None:
    name = f"{_safe_name(record['sample_id'])}.png"
    for category in record["categories"]:
        _save_four_panel(state, output_dir / category / _safe_name(record["dataset_name"]) / name)


def save_temporal_case(
    event: dict[str, Any], previous: dict[str, Any], current: dict[str, Any], output_dir: Path,
) -> None:
    path = (
        output_dir / "temporal_jump" / _safe_name(event["dataset_name"])
        / _safe_name(event["stream_id"])
        / f"{event['previous_source_frame_number']:06d}_to_{event['current_source_frame_number']:06d}"
    )
    _save_four_panel(previous, path / "previous.png")
    _save_four_panel(current, path / "current.png")
    dump_json(event, path / "metrics.json")


@torch.inference_mode()
def run_analysis(model: torch.nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> None:
    frame_cases, temporal_cases, previous = [], [], None
    category_counts, reason_counts, gt_state_counts = Counter(), Counter(), Counter()
    hardcase_ids, processed_samples = set(), 0
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
            left_gt = batch["original_left_mask"][index].unsqueeze(0).to(device)
            right_gt = batch["original_right_mask"][index].unsqueeze(0).to(device)
            size = left_gt.shape[-2:]
            left_logits = upsample_logits(outputs["left"]["high_res_masks"][index:index + 1], size)
            right_logits = upsample_logits(outputs["right"]["high_res_masks"][index:index + 1], size)
            record, current = analyze_frame(
                batch, index, left_logits, right_logits, left_gt, right_gt,
            )
            processed_samples += 1
            gt_state_counts.update(current["metrics"][side]["gt_state"] for side in ("left", "right"))
            if record is not None:
                frame_cases.append(record)
                category_counts.update(record["categories"])
                reason_counts.update(
                    f"{issue['category']}/{issue['reason']}"
                    for issue in record["issues"]
                )
                hardcase_ids.add((record["dataset_name"], record["sample_id"]))
                save_frame_case(record, current, args.output_dir)

            event = analyze_transition(previous, current)
            if event is not None:
                temporal_cases.append(event)
                category_counts["temporal_jump"] += 1
                hardcase_ids.update({
                    (event["dataset_name"], event["previous_sample_id"]),
                    (event["dataset_name"], event["current_sample_id"]),
                })
                save_temporal_case(event, previous, current, args.output_dir)
            previous = current

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            logging.info(
                "Scan samples=%d/%d | frame=%d | temporal=%d | categories=%s",
                processed_samples, len(loader.dataset), len(frame_cases), len(temporal_cases),
                dict(category_counts),
            )

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
        "category_counts": dict(category_counts), "reason_counts": dict(reason_counts),
        "gt_state_counts": dict(gt_state_counts),
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
