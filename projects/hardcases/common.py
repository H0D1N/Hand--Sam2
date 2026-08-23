"""模型无关的双手分割困难样本分析与结果保存。"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from projects.framewise_sam2_modified.utils import dump_json
from projects.framewise_sam2_modified.visualization import (
    save_dual_hand_four_panel_visualization,
)


# Basic decisions
MASK_LOGIT_THRESHOLD = 0.0
GT_MASK_THRESHOLD = 0.5

# GT / prediction states
GT_MIN_AREA_PIXELS = 64
GT_MIN_AREA_RATIO = 0.0005
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


def _bool_mask(tensor: torch.Tensor, threshold: float) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().squeeze().gt(threshold).cpu().numpy())


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
    min_area = max(GT_MIN_AREA_PIXELS, math.ceil(image_area * GT_MIN_AREA_RATIO))
    if gt_area == 0:
        return "empty", min_area
    if gt_area < min_area:
        return "tiny", min_area
    return "valid", min_area


def _prediction_state(pred_area: int, gt_area: int, image_area: int, gt_state: str) -> tuple[str, int]:
    if gt_state == "valid":
        min_area = max(PRED_MIN_AREA_PIXELS, math.ceil(gt_area * PRED_MIN_GT_AREA_RATIO))
    else:
        min_area = max(PRED_MIN_AREA_PIXELS, math.ceil(image_area * EMPTY_GT_PRED_MIN_AREA_RATIO))
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
    return max(COMPONENT_MIN_AREA_PIXELS, math.ceil(reference_area * COMPONENT_MIN_AREA_RATIO))


def _component_metrics(mask: np.ndarray, gt_area: int) -> tuple[int, float]:
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA]
    areas = areas[areas >= _minimum_component_area(gt_area)]
    largest_ratio = int(areas.max()) / int(areas.sum()) if areas.size else 0.0
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
            "wrong_hand_ratio": None, "wrong_hand_area": None,
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
        metrics=state["metrics"],
        save_path=path,
    )


def save_frame_case(record: dict[str, Any], state: dict[str, Any], output_dir: Path) -> None:
    name = f"{_safe_name(record['sample_id'])}.png"
    issue_types = sorted({(issue["category"], issue["reason"]) for issue in record["issues"]})
    for category, reason in issue_types:
        path = output_dir / category
        if category in {"region_error", "spatial_error"}:
            path /= reason
        _save_four_panel(state, path / _safe_name(record["dataset_name"]) / name)


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
