"""定量分析 Memory tracking、单帧预测和纠错是否有效。"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from statistics import fmean, median

import torch

from projects.dual_hand_memory.builder import build_sam2_dual_hand_memory_tiny
from projects.framewise_sam2_modified.dataset import build_center_point_prompt
from projects.framewise_sam2_modified.losses import iou_target_from_logits
from projects.framewise_sam2_modified.utils import configure_runtime, dump_json, set_seed, upsample_logits
from projects.framewise_sam2_modified.visualization import (
    save_dual_hand_correction_visualization,
    save_dual_hand_memory_comparison_visualization,
)
from projects.hardcases.common import analyze_frame, analyze_transition, save_frame_case, save_temporal_case
from projects.zero_shot_common import build_zero_shot_loader, parse_args


MAX_TRACKING_VISUALIZATIONS_PER_DATASET = 25
MAX_EXPERIMENT_VISUALIZATIONS_PER_DATASET = 10


def load_model(args, device: torch.device) -> torch.nn.Module:
    """按训练参数重建模型，并加载完整的 Memory checkpoint。"""

    checkpoint = torch.load(args.sam_checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    model = build_sam2_dual_hand_memory_tiny(
        sam_checkpoint=saved_args["sam_checkpoint"],
        framewise_checkpoint=saved_args["framewise_checkpoint"],
        device=device, mode="eval", image_size=saved_args["image_size"],
        use_image_adapter=saved_args["use_image_adapter"],
        use_decoder_adapter=saved_args["use_decoder_adapter"],
        adapter_dim=saved_args["adapter_dim"], adapter_dropout=saved_args["adapter_dropout"],
        adapter_init_scale=saved_args["adapter_init_scale"],
        num_init_cond_frames_for_train=saved_args["num_init_cond_frames_for_train"],
        num_frames_to_correct_for_train=saved_args["num_frames_to_correct_for_train"],
        add_all_frames_to_correct_as_cond=saved_args["add_all_frames_to_correct_as_cond"],
        num_correction_pt_per_frame=saved_args["num_correction_pt_per_frame"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    args.image_size = model.image_size
    logging.info("Loaded Memory checkpoint | %s | epoch=%s", args.sam_checkpoint, checkpoint.get("epoch", "unknown"))
    return model.eval()


def _new_stats() -> dict:
    return {
        "tracking": {
            "analyzed_frames": 0, "conditioning_frames": 0, "valid_hand_ious": [],
            "hardcase_frames": 0, "category_counts": Counter(), "reason_counts": Counter(),
            "temporal_comparisons": 0, "temporal_jumps": 0, "ious_by_frame_position": {},
        },
        "memory_vs_single_image": {
            "total_frames": 0, "memory_ious": [], "single_image_ious": [],
            "memory_wins": 0, "single_image_hardcase_frames": 0,
            "single_image_category_counts": Counter(),
        },
        "worst_frame_correction": {
            "total_frames": 0, "step_0_ious": [], "step_1_ious": [], "final_ious": [],
            "improved_hands": 0, "resolved_hardcase_frames": 0,
            "introduced_categories": Counter(), "resolved_categories": Counter(),
        },
        "prompt_reset_correction": {
            "total_frames": 0, "memory_ious": [], "prompt_ious": [], "final_ious": [],
            "prompt_worse_frames": 0, "introduced_hardcase_frames": 0,
            "recovered_hardcase_frames": 0, "introduced_categories": Counter(),
            "resolved_categories": Counter(),
        },
    }


def _dataset_group(dataset_name: str) -> str:
    return "dexycb" if dataset_name.lower() == "dexycb" else "multiserver"


def _frame_batch(batch: dict, sample_index: int, frame_index: int) -> dict:
    """把 clip 中的一帧整理成 common.py 使用的单帧格式。"""

    return {
        "dataset_name": [batch["dataset_name"][sample_index]],
        "sample_id": [batch["sample_id"][sample_index][frame_index]],
        "image_path": [batch["image_path"][sample_index][frame_index]],
        "stream_id": [batch["stream_id"][sample_index]],
        "source_frame_number": [batch["frame_numbers"][sample_index][frame_index]],
        "original_image": [batch["original_image"][sample_index][frame_index]],
    }


def _analyze_logits(
    batch: dict, sample_index: int, frame_index: int,
    left_logits: torch.Tensor, right_logits: torch.Tensor,
) -> tuple[dict | None, dict, torch.Tensor, torch.Tensor]:
    """恢复到原图尺寸后，复用统一的 hardcase 分类。"""

    device = left_logits.device
    left_gt = batch["original_left_mask"][sample_index][frame_index].unsqueeze(0).to(device)
    right_gt = batch["original_right_mask"][sample_index][frame_index].unsqueeze(0).to(device)
    original_size = left_gt.shape[-2:]
    left_logits = upsample_logits(left_logits, original_size)
    right_logits = upsample_logits(right_logits, original_size)
    record, state = analyze_frame(
        _frame_batch(batch, sample_index, frame_index), 0,
        left_logits, right_logits, left_gt, right_gt,
    )
    return record, state, left_logits, right_logits


def _categories(record: dict | None) -> set[str]:
    return set() if record is None else {issue["category"] for issue in record["issues"]}


def _valid_ious(state: dict) -> list[float]:
    return [
        state["metrics"][side]["region_iou"] for side in ("left", "right")
        if state["metrics"][side]["gt_state"] == "valid"
    ]


def _paired_ious(*states: dict) -> list[tuple[float, ...]]:
    return [
        tuple(state["metrics"][side]["region_iou"] for state in states)
        for side in ("left", "right")
        if states[0]["metrics"][side]["gt_state"] == "valid"
    ]


def _update_category_change(values: dict, before: dict | None, after: dict | None) -> None:
    before_categories, after_categories = _categories(before), _categories(after)
    values["introduced_categories"].update(after_categories - before_categories)
    values["resolved_categories"].update(before_categories - after_categories)


def _safe_name(value) -> str:
    return str(value).replace("/", "__").replace("\\", "__")


def _save_correction_steps(
    outputs: list[dict], frame_index: int, image: torch.Tensor,
    left_gt: torch.Tensor, right_gt: torch.Tensor, initial_points: int,
    title: str, output_dir: Path,
) -> None:
    """保存 step 0、step 1 和最终纠错结果；点沿用模型真实输入。"""

    frame_output = outputs[frame_index]
    num_steps = frame_output["left"]["multistep_pred_masks_high_res"].size(1)
    steps = [("step_0", 0)]
    if num_steps > 1:
        steps.append(("step_1", 1))
    if num_steps > 2:
        steps.append(("final", num_steps - 1))

    for step_name, step_index in steps:
        left_output, right_output = frame_output["left"], frame_output["right"]
        left_logits = left_output["multistep_pred_masks_high_res"][:, step_index:step_index + 1]
        right_logits = right_output["multistep_pred_masks_high_res"][:, step_index:step_index + 1]
        left_points = left_output["multistep_point_inputs"][step_index]
        right_points = right_output["multistep_point_inputs"][step_index]
        save_dual_hand_correction_visualization(
            normalized_image=image, left_gt_mask=left_gt, right_gt_mask=right_gt,
            left_pred_mask=left_logits > 0, right_pred_mask=right_logits > 0,
            left_point_input=left_points, right_point_input=right_points,
            left_initial_points=initial_points, right_initial_points=initial_points,
            left_iou=iou_target_from_logits(left_logits, left_gt).item(),
            right_iou=iou_target_from_logits(right_logits, right_gt).item(),
            title=f"{title} | {step_name}", save_path=output_dir / f"{step_name}.png",
        )


def _save_experiment_visualizations(
    batch: dict, sample_index: int, prompt_frame: int, worst_frame: int,
    images: torch.Tensor, left_masks: torch.Tensor, right_masks: torch.Tensor,
    memory_logits: tuple[torch.Tensor, torch.Tensor],
    single_logits: tuple[torch.Tensor, torch.Tensor],
    left_point_inputs: dict, right_point_inputs: dict, single_index: int,
    worst_outputs: list[dict], prompt_outputs: list[dict], output_dir: Path,
) -> None:
    """按数据集、序列和原始帧号保存少量代表性对照。"""

    dataset_name = batch["dataset_name"][sample_index]
    stream_id = batch["stream_id"][sample_index]
    base_dir = output_dir / "visualizations" / _safe_name(dataset_name) / _safe_name(stream_id)
    prompt_number = batch["frame_numbers"][sample_index][prompt_frame]
    prompt_dir = base_dir / f"frame_{prompt_number}"
    original_size = batch["original_left_mask"][sample_index][prompt_frame].shape[-2:]
    point_scale = images.new_tensor((original_size[1] / images.size(-1), original_size[0] / images.size(-2)))

    left_gt = batch["original_left_mask"][sample_index][prompt_frame].unsqueeze(0)
    right_gt = batch["original_right_mask"][sample_index][prompt_frame].unsqueeze(0)
    left_point = None if left_point_inputs["point_labels"][single_index].item() < 0 else left_point_inputs["point_coords"][single_index, 0] * point_scale
    right_point = None if right_point_inputs["point_labels"][single_index].item() < 0 else right_point_inputs["point_coords"][single_index, 0] * point_scale
    save_dual_hand_memory_comparison_visualization(
        original_image=batch["original_image"][sample_index][prompt_frame],
        left_gt_mask=left_gt, right_gt_mask=right_gt,
        left_no_memory_mask=single_logits[0] > 0, right_no_memory_mask=single_logits[1] > 0,
        left_memory_mask=memory_logits[0] > 0, right_memory_mask=memory_logits[1] > 0,
        left_no_memory_point=left_point, right_no_memory_point=right_point,
        title="Memory vs single image", save_path=prompt_dir / "memory_vs_single_image.png",
    )

    prompt_left_gt = left_masks[sample_index, prompt_frame:prompt_frame + 1]
    prompt_right_gt = right_masks[sample_index, prompt_frame:prompt_frame + 1]
    _save_correction_steps(
        prompt_outputs, prompt_frame, images[sample_index, prompt_frame], prompt_left_gt, prompt_right_gt,
        initial_points=1, title="Prompt reset correction", output_dir=prompt_dir / "prompt_reset_correction",
    )

    worst_number = batch["frame_numbers"][sample_index][worst_frame]
    worst_dir = base_dir / f"frame_{worst_number}" / "worst_frame_correction"
    worst_left_gt = left_masks[sample_index, worst_frame:worst_frame + 1]
    worst_right_gt = right_masks[sample_index, worst_frame:worst_frame + 1]
    _save_correction_steps(
        worst_outputs, worst_frame, images[sample_index, worst_frame], worst_left_gt, worst_right_gt,
        initial_points=0, title="Worst Memory frame correction", output_dir=worst_dir,
    )


def _tracking_summary(values: dict) -> dict:
    frames, ious = values["analyzed_frames"], values["valid_hand_ious"]
    comparisons = values["temporal_comparisons"]
    return {
        "analyzed_frames": frames, "conditioning_frames": values["conditioning_frames"],
        "evaluated_hands": len(ious), "mean_iou": round(fmean(ious), 4) if ious else None,
        "median_iou": round(median(ious), 4) if ious else None,
        "hardcase_frames": values["hardcase_frames"],
        "hardcase_rate": round(values["hardcase_frames"] / frames, 4) if frames else None,
        "category_counts": dict(values["category_counts"]),
        "reason_counts": dict(values["reason_counts"]),
        "iou_by_frame_position": {
            str(position): round(fmean(position_ious), 4)
            for position, position_ious in values["ious_by_frame_position"].items()
        },
        "temporal_comparisons": comparisons, "temporal_jumps": values["temporal_jumps"],
        "temporal_jump_rate": round(values["temporal_jumps"] / comparisons, 4) if comparisons else None,
    }


def _summary(stats: dict) -> dict:
    result = {}
    for group, group_stats in stats.items():
        tracking = group_stats["tracking"]
        comparison = group_stats["memory_vs_single_image"]
        worst = group_stats["worst_frame_correction"]
        prompt = group_stats["prompt_reset_correction"]
        comparison_hands = len(comparison["memory_ious"])
        worst_hands = len(worst["step_0_ious"])
        prompt_hands = len(prompt["memory_ious"])
        result[group] = {
            "tracking": _tracking_summary(tracking),
            "memory_vs_single_image": {
                "total_frames": comparison["total_frames"], "evaluated_hands": comparison_hands,
                "memory_mean_iou": round(fmean(comparison["memory_ious"]), 4) if comparison_hands else None,
                "single_image_mean_iou": round(fmean(comparison["single_image_ious"]), 4) if comparison_hands else None,
                "mean_iou_gain": round(fmean(m - s for m, s in zip(comparison["memory_ious"], comparison["single_image_ious"])), 4) if comparison_hands else None,
                "memory_win_rate": round(comparison["memory_wins"] / comparison_hands, 4) if comparison_hands else None,
                "single_image_hardcase_frames": comparison["single_image_hardcase_frames"],
                "single_image_category_counts": dict(comparison["single_image_category_counts"]),
            },
            "worst_frame_correction": {
                "total_frames": worst["total_frames"], "evaluated_hands": worst_hands,
                "step_0_mean_iou": round(fmean(worst["step_0_ious"]), 4) if worst_hands else None,
                "step_1_mean_iou": round(fmean(worst["step_1_ious"]), 4) if worst["step_1_ious"] else None,
                "final_mean_iou": round(fmean(worst["final_ious"]), 4) if worst_hands else None,
                "improved_hands": worst["improved_hands"],
                "resolved_hardcase_frames": worst["resolved_hardcase_frames"],
                "category_changes": {
                    "introduced": dict(worst["introduced_categories"]),
                    "resolved": dict(worst["resolved_categories"]),
                },
            },
            "prompt_reset_correction": {
                "total_frames": prompt["total_frames"], "evaluated_hands": prompt_hands,
                "memory_mean_iou": round(fmean(prompt["memory_ious"]), 4) if prompt_hands else None,
                "prompt_mean_iou": round(fmean(prompt["prompt_ious"]), 4) if prompt_hands else None,
                "final_mean_iou": round(fmean(prompt["final_ious"]), 4) if prompt_hands else None,
                "prompt_worse_frames": prompt["prompt_worse_frames"],
                "introduced_hardcase_frames": prompt["introduced_hardcase_frames"],
                "recovered_hardcase_frames": prompt["recovered_hardcase_frames"],
                "category_changes": {
                    "introduced_by_prompt": dict(prompt["introduced_categories"]),
                    "resolved_by_correction": dict(prompt["resolved_categories"]),
                },
            },
        }
    return result


@torch.inference_mode()
def run_analysis(model: torch.nn.Module, loader, device: torch.device, args) -> None:
    """对同一批 clips 运行 tracking、单帧预测和两种受控纠错实验。"""

    stats = {name: _new_stats() for name in ("overall", "multiserver", "dexycb")}
    frame_cases, temporal_cases = [], []
    tracking_visualizations, experiment_visualizations = Counter(), Counter()
    clip_number = 0

    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)
        batch_size, num_frames = images.shape[:2]

        # 第 0 帧使用 GT mask；第 1～T-1 帧只使用历史 Memory。
        memory_outputs = model(images=images, left_masks=left_masks, right_masks=right_masks, prompt_mode="mask")

        # 同一 checkpoint 逐帧运行，每帧分别提供一个中心 point prompt。
        flat_images = images[:, 1:].reshape(-1, *images.shape[2:])
        flat_left_masks = left_masks[:, 1:].reshape(-1, *left_masks.shape[2:])
        flat_right_masks = right_masks[:, 1:].reshape(-1, *right_masks.shape[2:])
        left_points = build_center_point_prompt(flat_left_masks)
        right_points = build_center_point_prompt(flat_right_masks)
        single_outputs = model.forward_single_image(
            images=flat_images, left_point_inputs=left_points, right_point_inputs=right_points,
            mask_inputs=None, multimask_output=False,
        )

        for sample_index in range(batch_size):
            dataset_name = batch["dataset_name"][sample_index]
            group = _dataset_group(dataset_name)
            current_stats = (stats["overall"], stats[group])
            memory_by_frame, previous = {}, None

            for frame_index, output in enumerate(memory_outputs):
                memory_record, memory_state, memory_left, memory_right = _analyze_logits(
                    batch, sample_index, frame_index,
                    output["left"]["pred_masks_high_res"][sample_index:sample_index + 1],
                    output["right"]["pred_masks_high_res"][sample_index:sample_index + 1],
                )
                memory_by_frame[frame_index] = (memory_record, memory_state, memory_left, memory_right)

                if frame_index == 0:
                    for values in current_stats:
                        values["tracking"]["conditioning_frames"] += 1
                    previous = memory_state
                    continue

                single_index = sample_index * (num_frames - 1) + frame_index - 1
                single_record, single_state, _, _ = _analyze_logits(
                    batch, sample_index, frame_index,
                    single_outputs["left"]["high_res_masks"][single_index:single_index + 1],
                    single_outputs["right"]["high_res_masks"][single_index:single_index + 1],
                )
                categories = _categories(memory_record)
                reasons = [] if memory_record is None else [f"{issue['category']}/{issue['reason']}" for issue in memory_record["issues"]]
                frame_ious = _valid_ious(memory_state)
                pairs = _paired_ious(memory_state, single_state)
                event = analyze_transition(previous, memory_state)
                if memory_record is not None:
                    frame_cases.append(memory_record)
                if event is not None:
                    temporal_cases.append(event)

                for values in current_stats:
                    tracking = values["tracking"]
                    tracking["analyzed_frames"] += 1
                    tracking["valid_hand_ious"].extend(frame_ious)
                    tracking["hardcase_frames"] += memory_record is not None
                    tracking["category_counts"].update(categories)
                    tracking["reason_counts"].update(reasons)
                    if frame_ious:
                        tracking["ious_by_frame_position"].setdefault(frame_index, []).extend(frame_ious)
                    tracking["temporal_comparisons"] += 1
                    tracking["temporal_jumps"] += event is not None

                    comparison = values["memory_vs_single_image"]
                    comparison["total_frames"] += bool(pairs)
                    comparison["memory_ious"].extend(pair[0] for pair in pairs)
                    comparison["single_image_ious"].extend(pair[1] for pair in pairs)
                    comparison["memory_wins"] += sum(memory_iou > single_iou for memory_iou, single_iou in pairs)
                    comparison["single_image_hardcase_frames"] += single_record is not None
                    comparison["single_image_category_counts"].update(_categories(single_record))

                if args.save_visualizations and tracking_visualizations[dataset_name] < MAX_TRACKING_VISUALIZATIONS_PER_DATASET:
                    if memory_record is not None:
                        save_frame_case(memory_record, memory_state, args.output_dir / "visualizations" / "tracking")
                        tracking_visualizations[dataset_name] += 1
                    elif event is not None:
                        save_temporal_case(event, previous, memory_state, args.output_dir / "visualizations" / "tracking")
                        tracking_visualizations[dataset_name] += 1
                previous = memory_state

            tracking_frames = [
                (frame_index, fmean(_valid_ious(memory_by_frame[frame_index][1])))
                for frame_index in range(1, num_frames) if _valid_ious(memory_by_frame[frame_index][1])
            ]
            if not tracking_frames:
                clip_number += 1
                continue

            worst_frame = min(tracking_frames, key=lambda item: item[1])[0]
            prompt_frame = 1 + clip_number % (num_frames - 1)
            clip_number += 1
            sample_images = images[sample_index:sample_index + 1]
            sample_left_masks = left_masks[sample_index:sample_index + 1]
            sample_right_masks = right_masks[sample_index:sample_index + 1]

            # 真实差帧保持为普通 Memory frame，只追加纠错点。
            worst_outputs = model(
                images=sample_images, left_masks=sample_left_masks, right_masks=sample_right_masks,
                prompt_mode="mask", correction_frame_indices=[worst_frame],
            )
            worst_output = worst_outputs[worst_frame]
            worst_steps = worst_output["left"]["multistep_pred_masks_high_res"].size(1)
            step_1_index = min(1, worst_steps - 1)
            worst_step_0 = _analyze_logits(
                batch, sample_index, worst_frame,
                worst_output["left"]["multistep_pred_masks_high_res"][:, 0:1],
                worst_output["right"]["multistep_pred_masks_high_res"][:, 0:1],
            )
            worst_step_1 = _analyze_logits(
                batch, sample_index, worst_frame,
                worst_output["left"]["multistep_pred_masks_high_res"][:, step_1_index:step_1_index + 1],
                worst_output["right"]["multistep_pred_masks_high_res"][:, step_1_index:step_1_index + 1],
            )
            worst_final = _analyze_logits(
                batch, sample_index, worst_frame,
                worst_output["left"]["multistep_pred_masks_high_res"][:, -1:],
                worst_output["right"]["multistep_pred_masks_high_res"][:, -1:],
            )
            worst_pairs = _paired_ious(worst_step_0[1], worst_step_1[1], worst_final[1])
            for values in current_stats:
                correction = values["worst_frame_correction"]
                correction["total_frames"] += bool(worst_pairs)
                correction["step_0_ious"].extend(pair[0] for pair in worst_pairs)
                correction["step_1_ious"].extend(pair[1] for pair in worst_pairs)
                correction["final_ious"].extend(pair[2] for pair in worst_pairs)
                correction["improved_hands"] += sum(final > initial for initial, _, final in worst_pairs)
                correction["resolved_hardcase_frames"] += worst_step_0[0] is not None and worst_final[0] is None
                _update_category_change(correction, worst_step_0[0], worst_final[0])

            # 均匀轮换测试帧：额外 point 会把该帧重置为初始条件帧，然后再纠错。
            prompt_outputs = model(
                images=sample_images, left_masks=sample_left_masks, right_masks=sample_right_masks,
                prompt_mode="mask", extra_init_point_frame_indices=[prompt_frame],
                correction_frame_indices=[prompt_frame],
            )
            prompt_output = prompt_outputs[prompt_frame]
            prompt_step_0 = _analyze_logits(
                batch, sample_index, prompt_frame,
                prompt_output["left"]["multistep_pred_masks_high_res"][:, 0:1],
                prompt_output["right"]["multistep_pred_masks_high_res"][:, 0:1],
            )
            prompt_final = _analyze_logits(
                batch, sample_index, prompt_frame,
                prompt_output["left"]["multistep_pred_masks_high_res"][:, -1:],
                prompt_output["right"]["multistep_pred_masks_high_res"][:, -1:],
            )
            memory_record, memory_state = memory_by_frame[prompt_frame][:2]
            prompt_pairs = _paired_ious(memory_state, prompt_step_0[1], prompt_final[1])
            for values in current_stats:
                correction = values["prompt_reset_correction"]
                correction["total_frames"] += bool(prompt_pairs)
                correction["memory_ious"].extend(pair[0] for pair in prompt_pairs)
                correction["prompt_ious"].extend(pair[1] for pair in prompt_pairs)
                correction["final_ious"].extend(pair[2] for pair in prompt_pairs)
                if prompt_pairs:
                    correction["prompt_worse_frames"] += fmean(pair[1] for pair in prompt_pairs) < fmean(pair[0] for pair in prompt_pairs)
                    correction["introduced_hardcase_frames"] += memory_record is None and prompt_step_0[0] is not None
                    correction["recovered_hardcase_frames"] += prompt_step_0[0] is not None and prompt_final[0] is None
                    memory_categories = _categories(memory_record)
                    prompt_categories = _categories(prompt_step_0[0])
                    final_categories = _categories(prompt_final[0])
                    correction["introduced_categories"].update(prompt_categories - memory_categories)
                    correction["resolved_categories"].update(prompt_categories - final_categories)

            if args.save_visualizations and experiment_visualizations[dataset_name] < MAX_EXPERIMENT_VISUALIZATIONS_PER_DATASET:
                single_index = sample_index * (num_frames - 1) + prompt_frame - 1
                _, _, single_left, single_right = _analyze_logits(
                    batch, sample_index, prompt_frame,
                    single_outputs["left"]["high_res_masks"][single_index:single_index + 1],
                    single_outputs["right"]["high_res_masks"][single_index:single_index + 1],
                )
                memory_left, memory_right = memory_by_frame[prompt_frame][2:]
                _save_experiment_visualizations(
                    batch, sample_index, prompt_frame, worst_frame, images, left_masks, right_masks,
                    (memory_left, memory_right), (single_left, single_right),
                    left_points, right_points, single_index,
                    worst_outputs, prompt_outputs, args.output_dir,
                )
                experiment_visualizations[dataset_name] += 1

        if step % max(args.log_interval, 1) == 0 or step == len(loader):
            overall = stats["overall"]
            logging.info(
                "Clips=%d/%d | tracking_frames=%d | hardcases=%d | correction_frames=%d",
                min(step * args.val_batch_size, len(loader.dataset)), len(loader.dataset),
                overall["tracking"]["analyzed_frames"], overall["tracking"]["hardcase_frames"],
                overall["worst_frame_correction"]["total_frames"],
            )

    dump_json({
        "dataset": args.dataset_mode, "model_checkpoint": str(args.sam_checkpoint),
        "clip_length": args.clip_length, "frame_cases": frame_cases, "temporal_cases": temporal_cases,
    }, args.output_dir / "hardcases.json")
    dump_json(_summary(stats), args.output_dir / "summary.json")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)
    model = load_model(args, device)
    loader = build_zero_shot_loader(args, device)
    run_analysis(model, loader, device, args)
    logging.info("Memory analysis complete: %s", args.output_dir)


if __name__ == "__main__":
    main()
