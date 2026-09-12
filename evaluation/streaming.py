"""逐帧因果执行长视频提示策略，并且只把最终结果写入 Memory。"""

from __future__ import annotations

from dataclasses import dataclass
import logging

import torch
import torch.nn.functional as F

from projects.framewise_sam2_modified.dataset import build_center_point_prompt
from projects.framewise_sam2_modified.losses import iou_target_from_logits


@dataclass(frozen=True)
class EvaluationPolicy:
    strategy: str
    prompt_mode: str = "mask"
    prompt_interval: int = 80
    iou_threshold: float = 0.5
    correction_points: int = 1
    max_condition_frames: int = 4

    def __post_init__(self):
        if self.strategy not in {"baseline", "fixed", "adaptive"}:
            raise ValueError(f"未知策略: {self.strategy}")
        if self.prompt_mode not in {"mask", "point"}:
            raise ValueError(f"未知 prompt_mode: {self.prompt_mode}")
        if self.prompt_interval < 1:
            raise ValueError("prompt_interval 必须大于 0")
        if not 0.0 <= self.iou_threshold <= 1.0:
            raise ValueError("iou_threshold 必须位于 [0, 1]")
        if self.correction_points < 1:
            raise ValueError("correction_points 必须大于 0")
        if self.max_condition_frames < 1:
            raise ValueError("max_condition_frames 必须大于 0")

    @property
    def configuration(self) -> str:
        if self.strategy == "fixed":
            return f"fixed_interval_{self.prompt_interval}"
        if self.strategy == "adaptive":
            threshold = format(self.iou_threshold, "g").replace(".", "p")
            return f"adaptive_iou_{threshold}"
        return "baseline"


MEMORY_KEYS = ("maskmem_features", "maskmem_pos_enc", "obj_ptr")


def _empty_output_dict() -> dict:
    return {
        hand: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}
        for hand in ("left", "right")
    }


def _original_iou_per_view(
    logits: torch.Tensor,
    original_masks: list[torch.Tensor],
) -> list[float]:
    values = []
    for view_index, target in enumerate(original_masks):
        target = target.unsqueeze(0).to(logits.device, non_blocking=True)
        resized_logits = F.interpolate(
            logits[view_index:view_index + 1],
            size=target.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        values.append(float(iou_target_from_logits(resized_logits, target).item()))
    return values


def _compact_memory_output(current_out: dict) -> dict:
    return {key: current_out[key] for key in MEMORY_KEYS}


def _prune_memory_bank(model, bank: dict, next_frame: int, num_frames: int) -> None:
    spatial_window = (
        0
        if model.num_maskmem <= 1
        else 1 + (model.num_maskmem - 2) * model.memory_temporal_stride_for_eval
    )
    pointer_window = (
        min(num_frames, model.max_obj_ptrs_in_encoder) - 1
        if model.num_maskmem > 0 and model.use_obj_ptrs_in_encoder
        else 0
    )
    for frame_index in list(bank):
        if frame_index < next_frame - spatial_window:
            bank[frame_index].pop("maskmem_features", None)
            bank[frame_index].pop("maskmem_pos_enc", None)
        if frame_index < next_frame - pointer_window:
            bank[frame_index].pop("obj_ptr", None)
        if not bank[frame_index]:
            del bank[frame_index]


class LongVideoEvaluator:
    def __init__(
        self,
        model,
        model_type: str,
        device,
        policy: EvaluationPolicy,
        amp=False,
        log_interval=50,
    ):
        if model_type not in {"memory", "multiview"}:
            raise ValueError(f"未知 model_type: {model_type}")
        self.model = model
        self.model_type = model_type
        self.device = torch.device(device)
        self.policy = policy
        self.amp = bool(amp and self.device.type == "cuda")
        self.log_interval = max(int(log_interval), 1)
        self.model.max_cond_frames_in_attn = policy.max_condition_frames
        if model_type == "memory":
            self.model.num_correction_pt_per_frame = policy.correction_points

    def _is_prompt_frame(self, frame_index: int) -> bool:
        return frame_index == 0 or (
            self.policy.strategy == "fixed"
            and frame_index % self.policy.prompt_interval == 0
        )

    def _track_step(
        self,
        *,
        hand,
        frame_index,
        is_conditioning,
        main_feature,
        main_pos_embed,
        feature_size,
        high_res_features,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        gt_masks,
        correct,
        run_mem_encoder,
        num_views,
    ):
        kwargs = {}
        if self.model_type == "multiview":
            kwargs.update(
                batch_size=1,
                num_views=num_views,
                num_correction_points_per_frame=self.policy.correction_points,
            )
        return self.model.track_step(
            hand=hand,
            frame_idx=frame_index,
            is_init_cond_frame=is_conditioning,
            main_feature=main_feature,
            main_pos_embed=main_pos_embed,
            feature_size=feature_size,
            high_res_features=high_res_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=False,
            run_mem_encoder=run_mem_encoder,
            frames_to_add_correction_pt=[frame_index] if correct else [],
            # SAM2 的误差点采样使用按位逻辑，要求 GT 为 bool。
            # 训练路径会在 prepare_prompt_inputs 中转换；流式评估绕过了
            # 该函数，因此需要在 track_step 边界保持相同约定。
            gt_masks=gt_masks.bool(),
            **kwargs,
        )

    @torch.inference_mode()
    def evaluate_sequence(self, dataset, sequence) -> list[dict]:
        num_views = len(sequence.view_names)
        if self.model_type == "memory" and num_views != 1:
            raise ValueError("memory baseline 每次只能评估一个视角")

        output_dict = _empty_output_dict()
        rows = []

        for frame_index in range(sequence.num_frames):
            frame = dataset.load_frame(sequence, frame_index)
            images = frame["image"].to(self.device, non_blocking=True)
            gt_masks_by_hand = {
                hand: frame[f"{hand}_mask"].to(self.device, non_blocking=True)
                for hand in ("left", "right")
            }
            is_prompt_frame = self._is_prompt_frame(frame_index)

            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.amp,
            ):
                backbone_out = self.model.forward_image(images)
                backbone_out, vision_feats, vision_pos, feat_sizes = (
                    self.model._prepare_backbone_features(backbone_out)
                )
                main_feature = vision_feats[-1]
                main_pos_embed = vision_pos[-1]
                feature_size = feat_sizes[-1]

                for hand in ("left", "right"):
                    gt_masks = gt_masks_by_hand[hand]
                    original_masks = frame[f"original_{hand}_mask"]
                    point_inputs = None
                    mask_inputs = None
                    if is_prompt_frame:
                        if self.policy.prompt_mode == "mask":
                            mask_inputs = gt_masks
                        else:
                            point_inputs = build_center_point_prompt(gt_masks)

                    corrected = False
                    if self.policy.strategy == "adaptive" and not is_prompt_frame:
                        trial_out = self._track_step(
                            hand=hand,
                            frame_index=frame_index,
                            is_conditioning=False,
                            main_feature=main_feature,
                            main_pos_embed=main_pos_embed,
                            feature_size=feature_size,
                            high_res_features=backbone_out[f"{hand}_high_res_features"],
                            point_inputs=None,
                            mask_inputs=None,
                            output_dict=output_dict[hand],
                            num_frames=sequence.num_frames,
                            gt_masks=gt_masks,
                            correct=False,
                            run_mem_encoder=False,
                            num_views=num_views,
                        )
                        before_ious = _original_iou_per_view(
                            trial_out["pred_masks_high_res"], original_masks
                        )
                        corrected = min(before_ious) < self.policy.iou_threshold
                        # 试预测只用于决定是否纠错；在正式提交当前帧前释放其
                        # decoder 输出，避免 adaptive 路径同时保留两套结果。
                        del trial_out
                        trial_out = None
                    else:
                        trial_out = None
                        before_ious = None

                    current_out = self._track_step(
                        hand=hand,
                        frame_index=frame_index,
                        is_conditioning=is_prompt_frame,
                        main_feature=main_feature,
                        main_pos_embed=main_pos_embed,
                        feature_size=feature_size,
                        high_res_features=backbone_out[f"{hand}_high_res_features"],
                        point_inputs=point_inputs,
                        mask_inputs=mask_inputs,
                        output_dict=output_dict[hand],
                        num_frames=sequence.num_frames,
                        gt_masks=gt_masks,
                        correct=corrected,
                        run_mem_encoder=True,
                        num_views=num_views,
                    )
                    after_ious = _original_iou_per_view(
                        current_out["pred_masks_high_res"], original_masks
                    )
                    if before_ious is None:
                        before_ious = after_ious

                    memory_kind = (
                        "cond_frame_outputs"
                        if is_prompt_frame or corrected
                        else "non_cond_frame_outputs"
                    )
                    output_dict[hand][memory_kind][frame_index] = (
                        _compact_memory_output(current_out)
                    )

                    if is_prompt_frame:
                        prompt_type = self.policy.prompt_mode
                    elif corrected:
                        prompt_type = "correction"
                    else:
                        prompt_type = "none"

                    for view_index, view_name in enumerate(sequence.view_names):
                        rows.append({
                            "model": self.model_type,
                            "strategy": self.policy.strategy,
                            "configuration": self.policy.configuration,
                            "prompt_mode": self.policy.prompt_mode,
                            "prompt_interval": (
                                self.policy.prompt_interval
                                if self.policy.strategy == "fixed" else None
                            ),
                            "iou_threshold": (
                                self.policy.iou_threshold
                                if self.policy.strategy == "adaptive" else None
                            ),
                            "correction_points": (
                                self.policy.correction_points
                                if self.policy.strategy == "adaptive" else None
                            ),
                            "dataset": sequence.dataset_name,
                            "sequence_id": sequence.sequence_id,
                            "evaluation_id": sequence.evaluation_id,
                            "segment_index": sequence.segment_index,
                            "view": view_name,
                            "frame_index": frame_index,
                            "frame_number": sequence.frame_numbers[frame_index],
                            "hand": hand,
                            "prompt_type": prompt_type,
                            "is_conditioning": is_prompt_frame or corrected,
                            "corrected": corrected,
                            "iou_before": before_ious[view_index],
                            "iou_after": after_ious[view_index],
                            "gt_present": bool(original_masks[view_index].any().item()),
                            "image_path": frame["image_path"][view_index],
                        })

                    del trial_out, current_out

            for hand in ("left", "right"):
                cond_bank = output_dict[hand]["cond_frame_outputs"]
                while len(cond_bank) > self.policy.max_condition_frames:
                    del cond_bank[min(cond_bank)]
                _prune_memory_bank(
                    self.model,
                    output_dict[hand]["non_cond_frame_outputs"],
                    next_frame=frame_index + 1,
                    num_frames=sequence.num_frames,
                )

            del frame, images, gt_masks_by_hand, backbone_out, vision_feats, vision_pos

            if (
                (frame_index + 1) % self.log_interval == 0
                or frame_index + 1 == sequence.num_frames
            ):
                logging.info(
                    "%s | %s | %d/%d frames",
                    self.policy.configuration,
                    sequence.evaluation_id,
                    frame_index + 1,
                    sequence.num_frames,
                )

        return rows
