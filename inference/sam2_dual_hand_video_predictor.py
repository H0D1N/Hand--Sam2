"""双手视频逐帧传播，仅保留后续计算会使用的 memory。"""

import torch

from training.model.sam2_dual_hand_memory import SAM2DualHandMemory


class SAM2DualHandVideoPredictor(SAM2DualHandMemory):
    def init_state(self, video_loader):
        if self.training:
            raise ValueError("视频推理需要先调用 model.eval()")
        if video_loader.batch_size != 1 or len(video_loader) == 0:
            raise ValueError("video_loader 必须非空且 batch_size=1")
        iterator = iter(video_loader)
        return {
            "iterator": iterator, "num_frames": len(video_loader), "frame_idx": 0,
            "first_frame": next(iterator), "prompts": {},
            "output_dict": {hand: {"cond_frame_outputs": {}, "non_cond_frame_outputs": {}} for hand in ("left", "right")},
        }

    def add_mask(self, state, hand, mask):
        """登记 Dataset 预处理后的首帧 mask，形状为 [1, 1, H, W]。"""
        if hand not in ("left", "right") or state["frame_idx"] != 0:
            raise ValueError("只能在传播前为 left/right 添加首帧提示")
        if mask.shape != (1, 1, self.image_size, self.image_size):
            raise ValueError("mask 必须为 [1, 1, image_size, image_size]")
        state["prompts"][hand] = {"mask_inputs": mask, "point_inputs": None}

    def add_points(self, state, hand, point_inputs):
        """登记首帧点提示，不在传播过程中追加纠错。"""
        if hand not in ("left", "right") or state["frame_idx"] != 0:
            raise ValueError("只能在传播前为 left/right 添加首帧提示")
        state["prompts"][hand] = {"mask_inputs": None, "point_inputs": point_inputs}

    @torch.inference_mode()
    def propagate_in_video(self, state):
        if state["frame_idx"] == 0 and set(state["prompts"]) != {"left", "right"}:
            raise ValueError("传播前需要登记左右手的首帧提示")
        device = next(self.parameters()).device
        while state["frame_idx"] < state["num_frames"]:
            frame_idx = state["frame_idx"]
            batch = state.pop("first_frame") if frame_idx == 0 else next(state["iterator"])
            frame_info = {key: batch[key][0] for key in ("image_path", "original_size", "dataset_root")}
            images = batch["image"].to(device, non_blocking=True)
            backbone_out = self.forward_image(images)
            backbone_out, vision_feats, vision_pos, feat_sizes = self._prepare_backbone_features(backbone_out)
            outputs = {}
            for hand in ("left", "right"):
                prompt = state["prompts"].get(hand, {})
                mask_inputs, point_inputs = prompt.get("mask_inputs"), prompt.get("point_inputs")
                mask_inputs = None if mask_inputs is None else mask_inputs.to(device, non_blocking=True)
                point_inputs = None if point_inputs is None else {key: value.to(device, non_blocking=True) for key, value in point_inputs.items()}
                current_out = self.track_step(
                    hand=hand, frame_idx=frame_idx, is_init_cond_frame=frame_idx == 0,
                    main_feature=vision_feats[-1], main_pos_embed=vision_pos[-1], feature_size=feat_sizes[-1],
                    high_res_features=backbone_out[f"{hand}_high_res_features"],
                    point_inputs=point_inputs, mask_inputs=mask_inputs,
                    output_dict=state["output_dict"][hand], num_frames=state["num_frames"],
                    track_in_reverse=False, run_mem_encoder=True, gt_masks=None, frames_to_add_correction_pt=[],
                )
                memory = state["output_dict"][hand]["cond_frame_outputs" if frame_idx == 0 else "non_cond_frame_outputs"]
                memory[frame_idx] = {key: current_out[key] for key in ("maskmem_features", "maskmem_pos_enc", "obj_ptr")}
                outputs[hand] = {"pred_masks_high_res": current_out["pred_masks_high_res"]}
            state["prompts"].clear()
            state["frame_idx"] = frame_idx + 1
            self._prune_memory(state)
            del batch, images, backbone_out, vision_feats, vision_pos, current_out, prompt, mask_inputs, point_inputs
            yield frame_idx, outputs, frame_info
            del outputs

    def _prune_memory(self, state):
        # 以下一帧为基准：空间特征和 pointer 的历史窗口不同。
        next_frame = state["frame_idx"]
        spatial_window = 0 if self.num_maskmem <= 1 else 1 + (self.num_maskmem - 2) * self.memory_temporal_stride_for_eval
        pointer_window = min(state["num_frames"], self.max_obj_ptrs_in_encoder) - 1 if self.num_maskmem > 0 and self.use_obj_ptrs_in_encoder else 0
        for hand in ("left", "right"):
            memory = state["output_dict"][hand]["non_cond_frame_outputs"]
            for frame_idx in list(memory):
                if frame_idx < next_frame - spatial_window:
                    memory[frame_idx].pop("maskmem_features", None)
                    memory[frame_idx].pop("maskmem_pos_enc", None)
                if frame_idx < next_frame - pointer_window:
                    memory[frame_idx].pop("obj_ptr", None)
                if not memory[frame_idx]:
                    del memory[frame_idx]
