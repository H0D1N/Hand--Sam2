import copy
import torch

from .sam2_modified import SAM2Modified
from sam2.modeling.sam2_utils import get_next_point, sample_box_points


class SAM2DualHandMemory(SAM2Modified):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.left_memory_attention = copy.deepcopy(self.memory_attention)
        self.right_memory_attention = copy.deepcopy(self.memory_attention)

        self.left_memory_encoder = copy.deepcopy(self.memory_encoder)
        self.right_memory_encoder = copy.deepcopy(self.memory_encoder)

        del self.memory_attention
        del self.memory_encoder

    def forward_sequence(
        self,
        images,
        left_masks,
        right_masks,
        prompt_mode,
    ):
        """
        images:      [B, T, 3, H, W]
        left_masks:  [B, T, 1, H, W]
        right_masks: [B, T, 1, H, W]
        prompt_mode: "point" 或 "mask"
        """
        if prompt_mode not in {"point", "mask"}:
            raise ValueError(f"不支持的 prompt_mode: {prompt_mode}")

        assert images.dim() == 5
        assert left_masks.dim() == 5
        assert right_masks.dim() == 5
        assert images.shape[:2] == left_masks.shape[:2] == right_masks.shape[:2]

        batch_size, num_frames = images.shape[:2]

        # [B,T,C,H,W] -> [T*B,C,H,W]
        flat_images = images.transpose(0, 1).flatten(0, 1)

        # 1. 提取整个序列的图像特征
        backbone_out = self.forward_image(flat_images)

        backbone_out["batch_size"] = batch_size
        backbone_out["num_frames"] = num_frames

        # 2. 根据左右手 GT 决定条件帧并生成 Prompt
        backbone_out = self.prepare_prompt_inputs(
            backbone_out=backbone_out,
            left_masks=left_masks,
            right_masks=right_masks,
            prompt_mode=prompt_mode,
        )

        # 3. 使用 Prompt 和左右 Memory 运行序列跟踪
        return self.forward_tracking(backbone_out)

    def prepare_prompt_inputs(
        self,
        backbone_out,
        left_masks,
        right_masks,
        prompt_mode,
        start_frame_idx=0,
    ):
        """ 依靠 GT 选择 prompt"""
        """
        Input:
        backbone_out[]:
        | `vision_features` | `[N, 256, 48, 48]` | 最低分辨率主特征，送入 Memory 和 Mask Decoder |
        | `backbone_fpn[0]` | `[N, 256, 192, 192]` | 1/4 尺度图像特征 |
        | `backbone_fpn[1]` | `[N, 256, 96, 96]` | 1/8 尺度图像特征 |
        | `backbone_fpn[2]` | `[N, 256, 48, 48]` | 1/16 尺度图像特征 |
        | `vision_pos_enc[0]` | `[N, 256, 192, 192]` | 1/4 特征的位置编码 |
        | `vision_pos_enc[1]` | `[N, 256, 96, 96]` | 1/8 特征的位置编码 |
        | `vision_pos_enc[2]` | `[N, 256, 48, 48]` | 1/16 特征的位置编码 |
        | `left_high_res_features[0]` | `[N, 32, 192, 192]` | 左手 Decoder 的高分辨率辅助特征 |
        | `left_high_res_features[1]` | `[N, 64, 96, 96]` | 左手 Decoder 的高分辨率辅助特征 |
        | `right_high_res_features[0]` | `[N, 32, 192, 192]` | 右手 Decoder 的高分辨率辅助特征 |
        | `right_high_res_features[1]` | `[N, 64, 96, 96]` | 右手 Decoder 的高分辨率辅助特征 |
        # 序列基本信息
        backbone_out["batch_size"] = batch_size int(B)
        backbone_out["num_frames"] = num_frames int(T)

        Output:
        上面的基础上再加上
        
        # Prompt 生成计划
        backbone_out["use_pt_input"]  # bool 决定使用 点提示或者框提示 / GT masks 提示
        backbone_out["init_cond_frames"]          # list[int] 这些帧给prompt
        backbone_out["frames_not_in_init_cond"]   # list[int] 这些帧依靠memory
        backbone_out["frames_to_add_correction_pt"]  # list[int] 这些帧允许补充点

        # 整理好的GT 和 Prompt 内容 
        "gt_masks_per_frame": {
            t: Tensor[2B, 1, 768, 768],
        },
        "point_inputs_per_frame": {
            frame_idx: {
                "point_coords": Tensor[2B, P, 2],
                "point_labels": Tensor[2B, P],
            }
            ...
        }
        "mask_inputs_per_frame": {
            frame_idx: Tensor[2B, 1, H, W]
            ...
        }
        """


        """
        left_masks:  [B, T, 1, H, W]
        right_masks: [B, T, 1, H, W]
        """

        # [B,T,1,H,W] + [B,T,1,H,W]
        # -> gt_masks:[T,2B,1,H,W]
        # gt_masks[t] : [2B, 1, H, W]
        # 前 B 个是左手，后 B 个是右手
        # 之后的完全照搬 sam2 设计
        gt_masks = torch.cat(
            [left_masks, right_masks],
            dim=0,
        ).transpose(0, 1).bool()

        # 1. 整理 GT
        gt_masks_per_frame = {
            frame_idx: masks
            for frame_idx, masks in enumerate(gt_masks)
        }
        backbone_out["gt_masks_per_frame"] = gt_masks_per_frame

        num_frames = backbone_out["num_frames"]

        # 2. Prompt计划：决定使用 point/box 还是 mask prompt
        if self.training:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_train
            prob_to_use_box_input = self.prob_to_use_box_input_for_train
            num_frames_to_correct = self.num_frames_to_correct_for_train
            rand_frames_to_correct = self.rand_frames_to_correct_for_train
            num_init_cond_frames = self.num_init_cond_frames_for_train
            rand_init_cond_frames = self.rand_init_cond_frames_for_train
        else:
            prob_to_use_pt_input = self.prob_to_use_pt_input_for_eval
            prob_to_use_box_input = self.prob_to_use_box_input_for_eval
            num_frames_to_correct = self.num_frames_to_correct_for_eval
            rand_frames_to_correct = self.rand_frames_to_correct_for_eval
            num_init_cond_frames = self.num_init_cond_frames_for_eval
            rand_init_cond_frames = self.rand_init_cond_frames_for_eval

        if prompt_mode == "point":
            prob_to_use_pt_input = 1.0
            prob_to_use_box_input = 0.0
        elif prompt_mode == "mask":
            prob_to_use_pt_input = 0.0

        if num_frames == 1:
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1

        # 2. Prompt 计划：决定哪一帧提供 Prompt 和纠错
        assert num_init_cond_frames >= 1
        use_pt_input = self.rng.random() < prob_to_use_pt_input

        if rand_init_cond_frames and num_init_cond_frames > 1:
            num_init_cond_frames = self.rng.integers(
                1,
                num_init_cond_frames,
                endpoint=True,
            )

        if (
            use_pt_input
            and rand_frames_to_correct
            and num_frames_to_correct > num_init_cond_frames
        ):
            num_frames_to_correct = self.rng.integers(
                num_init_cond_frames,
                num_frames_to_correct,
                endpoint=True,
            )

        backbone_out["use_pt_input"] = use_pt_input

        if num_init_cond_frames == 1:
            init_cond_frames = [start_frame_idx]
        else:
            init_cond_frames = [start_frame_idx] + self.rng.choice(
                range(start_frame_idx + 1, num_frames),
                num_init_cond_frames - 1,
                replace=False,
            ).tolist()

        frames_not_in_init_cond = [
            frame_idx
            for frame_idx in range(start_frame_idx, num_frames)
            if frame_idx not in init_cond_frames
        ]

        backbone_out["init_cond_frames"] = init_cond_frames
        backbone_out["frames_not_in_init_cond"] = frames_not_in_init_cond
        backbone_out["mask_inputs_per_frame"] = {}
        backbone_out["point_inputs_per_frame"] = {}

        # 3. 根据 GT, Prompt计划 给出提示
        for frame_idx in init_cond_frames:
            curr_frame_gt_masks = gt_masks_per_frame[frame_idx]

            if not use_pt_input:
                backbone_out["mask_inputs_per_frame"][frame_idx] = (
                    curr_frame_gt_masks
                )
                continue

            use_box_input = self.rng.random() < prob_to_use_box_input

            if use_box_input:
                points, labels = sample_box_points(curr_frame_gt_masks)
            else:
                points, labels = get_next_point(
                    gt_masks=curr_frame_gt_masks,
                    pred_masks=None,
                    method=(
                        "uniform"
                        if self.training
                        else self.pt_sampling_for_eval
                    ),
                )

            present = curr_frame_gt_masks.flatten(1).any(dim=1)
            points[~present] = 0
            labels[~present] = -1

            backbone_out["point_inputs_per_frame"][frame_idx] = {
                "point_coords": points,
                "point_labels": labels,
            }

        if not use_pt_input:
            frames_to_add_correction_pt = []
        elif num_frames_to_correct == num_init_cond_frames:
            frames_to_add_correction_pt = init_cond_frames
        else:
            extra_num = num_frames_to_correct - num_init_cond_frames
            frames_to_add_correction_pt = (
                init_cond_frames
                + self.rng.choice(
                    frames_not_in_init_cond,
                    extra_num,
                    replace=False,
                ).tolist()
            )

        backbone_out["frames_to_add_correction_pt"] = (
            frames_to_add_correction_pt
        )

        return backbone_out

    def forward_tracking(self, backbone_out, return_dict=False):
        """按帧跟踪左右手，并分别维护两套 Memory bank。"""

        # 0. 准备输入
        # backbone_fpn & vision_pos_enc are in (HW)NC format
        _, vision_feats, vision_pos_embeds, feat_sizes = self._prepare_backbone_features(backbone_out)

        batch_size = backbone_out["batch_size"]
        num_frames = backbone_out["num_frames"]
        init_cond_frames = backbone_out["init_cond_frames"]
        processing_order = init_cond_frames + backbone_out["frames_not_in_init_cond"]

        output_dict = {
            "left": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
            "right": {
                "cond_frame_outputs": {},
                "non_cond_frame_outputs": {},
            },
        }

        # 1. 处理输入
        for frame_idx in processing_order:
            # 1. 从整段视频特征中[H*W, T*B, C]，取出当前 frame_idx 帧对应的 B 张图片特征
            image_slice = slice(frame_idx * batch_size, (frame_idx + 1) * batch_size)

                # 只取 最小的特征
                # 高分辨率特征只在 mask_decoder 中以 backbone_out[f"{hand}_high_res_features"] 用到
            main_feature = vision_feats[-1][:, image_slice]
            main_pos_embed = vision_pos_embeds[-1][:, image_slice]
            feature_size = feat_sizes[-1]

            # 2. 取当前帧的 Prompt
            frame_point_inputs = backbone_out["point_inputs_per_frame"].get(frame_idx)
            frame_mask_inputs = backbone_out["mask_inputs_per_frame"].get(frame_idx)
                # 判断当前帧是否会有 Prompt， 进而确定要保存到哪一类
            is_cond_frame = frame_idx in init_cond_frames

            # 3. 分别处理左右手
            for hand, hand_slice in (
                ("left", slice(0, batch_size)),
                ("right", slice(batch_size, 2 * batch_size)),
            ):
                # 使用的是image_slice, 从整段视频特征中[T*B, C, H, W]，取出当前 frame_idx 帧对应的 B 张图片特征
                high_res_features = [
                    feature[image_slice]
                    for feature in backbone_out[f"{hand}_high_res_features"]
                ]
                # 使用的是hand_slice, 从 [2B, 1, H, W] 中取出当前 手 对应的 B 个 Prompt 
                point_inputs = None if frame_point_inputs is None else {
                    key: value[hand_slice] for key, value in frame_point_inputs.items()
                }
                mask_inputs = None if frame_mask_inputs is None else frame_mask_inputs[hand_slice]

                current_out = self.track_step(
                    hand=hand, frame_idx=frame_idx, is_init_cond_frame=is_cond_frame,
                    main_feature=main_feature, main_pos_embed=main_pos_embed,
                    feature_size=feature_size, high_res_features=high_res_features,
                    point_inputs=point_inputs, mask_inputs=mask_inputs,
                    output_dict=output_dict[hand], num_frames=num_frames,
                    track_in_reverse=False,
                )

                    # Append the output, depending on whether it's a conditioning frame
                if is_cond_frame:
                    output_dict[hand]["cond_frame_outputs"][frame_idx] = current_out
                else:
                    output_dict[hand]["non_cond_frame_outputs"][frame_idx] = current_out
                
        # 2. 整理输出

            # return_dict=True 时直接返回完整的 Memory bank
        if return_dict:
            return output_dict

            # 把 cond 和 non-cond 输出合并，并恢复成 0...T-1 的顺序
        left_outputs = {}
        left_outputs.update(output_dict["left"]["cond_frame_outputs"])
        left_outputs.update(output_dict["left"]["non_cond_frame_outputs"])
        left_outputs = [left_outputs[t] for t in range(num_frames)]

        right_outputs = {}
        right_outputs.update(output_dict["right"]["cond_frame_outputs"])
        right_outputs.update(output_dict["right"]["non_cond_frame_outputs"])
        right_outputs = [right_outputs[t] for t in range(num_frames)]

            # 删除 loss 不需要的 obj_ptr，组合成左右手输出
        all_frame_outputs = [
            {
                "left": {k: v for k, v in left_out.items() if k != "obj_ptr"},
                "right": {k: v for k, v in right_out.items() if k != "obj_ptr"},
            }
            for left_out, right_out in zip(left_outputs, right_outputs)
        ]

        return all_frame_outputs

    def track_step(
        self,
        hand,
        frame_idx,
        is_init_cond_frame,
        main_feature,
        main_pos_embed,
        feature_size,
        high_res_features,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
    ):
        """
        Output: current_out: dict

        # 给 loss/可视化：
        high_res_multimasks [B, M, 768, 768]
        ious [B, M]
        object_score_logits [B, 1]

        # 直接或处理后给 Memory Encoder：
        obj_ptr [B, 256]
        high_res_masks [B, 1, 768, 768]
        maskmem_features [B, 64, 48(main_feat:H), 48(main_feat:W)]
        maskmem_pos_enc [Tensor[B, 64, 48, 48]]
        """


        # 1. 选择当前手对应的三个模块
        if hand == "left":
            mask_decoder = self.left_mask_decoder
            memory_attention = self.left_memory_attention
            memory_encoder = self.left_memory_encoder
        else:
            mask_decoder = self.right_mask_decoder
            memory_attention = self.right_memory_attention
            memory_encoder = self.right_memory_encoder

        # 2. 读取历史 Memory，完成第一次预测
        sam_outputs = self._track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            main_feature=main_feature,
            main_pos_embed=main_pos_embed,
            feature_size=feature_size,
            high_res_features=high_res_features,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=num_frames,
            mask_decoder=mask_decoder,
            memory_attention=memory_attention,
            track_in_reverse=track_in_reverse,
        )

        (
            low_res_multimasks,   # [B,M,192,192]
            high_res_multimasks,  # [B,M,768,768]
            ious,                 # [B,M]
            low_res_masks,        # [B,1,192,192]
            high_res_masks,       # [B,1,768,768]
            obj_ptr,              # [B,256]
            object_score_logits,  # [B,1]
        ) = sam_outputs

        # 3. 保存现有 loss 和可视化需要的输出
        current_out = {
            "high_res_multimasks": high_res_multimasks,
            "ious": ious,
            "high_res_masks": high_res_masks,
            "obj_ptr": obj_ptr,
            "object_score_logits": object_score_logits,
        }

        # 4. 把当前预测编码成新 Memory
        # current_out 中加入
        # current_out["maskmem_features"] 
        # current_out["maskmem_pos_enc"] 
        self._encode_memory_in_output(
            current_vision_feats=[main_feature],
            feat_sizes=[feature_size],
            point_inputs=point_inputs,
            run_mem_encoder=run_mem_encoder,
            high_res_masks=high_res_masks,
            object_score_logits=object_score_logits,
            current_out=current_out,
            memory_encoder=memory_encoder,
        )

        return current_out

    def _track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        main_feature,
        main_pos_embed,
        feature_size,
        high_res_features,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        mask_decoder,
        memory_attention,
        track_in_reverse=False,
    ):
        """
        
        """

        # 将完整的GT mask 作为输出，可能用于 条件帧直接提供完整 GT mask
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = main_feature.permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feature_size)

            return self._use_mask_as_output(
                backbone_features=pix_feat,
                high_res_features=high_res_features,
                mask_inputs=mask_inputs,
                mask_decoder=mask_decoder,
            )

        # 融合 attention
        pix_feat = self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=[main_feature],
            current_vision_pos_embeds=[main_pos_embed],
            feat_sizes=[feature_size],
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
            memory_attention=memory_attention,
        )

        return self._forward_one_sam_head(
            mask_decoder=mask_decoder,
            prompt_encoder=self.sam_prompt_encoder,
            backbone_features=pix_feat,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=self._use_multimask(is_init_cond_frame, point_inputs),
        )
