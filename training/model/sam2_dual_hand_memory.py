import copy
import torch
from functools import partial

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

    def forward(
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
        prompt_mode: "auto"、"point" 或 "mask"
        """
        if prompt_mode not in {"auto", "point", "mask"}:
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
        """
        根据左右手 GT 和提示模式生成序列训练所需的 Prompt 计划。

        形状约定：
            B: batch size；
            T: 序列帧数；
            N = T * B: 展平时间和 batch 后的图像数量；
            (H, W): 输入图像和 GT mask 的空间尺寸；
            (H4, W4) = (H // 4, W // 4)；
            (H8, W8) = (H // 8, W // 8)；
            (H16, W16) = (H // 16, W // 16)。

        该函数不执行跟踪，只负责：
        1. 将左右手 GT 按帧整理；
        2. 选择初始条件帧；
        3. 为条件帧生成 point、box 或 mask prompt；
        4. 选择后续允许模拟纠错的帧。

        Args:
            backbone_out:
                `forward_image()` 的输出，主要包含：

                - vision_features:
                  Tensor[N, 256, H16, W16]，送入 Memory Attention 和 Mask Decoder 的最低分辨率主特征；

                - backbone_fpn:
                  长度为 3 的多尺度特征：
                  1. Tensor[N, 256, H4, W4]；
                  2. Tensor[N, 256, H8, W8]；
                  3. Tensor[N, 256, H16, W16]；

                - vision_pos_enc:
                  与 `backbone_fpn` 三个尺度对应的位置编码：
                  1. Tensor[N, 256, H4, W4]；
                  2. Tensor[N, 256, H8, W8]；
                  3. Tensor[N, 256, H16, W16]；

                - left_high_res_features:
                  左手 Mask Decoder 使用的高分辨率辅助特征：
                  1. Tensor[N, 32, H4, W4]；
                  2. Tensor[N, 64, H8, W8]；

                - right_high_res_features:
                  右手 Mask Decoder 使用的高分辨率辅助特征：
                  1. Tensor[N, 32, H4, W4]；
                  2. Tensor[N, 64, H8, W8]；

                - batch_size: B；
                - num_frames: T。

                本函数不会修改上述视觉特征，只会向字典中增加
                GT、Prompt 和帧选择相关字段。

            left_masks:
                左手 GT，Tensor[B, T, 1, H, W]。

            right_masks:
                右手 GT，Tensor[B, T, 1, H, W]。

            prompt_mode:
                提示模式：
                - "auto": 根据模型的 train/eval 概率选择
                  mask、box 或 point；
                - "point": 强制使用单点提示，不采样 box；
                - "mask": 强制使用完整 GT mask。

            start_frame_idx:
                开始跟踪的帧下标，该帧始终是第一个条件帧。

        Returns:
            增加以下字段后的 `backbone_out`：

            gt_masks_per_frame:
                dict[int, Tensor[2B, 1, H, W]]。
                每帧前 B 个目标是左手，后 B 个目标是右手。

            use_pt_input:
                是否使用 point/box prompt。False 表示使用 mask prompt。

            init_cond_frames:
                list[int]，获得初始人工提示的条件帧。

            frames_not_in_init_cond:
                list[int]，没有初始提示、需要依靠 Memory 跟踪的帧。

            point_inputs_per_frame:
                dict[int, dict]，使用 point/box prompt 时保存：
                - point_coords: Tensor[2B, P, 2]；
                - point_labels: Tensor[2B, P]。

                P=1 表示单点，P=2 表示 box。目标不存在时，
                对应坐标置零、标签设为 -1。

            mask_inputs_per_frame:
                dict[int, Tensor[2B, 1, H, W]]。
                使用 mask prompt 时保存条件帧的完整 GT mask。

            frames_to_add_correction_pt:
                list[int]，允许在 `track_step()` 中根据预测误差
                动态添加纠错点的帧。该函数这里只生成纠错计划，
                不直接采样纠错点。

        Notes:
            当 T=1 时，沿用 SAM2 的静态图像训练逻辑，强制使用
            point prompt，并将唯一一帧同时作为条件帧和纠错帧。
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
        frames_to_add_correction_pt = backbone_out["frames_to_add_correction_pt"]
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

            # 2. 取当前帧的 Prompt 和 GT
            frame_gt_masks = backbone_out["gt_masks_per_frame"].get(frame_idx)
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
                gt_masks = frame_gt_masks[hand_slice]

                current_out = self.track_step(
                    hand=hand, frame_idx=frame_idx, is_init_cond_frame=is_cond_frame,
                    main_feature=main_feature, main_pos_embed=main_pos_embed,
                    feature_size=feature_size, high_res_features=high_res_features,
                    point_inputs=point_inputs, mask_inputs=mask_inputs,
                    output_dict=output_dict[hand], num_frames=num_frames,
                    track_in_reverse=False,
                    gt_masks=gt_masks,
                    frames_to_add_correction_pt=frames_to_add_correction_pt,
                )

                    # Append the output, depending on whether it's a conditioning frame
                add_as_cond_frame = is_cond_frame or (
                    self.add_all_frames_to_correct_as_cond
                    and frame_idx in frames_to_add_correction_pt
                )
                if add_as_cond_frame:
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
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        """
        完成一只手在一帧上的预测、纠错和 Memory 写入。

        处理流程：
        1. 选择当前手的 Mask Decoder、Memory Attention 和 Memory Encoder。
        2. 融合历史 Memory，完成第一次预测。
        3. 如果当前帧需要纠错，迭代采样纠错点并重新预测。
        4. 保存完整纠错历史和最终预测。
        5. 将最终预测编码为当前帧 Memory。

        设：
            K 为纠错次数，S = K + 1 为总预测轮数；
            M_s 为第 s 轮候选 mask 数；
            P_s 为第 s 轮累计提示点数。

        Returns:
            current_out，包含：

            多轮纠错历史：
            - multistep_pred_masks:
            Tensor[B, S, h, w]
            - multistep_pred_masks_high_res:
                Tensor[B, S, H, W]
            - multistep_pred_multimasks:
                长度为 S 的 list，第 s 项为 Tensor[B, M_s, h, w]
            - multistep_pred_multimasks_high_res:
                长度为 S 的 list，第 s 项为 Tensor[B, M_s, H, W]
            - multistep_pred_ious:
                长度为 S 的 list，第 s 项为 Tensor[B, M_s]
            - multistep_point_inputs:
                长度为 S 的 list，第 s 项为 None， 或者：
                    {
                        "point_coords": Tensor[B, P_s, 2],
                        "point_labels": Tensor[B, P_s],
                    }
            - multistep_object_score_logits:
                长度为 S 的 list，第 s 项为 Tensor[B, 1]

            最终预测：
            - pred_masks:
                Tensor[B, 1, h, w]
            - pred_masks_high_res:
                Tensor[B, 1, H, W]

            Memory 状态：
            - obj_ptr:
                Tensor[B, hidden_dim]
            - maskmem_features:
                Tensor[B, memory_dim, Hm, Wm]， 未运行 Memory Encoder 时为 None
            - maskmem_pos_enc:
                包含一个 Tensor[B, memory_dim, Hm, Wm] 的 list， 未运行 Memory Encoder 时为 None
        """

        # 准备纠错的帧
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []

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
        sam_outputs, pix_feat_with_mem = self._track_step(
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

        # 3. 初始化交互预测历史。
        # 没有纠错时这里只有第一次预测；
        # 有纠错时，_iter_correct_pt_sampling 会更新为完整历史。
        current_out = {
            "multistep_pred_masks": low_res_masks,
            "multistep_pred_masks_high_res": high_res_masks,
            "multistep_pred_multimasks": [low_res_multimasks],
            "multistep_pred_multimasks_high_res": [high_res_multimasks],
            "multistep_pred_ious": [ious],
            "multistep_point_inputs": [point_inputs],
            "multistep_object_score_logits": [object_score_logits],
        }

        # 4. 在指定帧上模拟多轮用户纠错。
        if (
            frame_idx in frames_to_add_correction_pt
            and self.num_correction_pt_per_frame > 0
        ):

            # 提前固定_forward_one_sam_head 的 mask_decoder, prompt_encoder参数
            sam_head = partial(
                self._forward_one_sam_head,
                mask_decoder=mask_decoder,
                prompt_encoder=self.sam_prompt_encoder,
            )

            # 原版 SAM2 纠错：
            # 1. 根据当前预测和 GT 的误差采样新点；
            # 2. 累积 point_inputs；
            # 3. 把上一轮 low_res_masks 作为 mask prompt；
            # 4. 再运行一次当前手的 Decoder；
            # 5. 重复 num_correction_pt_per_frame 次。
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame=is_init_cond_frame,
                point_inputs=point_inputs,
                gt_masks=gt_masks,
                high_res_features=high_res_features,
                pix_feat_with_mem=pix_feat_with_mem,
                low_res_multimasks=low_res_multimasks,
                high_res_multimasks=high_res_multimasks,
                ious=ious,
                low_res_masks=low_res_masks,
                high_res_masks=high_res_masks,
                object_score_logits=object_score_logits,
                current_out=current_out,
                sam_head=sam_head,
            )

            # final_sam_outputs 现在是最后一次纠错的结果。
            (
                low_res_multimasks,
                high_res_multimasks,
                ious,
                low_res_masks,
                high_res_masks,
                obj_ptr,
                object_score_logits,
            ) = final_sam_outputs

        # 5. 保存最终分割结果和后续帧需要的 object pointer。
        current_out["pred_masks"] = low_res_masks
        current_out["pred_masks_high_res"] = high_res_masks
        current_out["obj_ptr"] = obj_ptr

        # 6. 把当前预测编码成新 Memory
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
        完成当前手在当前帧上的第一次预测。


        当提供 mask prompt 且启用直接输出时，将 mask prompt 转为
        SAM 输出；否则先融合历史 Memory，再运行当前手的 Mask Decoder。

        Returns:
            sam_outputs:
                当前帧第一次 SAM head 预测的 7 元组。
            pix_feat:
                当前帧的像素特征，形状为 [B, hidden_dim, h16, w16]；
                后续纠错轮次会复用该特征。
        """

        # 将完整的GT mask 作为输出，可能用于 条件帧直接提供完整 GT mask
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = main_feature.permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feature_size)

            sam_outputs = self._use_mask_as_output(
                backbone_features=pix_feat,
                high_res_features=high_res_features,
                mask_inputs=mask_inputs,
                mask_decoder=mask_decoder,
            )

        else:
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

            sam_outputs = self._forward_one_sam_head(
                mask_decoder=mask_decoder,
                prompt_encoder=self.sam_prompt_encoder,
                backbone_features=pix_feat,
                point_inputs=point_inputs,
                mask_inputs=mask_inputs,
                high_res_features=high_res_features,
                multimask_output=self._use_multimask(is_init_cond_frame, point_inputs),
            )

        return sam_outputs, pix_feat
