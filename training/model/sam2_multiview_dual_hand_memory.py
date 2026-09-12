import copy
from contextlib import nullcontext
from dataclasses import dataclass
from functools import partial
from typing import Literal

import torch

from .sam2_dual_hand_memory import SAM2DualHandMemory
from sam2.modeling.sam2_utils import get_next_point, sample_box_points


PromptMode = Literal["auto", "point", "mask"]


@dataclass(frozen=True)
class PromptRequest:
    """
    统一调整 Prompt 的入口。

    auto 模式：除 mode / start_frame_idx 外全部忽略，
    一切由模型配置（train/eval 的概率与帧数开关）决定。

    point / mask 模式：完全由本请求控制——
    - prompt_frame_indices: 哪些帧获得初始提示（成为条件帧）；
    - correction_frame_indices: 哪些帧追加纠错点，None 表示不纠错。

    值为 None 的字段回退到模型配置。
    """
    mode: PromptMode = "auto"
    start_frame_idx: int = 0

    # point/mask 模式：这些帧明确获得 prompt，并自动成为条件帧
    prompt_frame_indices: tuple[int, ...] = ()

    # point/mask 模式：这些帧明确追加纠错点；None 表示不纠错
    correction_frame_indices: tuple[int, ...] | None = None

    # None 回退到 self.num_correction_pt_per_frame
    num_correction_points_per_frame: int | None = None

    # None 回退到 self.add_all_frames_to_correct_as_cond
    add_correction_frames_as_cond: bool | None = None


@dataclass
class PromptPlan:
    batch_size: int
    num_views: int
    num_frames: int

    gt_masks_per_frame: dict[int, torch.Tensor]

    init_cond_frames: list[int]
    frames_not_in_init_cond: list[int]

    point_inputs_per_frame: dict[int, dict[str, torch.Tensor]]
    mask_inputs_per_frame: dict[int, torch.Tensor]

    frames_to_add_correction_pt: list[int]
    num_correction_points_per_frame: int
    add_correction_frames_as_cond: bool

    @property
    def num_images_per_frame(self) -> int:
        return self.batch_size * self.num_views

class SAM2MultiViewDualHandMemory(SAM2DualHandMemory):
    def __init__(
        self,
        *args,
        multiview_aggregator,
        multiview_distributor,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.left_multiview_aggregator = copy.deepcopy(multiview_aggregator)
        self.right_multiview_aggregator = copy.deepcopy(multiview_aggregator)

        self.left_multiview_distributor = copy.deepcopy(multiview_distributor)
        self.right_multiview_distributor = copy.deepcopy(multiview_distributor)

        del multiview_aggregator
        del multiview_distributor

    def forward(
        self,
        images,
        left_masks,
        right_masks,
        prompt_request,
        encoder_chunk_size=None,
    ):
        """
        images:      [B,V,T,3,H,W]
        left_masks:  [B,V,T,1,H,W]
        right_masks: [B,V,T,1,H,W]

        prompt_request 见 PromptRequest：
        auto 模式按模型配置随机采样，point/mask 模式完全由请求指定。

        encoder_chunk_size 控制单次送入 Image Encoder 的图像数。
        None 表示一次编码当前帧的 B*V 张图像。
        """
        if prompt_request.mode not in {"auto", "point", "mask"}:
            raise ValueError(f"不支持的 prompt mode: {prompt_request.mode}")

        assert images.dim() == left_masks.dim() == right_masks.dim() == 6
        assert images.shape[:3] == left_masks.shape[:3] == right_masks.shape[:3]

        # 1. Prompt
        prompt_plan = self.prepare_prompt_inputs(
            left_masks=left_masks,
            right_masks=right_masks,
            prompt_request=prompt_request,
        )

        # 2. 按帧、分块提取图像特征，并立即执行当前帧 Tracking。
        return self.forward_tracking(
            images,
            prompt_plan,
            encoder_chunk_size=encoder_chunk_size,
        )

    def _image_feature_modules(self):
        """返回 forward_image 中实际参与特征提取和投影的模块。"""

        return (
            self.image_encoder,
            self.left_mask_decoder.conv_s0,
            self.left_mask_decoder.conv_s1,
            self.right_mask_decoder.conv_s0,
            self.right_mask_decoder.conv_s1,
        )

    def _encode_image_chunk(self, images):
        """编码一小批图像，只保留当前帧 Tracking 会使用的特征。"""

        needs_grad = torch.is_grad_enabled() and (
            images.requires_grad
            or any(
                parameter.requires_grad
                for module in self._image_feature_modules()
                for parameter in module.parameters()
            )
        )
        context = nullcontext() if needs_grad else torch.no_grad()

        with context:
            backbone_out = self.forward_image(images)
            return {
                "main_feature": backbone_out["backbone_fpn"][-1],
                "main_pos_embed": backbone_out["vision_pos_enc"][-1],
                "left_high_res_features": backbone_out["left_high_res_features"],
                "right_high_res_features": backbone_out["right_high_res_features"],
            }

    def _encode_frame_views(self, frame_images, encoder_chunk_size=None):
        """分块编码当前帧的 B*V 张图像，并恢复 Mask Decoder 所需格式。"""

        batch_size, num_views = frame_images.shape[:2]
        flat_images = frame_images.flatten(0, 1)
        num_images = batch_size * num_views

        if encoder_chunk_size is None:
            encoder_chunk_size = num_images
        if encoder_chunk_size < 1:
            raise ValueError("encoder_chunk_size 必须大于 0")

        chunks = [
            self._encode_image_chunk(image_chunk)
            for image_chunk in flat_images.split(encoder_chunk_size, dim=0)
        ]

        main_feature_map = torch.cat(
            [chunk["main_feature"] for chunk in chunks],
            dim=0,
        )
        main_pos_map = torch.cat(
            [chunk["main_pos_embed"] for chunk in chunks],
            dim=0,
        )
        high_res_features = {
            hand: [
                torch.cat(
                    [chunk[f"{hand}_high_res_features"][level] for chunk in chunks],
                    dim=0,
                )
                for level in range(len(chunks[0][f"{hand}_high_res_features"]))
            ]
            for hand in ("left", "right")
        }

        feature_size = main_feature_map.shape[-2:]
        main_feature = main_feature_map.flatten(2).permute(2, 0, 1)
        main_pos_embed = main_pos_map.flatten(2).permute(2, 0, 1)
        return main_feature, main_pos_embed, feature_size, high_res_features
    
    def prepare_prompt_inputs(
        self,
        left_masks,
        right_masks,
        prompt_request,
    ):
        """
        该函数不执行跟踪，只按四段组织：
        1. 整理 GT；
        2. 决定使用 point/box 还是 mask prompt；
        3. 决定哪一帧提供 Prompt 和纠错；
        4. 根据 GT 和帧计划生成具体 Prompt。

        模式语义：
        - "auto"：按照模型的 train/eval 配置自动生成 Prompt 计划；
        - "point"：强制使用单点提示，Prompt 帧和纠错帧由请求指定；
        - "mask"：强制使用完整 GT mask，Prompt 帧和纠错帧由请求指定。

        特殊规则：
        - T=1 时强制使用 point prompt；
        - point/mask 模式不使用自动选择 Prompt 帧和纠错帧的配置；
        - box prompt 遇到空目标时，坐标置零、标签设为 -1。
        """
        B, V, num_frames = left_masks.shape[:3]
        start_frame_idx = prompt_request.start_frame_idx
        mode = prompt_request.mode

        # 1. 整理 GT
        # [B,V,T,1,H,W] -> [T,2BV,1,H,W]
        gt_masks = torch.cat(
            [
                left_masks.permute(2, 0, 1, 3, 4, 5).flatten(1, 2),
                right_masks.permute(2, 0, 1, 3, 4, 5).flatten(1, 2),
            ],
            dim=1,
        ).bool()
        gt_masks_per_frame = dict(enumerate(gt_masks))

        # 2. Prompt计划：决定使用 point/box 还是 mask prompt
        # 2.1 读 train/eval 配置
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

        # 2.2 mode -> 概率钳制：auto 保持配置值不变；
        # point 强制点、禁 box；mask 强制 mask。
        if mode == "point":
            prob_to_use_pt_input = 1.0
            prob_to_use_box_input = 0.0
        elif mode == "mask":
            prob_to_use_pt_input = 0.0

        # 2.3 T=1 时强制 point prompt
        if num_frames == 1:
            prob_to_use_pt_input = 1.0
            num_frames_to_correct = 1
            num_init_cond_frames = 1

        # 2.4 统一采样：True 用 point/box，False 用 mask
        use_pt_input = self.rng.random() < prob_to_use_pt_input


        # 3. Prompt 计划：决定哪一帧提供 Prompt 和纠错
        if mode == "auto":
            # 3.1 随机化数量
            if rand_init_cond_frames and num_init_cond_frames > 1:
                num_init_cond_frames = self.rng.integers(
                    1, num_init_cond_frames, endpoint=True
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

            # 3.2 条件帧：首帧必选，其余随机
            init_cond_frames = [start_frame_idx]
            if num_init_cond_frames > 1:
                init_cond_frames += self.rng.choice(
                    range(start_frame_idx + 1, num_frames),
                    num_init_cond_frames - 1,
                    replace=False,
                ).tolist()
        else:
            # point/mask：条件帧完全由请求指定，去重保序
            init_cond_frames = list(dict.fromkeys(
                prompt_request.prompt_frame_indices
            ))
            if not init_cond_frames:
                raise ValueError(f"{mode} 模式至少需要一个 Prompt 帧")
            assert all(
                start_frame_idx <= frame_idx < num_frames
                for frame_idx in init_cond_frames
            ), f"Prompt 帧下标越界: {init_cond_frames}"

        frames_not_in_init_cond = [
            frame_idx
            for frame_idx in range(start_frame_idx, num_frames)
            if frame_idx not in init_cond_frames
        ]

        # 3.3 纠错帧
        if mode == "auto":
            if not use_pt_input:
                frames_to_add_correction_pt = []
            elif num_frames_to_correct == num_init_cond_frames:
                frames_to_add_correction_pt = init_cond_frames
            else:
                extra_num = num_frames_to_correct - num_init_cond_frames
                frames_to_add_correction_pt = init_cond_frames + self.rng.choice(
                    frames_not_in_init_cond,
                    extra_num,
                    replace=False,
                ).tolist()
        else:
            # point/mask：纠错帧完全由请求指定，None 表示不纠错
            frames_to_add_correction_pt = list(dict.fromkeys(
                prompt_request.correction_frame_indices or ()
            ))

        assert all(
            start_frame_idx <= frame_idx < num_frames
            for frame_idx in frames_to_add_correction_pt
        ), f"纠错帧下标越界: {frames_to_add_correction_pt}"

        # 3.4 决定纠错执行方式
        if mode == "auto":
            num_correction_points_per_frame = self.num_correction_pt_per_frame
            add_correction_frames_as_cond = self.add_all_frames_to_correct_as_cond
            
        else:
            num_correction_points_per_frame = (
                self.num_correction_pt_per_frame
                if prompt_request.num_correction_points_per_frame is None
                else prompt_request.num_correction_points_per_frame
            )
            add_correction_frames_as_cond = (
                self.add_all_frames_to_correct_as_cond
                if prompt_request.add_correction_frames_as_cond is None
                else prompt_request.add_correction_frames_as_cond
            )

        # 4. 根据 GT, Prompt 计划给出提示
        point_inputs_per_frame = {}
        mask_inputs_per_frame = {}

        for frame_idx in init_cond_frames:
            curr_gt_masks = gt_masks_per_frame[frame_idx]

            if not use_pt_input:
                mask_inputs_per_frame[frame_idx] = curr_gt_masks
                continue

            use_box_input = self.rng.random() < prob_to_use_box_input

            if use_box_input:
                points, labels = sample_box_points(curr_gt_masks)

                # 空目标无法合法生成 box，坐标置零、标签设为 -1
                present = curr_gt_masks.flatten(1).any(dim=1)
                points[~present] = 0
                labels[~present] = -1
            else:
                points, labels = get_next_point(
                    gt_masks=curr_gt_masks,
                    pred_masks=None,
                    method="uniform" if self.training else self.pt_sampling_for_eval,
                )

            point_inputs_per_frame[frame_idx] = {
                "point_coords": points,
                "point_labels": labels,
            }



        return PromptPlan(
            batch_size=B,
            num_views=V,
            num_frames=num_frames,
            gt_masks_per_frame=gt_masks_per_frame,
            init_cond_frames=init_cond_frames,
            frames_not_in_init_cond=frames_not_in_init_cond,
            point_inputs_per_frame=point_inputs_per_frame,
            mask_inputs_per_frame=mask_inputs_per_frame,
            frames_to_add_correction_pt=frames_to_add_correction_pt,
            num_correction_points_per_frame=num_correction_points_per_frame,
            add_correction_frames_as_cond=add_correction_frames_as_cond,
        )
    
    def forward_tracking(
        self,
        images,
        prompt_plan,
        return_dict=False,
        encoder_chunk_size=None,
    ):
        """
        按照 PromptPlan 逐帧跟踪左右手，并分别维护两套 Memory bank。

        同一时刻的 B*V 张图像作为当前帧的图像 batch。
        """

        batch_size = prompt_plan.batch_size
        num_views = prompt_plan.num_views
        num_images_per_frame = prompt_plan.num_images_per_frame
        num_frames = prompt_plan.num_frames

        init_cond_frames = prompt_plan.init_cond_frames
        frames_to_add_correction_pt = prompt_plan.frames_to_add_correction_pt
        processing_order = init_cond_frames + prompt_plan.frames_not_in_init_cond

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

        # 1. 处理输入，按 PromptPlan 指定的顺序逐帧跟踪
        for frame_idx in processing_order:
            # 1. 当前帧只同时保留 B*V 份精简特征；Image Encoder 可按更小的
            # chunk 顺序执行，避免一次编码 T*B*V 张完整图像。
            (
                main_feature,
                main_pos_embed,
                feature_size,
                high_res_features_by_hand,
            ) = self._encode_frame_views(
                images[:, :, frame_idx],
                encoder_chunk_size=encoder_chunk_size,
            )

            # 2. 取当前帧的 Prompt 和 GT
            frame_gt_masks = prompt_plan.gt_masks_per_frame[frame_idx]
            frame_point_inputs = prompt_plan.point_inputs_per_frame.get(frame_idx)
            frame_mask_inputs = prompt_plan.mask_inputs_per_frame.get(frame_idx)
                # 判断当前帧是否会有 Prompt， 进而确定要保存到哪一类
            is_cond_frame = frame_idx in init_cond_frames

            # 3. 分别处理左右手
            # 每帧 Prompt/GT 为 [2BV,...]
            for hand, hand_slice in (
                ("left", slice(0, num_images_per_frame)),
                ("right", slice(num_images_per_frame, 2 * num_images_per_frame)),
            ):
                high_res_features = high_res_features_by_hand[hand]
                # 使用的是hand_slice, 从 [2BV, 1, H, W] 中取出当前 手 对应的 BV 个 Prompt 
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
                    batch_size=batch_size,num_views=num_views,
                    num_correction_points_per_frame=prompt_plan.num_correction_points_per_frame,
                )

                    # Append the output, depending on whether it's a conditioning frame
                add_as_cond_frame = is_cond_frame or (
                    prompt_plan.add_correction_frames_as_cond
                    and frame_idx in frames_to_add_correction_pt
                )
                if add_as_cond_frame:
                    output_dict[hand]["cond_frame_outputs"][frame_idx] = current_out
                else:
                    output_dict[hand]["non_cond_frame_outputs"][frame_idx] = current_out
                
        # 2. 整理输出，合并条件帧和非条件帧输出

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
        batch_size,
        num_views,
        num_correction_points_per_frame,
        track_in_reverse=False,
        run_mem_encoder=True,
        frames_to_add_correction_pt=None,
        gt_masks=None,
    ):
        """
        完成一只手在一帧上的预测、纠错和 Memory 写入。

        处理流程：
        1. 选择当前手的 Mask Decoder、Memory Attention 和 Memory Encoder。
        2. Memory Attention -> 多视角融合 -> 第一次预测
        3. 如果当前帧需要纠错，迭代采样纠错点并重新预测。
        4. 保存完整纠错历史和最终预测。
        5. 将最终预测编码为当前帧 Memory。

    设：
        H、W 为输入图像和高分辨率 mask 的尺寸；
        H4、W4 为低分辨率 mask 尺寸；
        H16、W16 为pix_feat特征和空间 Memory 的尺寸；
        M 为 Multi-view Aggregator 的共享 token 数；
        R 为当前帧的纠错次数，S = R + 1 为总预测轮数；
        K_s 为第 s 轮的候选 mask 数；
        P_s 为第 s 轮累计的提示点数。

    Returns:
        current_out，包含：

        多轮纠错历史：
        - multistep_pred_masks:
            Tensor[B*V, S, H4, W4]
        - multistep_pred_masks_high_res:
            Tensor[B*V, S, H, W]
        - multistep_pred_multimasks:
            长度为 S 的 list，第 s 项为
            Tensor[B*V, K_s, H4, W4]
        - multistep_pred_multimasks_high_res:
            长度为 S 的 list，第 s 项为
            Tensor[B*V, K_s, H, W]
        - multistep_pred_ious:
            长度为 S 的 list，第 s 项为 Tensor[B*V, K_s]
        - multistep_point_inputs:
            长度为 S 的 list，第 s 项为 None，或者：
            {
                "point_coords": Tensor[B*V, P_s, 2],
                "point_labels": Tensor[B*V, P_s],
            }
        - multistep_object_score_logits:
            长度为 S 的 list，第 s 项为 Tensor[B*V, 1]

        最终预测：
        - pred_masks:
            Tensor[B*V, 1, H4, W4]
        - pred_masks_high_res:
            Tensor[B*V, 1, H, W]

        Memory 状态：
        - obj_ptr:
            Tensor[B*V, C]
        - maskmem_features:
            Tensor[B*V, C_mem, H16, W16]；未运行 Memory Encoder 时为 None
        - maskmem_pos_enc:
            包含 Tensor[B*V, C_mem, H16, W16] 的 list；未运行 Memory Encoder 时为 None
        """

        # 准备纠错的帧
        if frames_to_add_correction_pt is None:
            frames_to_add_correction_pt = []

        # 1. 选择当前手对应的专属模块
        if hand == "left":
            mask_decoder = self.left_mask_decoder
            memory_attention = self.left_memory_attention
            memory_encoder = self.left_memory_encoder
            multiview_aggregator = self.left_multiview_aggregator
            multiview_distributor = self.left_multiview_distributor
        else:
            mask_decoder = self.right_mask_decoder
            memory_attention = self.right_memory_attention
            memory_encoder = self.right_memory_encoder
            multiview_aggregator = self.right_multiview_aggregator
            multiview_distributor = self.right_multiview_distributor

        # 2. Memory Attention -> 多视角融合 -> 第一次预测
        sam_outputs, pix_feat_with_mem_with_multiview = self._track_step(
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
            multiview_aggregator=multiview_aggregator,
            multiview_distributor=multiview_distributor,
            batch_size=batch_size,
            num_views=num_views,
            track_in_reverse=track_in_reverse,
        )

        (
            low_res_multimasks,   # [B*V,K,H4,W4]
            high_res_multimasks,  # [B*V,K,H,W]
            ious,                 # [B*V,K]
            low_res_masks,        # [B*V,1,H4,W4]
            high_res_masks,       # [B*V,1,H,W]
            obj_ptr,              # [B*V,C]
            object_score_logits,  # [B*V,1]
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
            and num_correction_points_per_frame > 0
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
            # 5. 重复 num_correction_points_per_frame 次。
            point_inputs, final_sam_outputs = self._iter_correct_pt_sampling(
                is_init_cond_frame=is_init_cond_frame,
                point_inputs=point_inputs,
                gt_masks=gt_masks,
                high_res_features=high_res_features,
                pix_feat_with_mem=pix_feat_with_mem_with_multiview,
                low_res_multimasks=low_res_multimasks,
                high_res_multimasks=high_res_multimasks,
                ious=ious,
                low_res_masks=low_res_masks,
                high_res_masks=high_res_masks,
                object_score_logits=object_score_logits,
                current_out=current_out,
                sam_head=sam_head,
                num_correction_points_per_frame=num_correction_points_per_frame,
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
        multiview_aggregator,
        multiview_distributor,
        batch_size,
        num_views,
        track_in_reverse=False,
    ):
        """
        与父类的差异：在 Memory Attention 之后、Mask Decoder 之前，
        插入多视角 Aggregator / Distributor 融合。其余行为不变。
        """

        # mask 直接作为输出：不跑 Memory Attention 和 Mask Decoder，也不融合多视角。
        if mask_inputs is not None and self.use_mask_input_as_output_without_sam:
            pix_feat = main_feature.permute(1, 2, 0)
            pix_feat = pix_feat.view(-1, self.hidden_dim, *feature_size)

            sam_outputs = self._use_mask_as_output(
                backbone_features=pix_feat,
                high_res_features=high_res_features,
                mask_inputs=mask_inputs,
                mask_decoder=mask_decoder,
            )
            return sam_outputs, pix_feat

        # 1. Memory Attention。条件帧走 no-mem 捷径，但输出形状相同，融合统一生效。
        # pix_feat_with_mem: [B*V, C, H16, W16]
        pix_feat_with_mem = self._prepare_memory_conditioned_features(
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
        

        # 2. 多视角融合：Aggregator 聚合成共享 token，Distributor 分发回各视角。
        # pix_feat_with_mem_with_multiview: [B*V, C, H16, W16]，纠错轮复用该结果
        pix_feat_with_mem_with_multiview = self._fuse_multiview_features(
            pix_feat=pix_feat_with_mem,               
            pos_embed=main_pos_embed, 
            multiview_aggregator=multiview_aggregator,
            multiview_distributor=multiview_distributor,
            batch_size=batch_size,
            num_views=num_views,
            feature_size=feature_size,
        )


        # 3. Mask Decoder：与父类一致。融合结果同时被后续纠错轮复用
        sam_outputs = self._forward_one_sam_head(
            mask_decoder=mask_decoder,
            prompt_encoder=self.sam_prompt_encoder,
            backbone_features=pix_feat_with_mem_with_multiview,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            high_res_features=high_res_features,
            multimask_output=self._use_multimask(is_init_cond_frame, point_inputs),
        )

        return sam_outputs, pix_feat_with_mem_with_multiview


    def _fuse_multiview_features(
        self,
        pix_feat,               # [B*V, C, H, W]，memory attention 的输出
        pos_embed,              # [H*W, B*V, C]，backbone 空间位置编码
        multiview_aggregator,
        multiview_distributor,
        batch_size,
        num_views,
        feature_size,           # (H, W)
    ):
        """打包 BVNC、聚合共享 token、分发回各视角、解包回解码器输入格式。"""
        B, V = batch_size, num_views
        H, W = feature_size
        N = H * W
        C = self.hidden_dim

        # [B*V, C, H, W] -> [B, V, N, C]
        # b-major（forward 的 permute(2,0,1).flatten 决定），split dim0 即按视角分组
        view_features = pix_feat.reshape(B, V, C, N).permute(0, 1, 3, 2)
        view_pos = pos_embed.permute(1, 0, 2).reshape(B, V, N, C)

        # 视角间暂时只用空间位置编码，不引入相机内外参。
        shared_tokens = multiview_aggregator(
            multiview_features=view_features,
            multiview_pos=view_pos,
        )  # [B, M, C]

        distributed = multiview_distributor(
            view_features=view_features,
            view_pos=view_pos,
            shared_tokens=shared_tokens,
        )  # [B, V, N, C]

        # [B, V, N, C] -> [B*V, C, H, W]，恢复为 Mask Decoder 的输入形状
        return distributed.reshape(B * V, N, C).permute(0, 2, 1).reshape(B * V, C, H, W)
