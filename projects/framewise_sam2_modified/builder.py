from pathlib import Path

import torch
import logging

from sam2.modeling.backbones.hieradet import Hiera
from sam2.modeling.backbones.image_encoder import FpnNeck, ImageEncoder
from sam2.modeling.memory_attention import (
    MemoryAttention,
    MemoryAttentionLayer,
)
from sam2.modeling.memory_encoder import (
    CXBlock,
    Fuser,
    MaskDownSampler,
    MemoryEncoder,
)
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.modeling.sam.transformer import RoPEAttention

from training.model.sam2_modified import SAM2Modified
from training.utils.sam2_modified_checkpoint import (
    DuplicateMaskDecoderWeights,
)

from training.model.adapter import (
    inject_image_encoder_adapters,
    inject_mask_decoder_adapters,
    iter_image_encoder_adapters,
    iter_mask_decoder_adapters,
)


def build_sam2_modified_tiny(
    checkpoint_path,
    device="cpu",
    mode="eval",
    image_size=1024,
    use_image_adapter=False,
    use_decoder_adapter=False,
    adapter_dim=64,
    adapter_dropout=0.1,
    adapter_init_scale=1e-3,
):
    """Build a SAM2.1 tiny model with independent left/right mask decoders."""


    # 0. 检查传入的参数
    if mode not in {"train", "eval"}:
        raise ValueError(f"mode must be 'train' or 'eval', but got {mode!r}")
    if image_size <= 0 or image_size % 16 != 0:
        raise ValueError("image_size must be a positive multiple of 16, "f"but got {image_size}")

    # 1. 处理参数
    device = torch.device(device)
    sam_feature_size = image_size // 16


    # 2. 创建图像编码器 ImageEncoder
    # Hiera 是真正从图片中提取视觉特征的主干网络
    image_backbone = Hiera(
        embed_dim=96,
        num_heads=1,
        stages=(1, 2, 7, 2),
        global_att_blocks=(5, 7, 9),
        window_pos_embed_bkg_spatial_size=(7, 7),
    )

    # 给图像特征添加位置信息。
    image_position_encoding = PositionEmbeddingSine(
        num_pos_feats=256,
        normalize=True,
        scale=None,
        temperature=10000,
    )

    # FPN负责整理Hiera输出的多尺度特征。
    image_neck = FpnNeck(
        position_encoding=image_position_encoding,
        d_model=256,
        backbone_channel_list=[768, 384, 192, 96],
        fpn_top_down_levels=[2, 3],
        fpn_interp_model="nearest",
    )

    # 把Hiera和FPN组合成完整的ImageEncoder。
    image_encoder = ImageEncoder(
        trunk=image_backbone,
        neck=image_neck,
        scalp=1,
    )


    # 3. 创建记忆注意力 MemoryAttention
    # 当前帧内部的注意力。
    memory_self_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=(sam_feature_size, sam_feature_size),
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
    )

    # 当前帧特征和历史memory之间的注意力。
    memory_cross_attention = RoPEAttention(
        rope_theta=10000.0,
        feat_sizes=(sam_feature_size, sam_feature_size),
        rope_k_repeat=True,
        embedding_dim=256,
        num_heads=1,
        downsample_rate=1,
        dropout=0.1,
        kv_in_dim=64,
    )

    # 组合一次self-attention、cross-attention和前馈网络。
    memory_attention_layer = MemoryAttentionLayer(
        activation="relu",
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        pos_enc_at_attn=False,
        pos_enc_at_cross_attn_keys=True,
        pos_enc_at_cross_attn_queries=False,
        self_attention=memory_self_attention,
        cross_attention=memory_cross_attention,
    )

    # 使用4层MemoryAttentionLayer。
    memory_attention = MemoryAttention(
        d_model=256,
        pos_enc_at_input=True,
        layer=memory_attention_layer,
        num_layers=4,
    )


    # 4. 创建记忆编码器 MemoryEncoder
    # Memory特征使用64维的位置编码。
    memory_position_encoding = PositionEmbeddingSine(
        num_pos_feats=64,
        normalize=True,
        scale=None,
        temperature=10000,
    )

    # 将高分辨率mask逐渐缩小到memory需要的尺寸。
    mask_downsampler = MaskDownSampler(
        kernel_size=3,
        stride=2,
        padding=1,
    )

    # 一层用于融合mask特征和图像特征的卷积块。
    memory_fuser_layer = CXBlock(
        dim=256,
        kernel_size=7,
        padding=3,
        layer_scale_init_value=1e-6,
        use_dwconv=True,
    )

    # 使用两层融合卷积块。
    memory_fuser = Fuser(
        layer=memory_fuser_layer,
        num_layers=2,
    )

    # 组合成完整的MemoryEncoder。
    memory_encoder = MemoryEncoder(
        out_dim=64,
        position_encoding=memory_position_encoding,
        mask_downsampler=mask_downsampler,
        fuser=memory_fuser,
    )


    # 5. 创建SAM2Modified
    model = SAM2Modified(
        # 前面手动创建的三个主要模块。
        image_encoder=image_encoder,
        memory_attention=memory_attention,
        memory_encoder=memory_encoder,

        # 最多保存7组mask memory。
        num_maskmem=7,

        # 输入图片尺寸。
        image_size=image_size,

        # mask进入MemoryEncoder前的数值变换。
        sigmoid_scale_for_mem_enc=20.0,
        sigmoid_bias_for_mem_enc=-10.0,

        # 如果输入直接是mask，允许直接把它作为输出。
        use_mask_input_as_output_without_sam=True,

        # 第一帧没有历史memory时，直接加入no-memory embedding。
        directly_add_no_mem_embed=True,

        # 没有目标时，给空间memory加入一个特殊embedding。
        no_obj_embed_spatial=True,

        # MaskDecoder使用高分辨率辅助特征。
        use_high_res_features_in_sam=True,

        # SAM支持输出多个候选mask。
        multimask_output_in_sam=True,

        # IoU预测限制在0到1之间。
        iou_prediction_use_sigmoid=True,

        # 在视频memory中使用object pointer。
        use_obj_ptrs_in_encoder=True,

        # 给object pointer加入时间位置编码。
        add_tpos_enc_to_obj_ptrs=True,
        proj_tpos_enc_in_obj_ptrs=True,
        use_signed_tpos_enc_to_obj_ptrs=True,

        # 推理时只使用当前帧之前的object pointer。
        only_obj_ptrs_in_the_past_for_eval=True,

        # 预测当前帧中是否存在目标。
        pred_obj_scores=True,
        pred_obj_scores_mlp=True,
        fixed_no_obj_ptr=True,

        # 视频跟踪相关的多mask设置。
        multimask_output_for_tracking=True,
        use_multimask_token_for_obj_ptr=True,
        multimask_min_pt_num=0,
        multimask_max_pt_num=1,

        # 使用MLP生成object pointer。
        use_mlp_for_obj_ptr_proj=True,

        # tiny模型不编译ImageEncoder。
        compile_image_encoder=False,
    )
    # 创建SAM2Modified时，内部已经自动创建：
    # model.left_mask_decoder
    # model.right_mask_decoder
    # 原来的model.sam_mask_decoder已经被删除。

    
    # 6. 加载checkpoint
    checkpoint_path = Path(checkpoint_path)

    # 检查checkpoint文件是否真实存在。
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # 从硬盘读取checkpoint，参数先保存在CPU中。
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # 官方SAM2 checkpoint的模型参数保存在"model"中。
    if "model" not in checkpoint:
        raise KeyError(
            "The checkpoint does not contain a 'model' state_dict."
        )

    official_state_dict = checkpoint["model"]

    # 官方checkpoint只有一个sam_mask_decoder。
    # 这里把它转换成left_mask_decoder和right_mask_decoder。
    converter = DuplicateMaskDecoderWeights()

    modified_state_dict = converter(state_dict=official_state_dict)

    # 把转换后的参数装入SAM2Modified。
    # strict=True要求所有参数名称都必须完全匹配。
    model.load_state_dict(modified_state_dict, strict=True)

    # 6.1 插入 image_encoder的adapter
    if use_image_adapter:
        adapter_count = inject_image_encoder_adapters(
            model,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
            adapter_init_scale=adapter_init_scale,
        )

        if adapter_count == 0:
            raise RuntimeError("没有向 Image Encoder 注入任何 Adapter")

    # 6.2 插入 mask_decoder的adapter
    if use_decoder_adapter:
        adapter_count = inject_mask_decoder_adapters(
            model,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
            adapter_init_scale=adapter_init_scale,
        )

        if adapter_count == 0:
            raise RuntimeError(
                "没有向 Mask Decoder 注入任何 Adapter"
            )


    # 7. 把模型移动到CPU或GPU
    model = model.to(device)

    # 8. 设置模型工作模式
    if mode == "train":
        model.train()
    else:
        model.eval()

    return model

def configure_finetune_stage(
    model: torch.nn.Module,
    use_decoder_adapter: bool = False,
) -> None:
    """
    冻结模型主体，训练左右 MaskDecoder。

    使用 Decoder Adapter 时：
    - 冻结原始 TwoWayTransformer
    - 训练 Transformer 内的 Adapter
    - MaskDecoder 其他部分保持训练
    """
    image_adapters = list(iter_image_encoder_adapters(model))
    decoder_adapters = list(iter_mask_decoder_adapters(model))

    model.requires_grad_(False)
    model.left_mask_decoder.requires_grad_(True)
    model.right_mask_decoder.requires_grad_(True)

    if use_decoder_adapter:
        if not decoder_adapters:
            raise RuntimeError("启用了 Decoder Adapter，但模型中没有找到 Adapter")

        model.left_mask_decoder.transformer.requires_grad_(False)
        model.right_mask_decoder.transformer.requires_grad_(False)

        for adapter in decoder_adapters:
            adapter.requires_grad_(True)

    for adapter in image_adapters:
        adapter.requires_grad_(True)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    adapter_params = sum(
        p.numel()
        for adapter in image_adapters + decoder_adapters
        for p in adapter.parameters()
        if p.requires_grad
    )
    other_params = trainable_params - adapter_params

    logging.info(
        "Finetune parameters | total=%s | trainable=%s (%.2f%%) | "
        "adapter=%s | other=%s",
        f"{total_params:,}",
        f"{trainable_params:,}",
        100.0 * trainable_params / total_params,
        f"{adapter_params:,}",
        f"{other_params:,}",
    )
