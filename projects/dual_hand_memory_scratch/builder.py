"""构建 SAM 分割组件预训练、完整 Memory 子系统随机初始化的模型。"""

import logging
from pathlib import Path

import torch

from projects.dual_hand_memory.builder import _configure_prompt_sampling, configure_memory_training
from projects.framewise_sam2_modified.builder import create_sam2_modified_tiny, inject_sam2_modified_adapters
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory
from training.utils.sam2_modified_checkpoint import DuplicateMaskDecoderWeights


PRETRAINED_STATE_PREFIXES = (
    "image_encoder.",
    "sam_prompt_encoder.",
    "left_mask_decoder.",
    "right_mask_decoder.",
)
# 散落在 SAM2Base 顶层、不在左右 Memory 模块内的共享参数。
EXTRA_MEMORY_PARAMETER_NAMES = (
    "maskmem_tpos_enc",
    "no_mem_embed",
    "no_mem_pos_enc",
    "no_obj_ptr",
    "no_obj_embed_spatial",
)


def _is_pretrained_state(name: str) -> bool:
    return name.startswith(PRETRAINED_STATE_PREFIXES)


def build_sam2_dual_hand_memory_scratch_tiny(
    sam_checkpoint: str | Path,
    device: str | torch.device = "cpu",
    mode: str = "eval",
    image_size: int = 768,
    use_image_adapter: bool = False,
    use_decoder_adapter: bool = False,
    adapter_dim: int = 64,
    adapter_dropout: float = 0.1,
    adapter_init_scale: float = 1e-3,
    num_init_cond_frames_for_train: int = 2,
    num_frames_to_correct_for_train: int = 2,
    add_all_frames_to_correct_as_cond: bool = True,
    num_correction_pt_per_frame: int = 7,
) -> SAM2DualHandMemory:
    """
    构建完整 Memory 随机初始化的双手 SAM2 模型。

    模型先按默认方式创建，因此所有 Memory 参数都有各自模块原生的随机
    初始化；随后只加载 Image Encoder、Prompt Encoder 和左右 Mask Decoder
    的 SAM 权重。官方 checkpoint 中的 Memory 权重不会进入模型。

    Adapter 在预训练权重加载完成后注入，继续使用 Adapter 自己的初始化。
    提示采样策略、设备迁移和 train/eval 模式沿用原双手 Memory Builder。
    """

    if mode not in {"train", "eval"}:
        raise ValueError(f"Invalid mode: {mode}")

    # 1. 构造模型。Memory 保留各模块自身的默认初始化。
    model = create_sam2_modified_tiny(
        image_size=image_size,
        model_cls=SAM2DualHandMemory,
    )

    # 2. 只加载 Image Encoder、Prompt Encoder 和左右 Mask Decoder。
    checkpoint = torch.load(sam_checkpoint, map_location="cpu", weights_only=True)
    state_dict = checkpoint["model"]
    model = _initialize_from_sam_checkpoint(model=model, state_dict=state_dict)

    # 3. Adapter 沿用自身的专用初始化。
    inject_sam2_modified_adapters(
        model=model,
        use_image_adapter=use_image_adapter,
        use_decoder_adapter=use_decoder_adapter,
        adapter_dim=adapter_dim,
        adapter_dropout=adapter_dropout,
        adapter_init_scale=adapter_init_scale,
    )

    # 4. 复用原训练的提示策略和设备、模式配置。
    _configure_prompt_sampling(
        model=model,
        num_init_cond_frames_for_train=num_init_cond_frames_for_train,
        num_frames_to_correct_for_train=num_frames_to_correct_for_train,
        add_all_frames_to_correct_as_cond=add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=num_correction_pt_per_frame,
    )

    model = model.to(device)
    model.train() if mode == "train" else model.eval()
    return model


def _initialize_from_sam_checkpoint(model, state_dict):
    """
    加载与单帧分割直接相关的 SAM 权重，跳过完整 Memory 子系统。

    官方 SAM 只有一个 Mask Decoder，因此先复制成左右两个 Decoder，再通过
    白名单过滤掉 Memory Encoder、Memory Attention、object pointer 投影、
    Memory 位置编码以及 no-memory/no-object 状态。这里使用 strict=False
    是因为这些被跳过的随机 Memory state 会按设计出现在 missing keys 中。
    """

    state_dict = DuplicateMaskDecoderWeights()(state_dict)
    state_dict = {name: value for name, value in state_dict.items() if _is_pretrained_state(name)}

    _, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if unexpected_keys:
        raise RuntimeError(f"预训练权重与模型不匹配: unexpected={unexpected_keys}")

    return model


def configure_memory_scratch_training(
    model: torch.nn.Module,
    finetune_mode: str,
) -> None:
    """
    沿用原 finetune 模式，并补齐完整 Memory 数据流中的可训练参数。

    ``configure_memory_training`` 已经负责最基本的四个左右手模块：

    - ``left/right_memory_encoder``：把当前图像特征与预测 mask 编码为空间
      Memory feature；
    - ``left/right_memory_attention``：读取历史空间 Memory 和 object pointer，
      将其融合到当前帧视觉特征中。

    因此下面不重复列出这四个模块，只补充原配置没有开启的共享 Memory 接口：

    - ``mask_downsample``：条件帧直接输入 GT mask 时，将 mask 转成 Decoder
      可用的输入，以便生成该帧的 object pointer；
    - ``obj_ptr_proj``：把 Mask Decoder token 投影成写入 Memory bank 的
      object pointer；
    - ``obj_ptr_tpos_proj``：把 object pointer 的时间位置编码投影到 Memory
      特征维度；
    - ``maskmem_tpos_enc``：标记空间 Memory 来自哪个历史时间位置；
    - ``no_mem_embed`` / ``no_mem_pos_enc``：表示当前没有可读取的历史 Memory；
    - ``no_obj_ptr``：表示当前帧中目标不存在时的 object pointer；
    - ``no_obj_embed_spatial``：表示目标不存在时的空间 Memory feature。

    Adapter 和 Mask Decoder 是否训练仍由原来的 ``finetune_mode`` 决定。
    """

    configure_memory_training(model, finetune_mode)

    model.mask_downsample.requires_grad_(True)
    model.obj_ptr_proj.requires_grad_(True)
    model.obj_ptr_tpos_proj.requires_grad_(True)

    for name in EXTRA_MEMORY_PARAMETER_NAMES:
        getattr(model, name).requires_grad_(True)

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    logging.info(
        "Scratch Memory parameters | total=%s | trainable=%s (%.2f%%)",
        f"{total_params:,}",
        f"{trainable_params:,}",
        100.0 * trainable_params / total_params,
    )
