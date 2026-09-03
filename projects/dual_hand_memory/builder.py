from pathlib import Path

import torch
import logging

from projects.framewise_sam2_modified.builder import (
    create_sam2_modified_tiny,
    inject_sam2_modified_adapters,
    configure_finetune_stage,
)
from training.model.adapter import iter_mask_decoder_adapters
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory
from training.utils.sam2_dual_hand_memory_checkpoint import (
    DuplicateMemoryWeights,
)
from training.utils.sam2_modified_checkpoint import (
    DuplicateMaskDecoderWeights,
)

def build_sam2_dual_hand_memory_tiny(
    sam_checkpoint=None,
    framewise_checkpoint=None,
    device="cpu",
    mode="eval",
    image_size=768,
    use_image_adapter=False,
    use_decoder_adapter=False,
    adapter_dim=64,
    adapter_dropout=0.1,
    adapter_init_scale=1e-3,
    num_init_cond_frames_for_train=2,
    num_frames_to_correct_for_train=2,
    add_all_frames_to_correct_as_cond=True,
    num_correction_pt_per_frame=7,
):
    if (sam_checkpoint is None) == (framewise_checkpoint is None):
        raise ValueError("Exactly one of sam_checkpoint and framewise_checkpoint is required")

    if mode not in {"train", "eval"}:
        raise ValueError(f"Invalid mode: {mode}")

    using_sam_checkpoint = sam_checkpoint is not None

    if using_sam_checkpoint:
        # 准备参数
        checkpoint = torch.load(
            sam_checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        state_dict = checkpoint["model"]

        # 创建模型
        model = create_sam2_modified_tiny(
            image_size=image_size,
            model_cls=SAM2DualHandMemory,
        )

        # 初始化模型参数
        model = _initialize_from_sam_checkpoint(
            model=model,
            state_dict=state_dict,
            use_image_adapter=use_image_adapter,
            use_decoder_adapter=use_decoder_adapter,
            adapter_dim=adapter_dim,
            adapter_dropout=adapter_dropout,
            adapter_init_scale=adapter_init_scale,
        )
    else:
        checkpoint = torch.load(
            framewise_checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        checkpoint_args = checkpoint["args"]
        image_size = checkpoint_args["image_size"]
        state_dict = checkpoint["model_state"]

        model = create_sam2_modified_tiny(
            image_size=image_size,
            model_cls=SAM2DualHandMemory,
        )

        model = _initialize_from_framewise_checkpoint(
            model=model,
            state_dict=state_dict,
            checkpoint_args=checkpoint_args,
        )

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


def load_sam2_dual_hand_memory_tiny(model_checkpoint, device="cpu"):
    """按训练参数重建模型并加载完整的 Dual-hand Memory checkpoint。"""
    model_checkpoint = Path(model_checkpoint)
    if not model_checkpoint.is_file():
        raise FileNotFoundError(f"Memory checkpoint not found: {model_checkpoint}")

    checkpoint = torch.load(model_checkpoint, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]
    model = build_sam2_dual_hand_memory_tiny(
        sam_checkpoint=checkpoint_args["sam_checkpoint"],
        framewise_checkpoint=checkpoint_args["framewise_checkpoint"],
        device=device,
        mode="eval",
        image_size=checkpoint_args["image_size"],
        use_image_adapter=checkpoint_args["use_image_adapter"],
        use_decoder_adapter=checkpoint_args["use_decoder_adapter"],
        adapter_dim=checkpoint_args["adapter_dim"],
        adapter_dropout=checkpoint_args["adapter_dropout"],
        adapter_init_scale=checkpoint_args["adapter_init_scale"],
        num_init_cond_frames_for_train=checkpoint_args["num_init_cond_frames_for_train"],
        num_frames_to_correct_for_train=checkpoint_args["num_frames_to_correct_for_train"],
        add_all_frames_to_correct_as_cond=checkpoint_args["add_all_frames_to_correct_as_cond"],
        num_correction_pt_per_frame=checkpoint_args["num_correction_pt_per_frame"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    logging.info(
        "Loaded Dual-hand Memory checkpoint | %s | epoch=%s",
        model_checkpoint,
        checkpoint.get("epoch", "unknown"),
    )
    return model.eval()


def _configure_prompt_sampling(
    model,
    num_init_cond_frames_for_train,
    num_frames_to_correct_for_train,
    add_all_frames_to_correct_as_cond,
    num_correction_pt_per_frame,
):
    """配置双手 Memory baseline 的提示与纠错策略。"""


    # 训练：SAM2 原版提示分布。
    model.prob_to_use_pt_input_for_train = 0.5
    model.prob_to_use_box_input_for_train = 0.5
    model.prob_to_sample_from_gt_for_train = 0.1

    model.num_init_cond_frames_for_train = num_init_cond_frames_for_train
    model.rand_init_cond_frames_for_train = True

    model.num_frames_to_correct_for_train = num_frames_to_correct_for_train
    model.rand_frames_to_correct_for_train = True

    model.add_all_frames_to_correct_as_cond = add_all_frames_to_correct_as_cond
    model.num_correction_pt_per_frame = num_correction_pt_per_frame

    # 验证：第 0 帧提供 GT mask，后续帧纯 Memory tracking。
    model.prob_to_use_pt_input_for_eval = 0.0
    model.prob_to_use_box_input_for_eval = 0.0
    model.num_init_cond_frames_for_eval = 1
    model.rand_init_cond_frames_for_eval = False
    model.num_frames_to_correct_for_eval = 1
    model.rand_frames_to_correct_for_eval = False
    model.pt_sampling_for_eval = "center"

def _initialize_from_sam_checkpoint(
    model,
    state_dict,
    use_image_adapter,
    use_decoder_adapter,
    adapter_dim,
    adapter_dropout,
    adapter_init_scale,
):


    state_dict = DuplicateMaskDecoderWeights()(state_dict)
    state_dict = DuplicateMemoryWeights()(state_dict)

    model.load_state_dict(state_dict, strict=True)

    inject_sam2_modified_adapters(
        model=model,
        use_image_adapter=use_image_adapter,
        use_decoder_adapter=use_decoder_adapter,
        adapter_dim=adapter_dim,
        adapter_dropout=adapter_dropout,
        adapter_init_scale=adapter_init_scale,
    )

    return model

def _initialize_from_framewise_checkpoint(
    model,
    state_dict,
    checkpoint_args,
):

    print("当前使用预训练模型初始化，传入的Adapter超参数无效")

    inject_sam2_modified_adapters(
        model=model,
        use_image_adapter=checkpoint_args["use_image_adapter"],
        use_decoder_adapter=checkpoint_args["use_decoder_adapter"],
        adapter_dim=checkpoint_args["adapter_dim"],
        adapter_dropout=checkpoint_args["adapter_dropout"],
        adapter_init_scale=checkpoint_args["adapter_init_scale"],
    )

    state_dict = DuplicateMemoryWeights()(state_dict)
    model.load_state_dict(state_dict, strict=True)

    return model

def configure_memory_training(
    model: torch.nn.Module,
    finetune_mode: str,
) -> None:
    """冻结已有帧级模型，只训练左右手各自的 Memory 模块。"""

    if finetune_mode == "memory-only":
        model.requires_grad_(False)

    elif finetune_mode == "decoder-memory":
        has_decoder_adapter = any(True for _ in iter_mask_decoder_adapters(model))

        configure_finetune_stage(
            model=model,
            use_decoder_adapter=has_decoder_adapter,
        )

    else:
        raise ValueError(f"Invalid finetune mode: {finetune_mode}")

    model.left_memory_attention.requires_grad_(True)
    model.right_memory_attention.requires_grad_(True)
    model.left_memory_encoder.requires_grad_(True)
    model.right_memory_encoder.requires_grad_(True)

    trainable_names = [
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]

    if not trainable_names:
        raise RuntimeError("没有找到可训练的 Memory 参数")

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    logging.info(
        "Memory parameters | total=%s | trainable=%s (%.2f%%)",
        f"{total_params:,}",
        f"{trainable_params:,}",
        100.0 * trainable_params / total_params,
    )
