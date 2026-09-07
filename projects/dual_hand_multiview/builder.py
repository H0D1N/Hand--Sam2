"""构建多视角双手 Memory 模型。"""

from functools import partial

import torch
import logging

from projects.dual_hand_memory.builder import _configure_prompt_sampling, configure_memory_training
from projects.framewise_sam2_modified.builder import create_sam2_modified_tiny, inject_sam2_modified_adapters
from sam2.modeling.sam.transformer import Attention
from training.model.multiview_aggregator import MultiViewAggregationLayer, MultiViewFeatureAggregator
from training.model.multiview_distributor import MultiViewDistributionLayer, MultiViewFeatureDistributor
from training.model.sam2_multiview_dual_hand_memory import SAM2MultiViewDualHandMemory



from training.utils.sam2_dual_hand_memory_checkpoint import DuplicateMemoryWeights
from training.utils.sam2_modified_checkpoint import DuplicateMaskDecoderWeights


ADAPTER_KEYS = (
    "use_image_adapter",
    "use_decoder_adapter",
    "adapter_dim",
    "adapter_dropout",
    "adapter_init_scale",
)

MULTIVIEW_PREFIXES = (
    "left_multiview_aggregator.",
    "right_multiview_aggregator.",
    "left_multiview_distributor.",
    "right_multiview_distributor.",
)


def _attention():
    return Attention(embedding_dim=256, num_heads=1, downsample_rate=1, dropout=0.1)


def build_multiview_modules(num_latents, num_aggregator_layers, num_distributor_layers):
    aggregation_layer = MultiViewAggregationLayer(
        d_model=256,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        self_attention=_attention(),
        cross_attention=_attention(),
    )
    distribution_layer = MultiViewDistributionLayer(
        d_model=256,
        dropout=0.1,
        cross_attention=_attention(),
    )

    aggregator = MultiViewFeatureAggregator(
        d_model=256,
        layer=aggregation_layer,
        num_layers=num_aggregator_layers,
        num_latents=num_latents,
    )
    distributor = MultiViewFeatureDistributor(
        d_model=256,
        layer=distribution_layer,
        num_layers=num_distributor_layers,
    )
    return aggregator, distributor


def _create_model(image_size, num_latents, num_aggregator_layers, num_distributor_layers):
    aggregator, distributor = build_multiview_modules(
        num_latents=num_latents,
        num_aggregator_layers=num_aggregator_layers,
        num_distributor_layers=num_distributor_layers,
    )
    model_cls = partial(
        SAM2MultiViewDualHandMemory,
        multiview_aggregator=aggregator,
        multiview_distributor=distributor,
    )
    return create_sam2_modified_tiny(image_size=image_size, model_cls=model_cls)


def _load_base_weights(model, state_dict):
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    missing = [name for name in missing if not name.startswith(MULTIVIEW_PREFIXES)]

    if missing or unexpected:
        raise RuntimeError(f"Checkpoint 不匹配：missing={missing}, unexpected={unexpected}")

def _build_from_sam_checkpoint(
    checkpoint_path,
    image_size,
    num_latents,
    num_aggregator_layers,
    num_distributor_layers,
    adapter_args,
):
    model = _create_model(
        image_size,
        num_latents,
        num_aggregator_layers,
        num_distributor_layers,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = DuplicateMaskDecoderWeights()(checkpoint["model"])
    state_dict = DuplicateMemoryWeights()(state_dict)
    _load_base_weights(model, state_dict)

    inject_sam2_modified_adapters(model=model, **adapter_args)
    return model


def _build_from_memory_checkpoint(
    checkpoint_path,
    num_latents,
    num_aggregator_layers,
    num_distributor_layers,
):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = checkpoint["args"]

    model = _create_model(
        checkpoint_args["image_size"],
        num_latents,
        num_aggregator_layers,
        num_distributor_layers,
    )

    adapter_args = {key: checkpoint_args[key] for key in ADAPTER_KEYS}
    inject_sam2_modified_adapters(model=model, **adapter_args)
    _load_base_weights(model, checkpoint["model_state"])
    return model

def build_sam2_multiview_dual_hand_memory_tiny(
    num_aggregator_layers,
    num_distributor_layers,
    sam_checkpoint=None,
    memory_checkpoint=None,
    device="cpu",
    mode="eval",
    image_size=768,
    num_latents=144,
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
    if (sam_checkpoint is None) == (memory_checkpoint is None):
        raise ValueError("sam_checkpoint 和 memory_checkpoint 必须且只能提供一个")
    if mode not in {"train", "eval"}:
        raise ValueError(f"Invalid mode: {mode}")

    if sam_checkpoint is not None:
        adapter_args = {
            "use_image_adapter": use_image_adapter,
            "use_decoder_adapter": use_decoder_adapter,
            "adapter_dim": adapter_dim,
            "adapter_dropout": adapter_dropout,
            "adapter_init_scale": adapter_init_scale,
        }
        model = _build_from_sam_checkpoint(
            sam_checkpoint,
            image_size,
            num_latents,
            num_aggregator_layers,
            num_distributor_layers,
            adapter_args,
        )
    else:
        model = _build_from_memory_checkpoint(
            memory_checkpoint,
            num_latents,
            num_aggregator_layers,
            num_distributor_layers,
        )

    _configure_prompt_sampling(
        model,
        num_init_cond_frames_for_train,
        num_frames_to_correct_for_train,
        add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame,
    )

    model = model.to(device)
    model.train() if mode == "train" else model.eval()
    return model


def configure_multiview_training(model, finetune_mode):
    """
    配置多视角模型的可训练参数。

    multiview-only:
        只训练 Aggregator 和 Distributor。

    multiview-memory:
        训练 Aggregator、Distributor 和左右 Memory。

    multiview-memory-sam1:
        训练帧级 SAM 模块、左右 Memory、Aggregator 和 Distributor；
        Adapter 的处理沿用帧级 SAM 微调设置。
    """

    # 1. 先按照模式配置原有 SAM2 和 Memory 参数
    if finetune_mode == "multiview-only":
        model.requires_grad_(False)
    elif finetune_mode == "multiview-memory":
        configure_memory_training(model, "memory-only")
    elif finetune_mode == "multiview-memory-sam1":
        configure_memory_training(model, "decoder-memory")
    else:
        raise ValueError(f"Invalid finetune mode: {finetune_mode}")

    # 2. 无论选择哪种模式，新增的多视角模块都必须训练
    model.left_multiview_aggregator.requires_grad_(True)
    model.right_multiview_aggregator.requires_grad_(True)
    model.left_multiview_distributor.requires_grad_(True)
    model.right_multiview_distributor.requires_grad_(True)

    # 3. 统计最终实际参与训练的参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logging.info(
        "Multiview parameters | mode=%s | trainable=%s/%s (%.2f%%)",
        finetune_mode,
        f"{trainable_params:,}",
        f"{total_params:,}",
        100.0 * trainable_params / total_params,
    )