"""不依赖训练初始化 checkpoint，直接重建并加载完整评估模型。"""

from pathlib import Path
import logging

import torch

from projects.dual_hand_memory.builder import _configure_prompt_sampling
from projects.dual_hand_multiview.builder import _create_model as create_multiview_model
from projects.framewise_sam2_modified.builder import (
    create_sam2_modified_tiny,
    inject_sam2_modified_adapters,
)
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory


ADAPTER_DEFAULTS = {
    "use_image_adapter": False,
    "use_decoder_adapter": False,
    "adapter_dim": 64,
    "adapter_dropout": 0.1,
    "adapter_init_scale": 1e-3,
}


def _load_checkpoint(path: str | Path) -> tuple[Path, dict, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型 checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "args" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("评估必须使用包含 args 和 model_state 的训练 checkpoint")
    return path, checkpoint, checkpoint["args"]


def _inject_saved_adapters(model, checkpoint_args: dict) -> None:
    inject_sam2_modified_adapters(
        model=model,
        **{
            key: checkpoint_args.get(key, default)
            for key, default in ADAPTER_DEFAULTS.items()
        },
    )


def _configure_saved_prompt_settings(model, checkpoint_args: dict) -> None:
    _configure_prompt_sampling(
        model=model,
        num_init_cond_frames_for_train=checkpoint_args.get(
            "num_init_cond_frames_for_train", 2
        ),
        num_frames_to_correct_for_train=checkpoint_args.get(
            "num_frames_to_correct_for_train", 2
        ),
        add_all_frames_to_correct_as_cond=checkpoint_args.get(
            "add_all_frames_to_correct_as_cond", True
        ),
        num_correction_pt_per_frame=checkpoint_args.get(
            "num_correction_pt_per_frame", 7
        ),
    )


def load_evaluation_model(model_type: str, checkpoint_path, device):
    path, checkpoint, checkpoint_args = _load_checkpoint(checkpoint_path)
    image_size = checkpoint_args["image_size"]

    if model_type == "memory":
        model = create_sam2_modified_tiny(
            image_size=image_size,
            model_cls=SAM2DualHandMemory,
        )
    elif model_type == "multiview":
        model = create_multiview_model(
            image_size=image_size,
            num_latents=checkpoint_args["num_latents"],
            num_aggregator_layers=checkpoint_args["num_aggregator_layers"],
            num_distributor_layers=checkpoint_args["num_distributor_layers"],
        )
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    _inject_saved_adapters(model, checkpoint_args)
    _configure_saved_prompt_settings(model, checkpoint_args)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    logging.info(
        "Loaded %s checkpoint | %s | epoch=%s",
        model_type,
        path,
        checkpoint.get("epoch", "unknown"),
    )
    return model

