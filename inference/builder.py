"""按训练 checkpoint 创建推理模型。"""

import argparse
import logging
from pathlib import Path

import torch

from projects.dual_hand_memory.builder import _configure_prompt_sampling
from projects.dual_hand_multiview.builder import (
    _create_model as create_multiview_model,
)
from projects.framewise_sam2_modified.builder import (
    build_sam2_modified_tiny,
    create_sam2_modified_tiny,
    inject_sam2_modified_adapters,
)
from inference.sam2_dual_hand_video_predictor import SAM2DualHandVideoPredictor
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory
from training.model.sam2_modified import SAM2Modified


ADAPTER_DEFAULTS = {
    "use_image_adapter": False,
    "use_decoder_adapter": False,
    "adapter_dim": 64,
    "adapter_dropout": 0.1,
    "adapter_init_scale": 1e-3,
}


def _load_training_checkpoint(path: str | Path) -> tuple[Path, dict, dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"找不到模型 checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if "args" not in checkpoint or "model_state" not in checkpoint:
        raise ValueError("checkpoint 必须包含 args 和 model_state")
    return path, checkpoint, checkpoint["args"]


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


def load_trained_model(
    model_type: str,
    checkpoint_path,
    device,
    *,
    memory_predictor: bool = False,
) -> torch.nn.Module:
    """从完整训练 checkpoint 重建三种项目模型。"""
    path, checkpoint, saved_args = _load_training_checkpoint(checkpoint_path)
    image_size = saved_args["image_size"]

    if model_type == "framewise":
        model = create_sam2_modified_tiny(
            image_size=image_size,
            model_cls=SAM2Modified,
        )
    elif model_type == "memory":
        model = create_sam2_modified_tiny(
            image_size=image_size,
            model_cls=(
                SAM2DualHandVideoPredictor
                if memory_predictor
                else SAM2DualHandMemory
            ),
        )
    elif model_type == "multiview":
        model = create_multiview_model(
            image_size=image_size,
            num_latents=saved_args["num_latents"],
            num_aggregator_layers=saved_args["num_aggregator_layers"],
            num_distributor_layers=saved_args["num_distributor_layers"],
            multiview_residual_scale_init=saved_args.get(
                "multiview_residual_scale_init", 1e-3
            ),
        )
    else:
        raise ValueError(f"不支持的 model_type: {model_type}")

    inject_sam2_modified_adapters(
        model=model,
        **{
            key: saved_args.get(key, default)
            for key, default in ADAPTER_DEFAULTS.items()
        },
    )
    if model_type in {"memory", "multiview"}:
        _configure_saved_prompt_settings(model, saved_args)
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(device).eval()
    logging.info(
        "Loaded %s checkpoint | %s | epoch=%s",
        model_type,
        path,
        checkpoint.get("epoch", "unknown"),
    )
    return model


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    if args.model == "sam2":
        model = build_sam2_modified_tiny(
            checkpoint_path=args.model_checkpoint,
            device=args.device,
            mode="eval",
            image_size=args.image_size,
        )
        logging.info("Loaded original SAM2 checkpoint: %s", args.model_checkpoint)
        return model

    model = load_trained_model(
        args.model,
        args.model_checkpoint,
        args.device,
        memory_predictor=args.model == "memory",
    )
    args.image_size = model.image_size
    return model
