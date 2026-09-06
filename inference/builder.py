"""按训练 checkpoint 创建推理模型。"""

import argparse
import logging

import torch

from projects.framewise_sam2_modified.builder import create_sam2_modified_tiny, inject_sam2_modified_adapters
from training.model.sam2_modified import SAM2Modified
from inference.sam2_dual_hand_video_predictor import SAM2DualHandVideoPredictor


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    model_cls = SAM2DualHandVideoPredictor if args.model == "memory" else SAM2Modified
    model = create_sam2_modified_tiny(
        image_size=saved_args["image_size"],
        model_cls=model_cls,
    )
    inject_sam2_modified_adapters(
        model=model,
        use_image_adapter=saved_args["use_image_adapter"],
        use_decoder_adapter=saved_args["use_decoder_adapter"],
        adapter_dim=saved_args["adapter_dim"],
        adapter_dropout=saved_args["adapter_dropout"],
        adapter_init_scale=saved_args["adapter_init_scale"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(args.device).eval()
    args.image_size = saved_args["image_size"]
    logging.info("Loaded %s checkpoint: %s", args.model, args.model_checkpoint)
    return model
