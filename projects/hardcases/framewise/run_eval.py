"""Evaluate the unfinetuned dual-decoder structure on the validation split."""

import logging

import torch

from ...framewise_sam2_modified.builder import build_sam2_modified_tiny
from ...framewise_sam2_modified.dataset import build_center_point_prompt
from ...framewise_sam2_modified.trainer import run_validation_epoch
from ...framewise_sam2_modified.utils import configure_runtime, dump_json, set_seed
from projects.dual_hand_memory.dataset import collate_clip_batch
from projects.hardcases.evaluation_common import build_zero_shot_loader, parse_args


def collate_tracking_frames(items):
    """丢掉条件帧，并把 [B,T-1,...] 展平成 Framewise 的 [B*(T-1),...]。"""
    batch = collate_clip_batch(items)
    tracking_frames = batch["image"].size(1) - 1
    return {
        "image": batch["image"][:, 1:].flatten(0, 1),
        "left_mask": batch["left_mask"][:, 1:].flatten(0, 1),
        "right_mask": batch["right_mask"][:, 1:].flatten(0, 1),
        "original_image": [value for sequence in batch["original_image"] for value in sequence[1:]],
        "original_left_mask": [value for sequence in batch["original_left_mask"] for value in sequence[1:]],
        "original_right_mask": [value for sequence in batch["original_right_mask"] for value in sequence[1:]],
        "sample_id": [value for sequence in batch["sample_id"] for value in sequence[1:]],
        "dataset_name": [name for name in batch["dataset_name"] for _ in range(tracking_frames)],
    }

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(
        device,
        use_tf32=not args.disable_tf32,
        channels_last=args.channels_last,
    )

    val_loader = build_zero_shot_loader(args, device, collate_fn=collate_tracking_frames)

    # 官方 SAM2 权重复制为左右 Decoder；zero-shot 不注入未训练的 Adapter。
    model = build_sam2_modified_tiny(
        checkpoint_path=args.sam_checkpoint, device=args.device,
        mode="eval", image_size=args.image_size,
    )

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    point_prompt_fn = build_center_point_prompt if args.prompt_mode == "point" else None
    validation_metrics = run_validation_epoch(
        model=model,
        loader=val_loader,
        device=device,
        epoch=0,
        args=args,
        point_prompt_fn=point_prompt_fn,
    )
    overall = validation_metrics["overall"]
    epoch_metrics = {
        "epoch": 0.0,
        "lr": None,
        "train_loss": None,
        "validation": validation_metrics,
    }

    dump_json(
        {
            "best_val_iou": float(overall["iou"]),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "epochs": [epoch_metrics],
        },
        args.output_dir / "metrics.json",
    )

    logging.info(
        "ZERO-SHOT BASELINE COMPLETE | "
        "val_loss=%.4f | val_iou=%.4f | val_dice=%.4f | "
        "obj_acc=%.4f | obj_precision=%.4f | "
        "obj_recall=%.4f | obj_f1=%.4f",
        overall["loss"],
        overall["iou"],
        overall["dice"],
        overall["object_accuracy"],
        overall["object_precision"],
        overall["object_recall"],
        overall["object_f1"],
    )


if __name__ == "__main__":
    main()
