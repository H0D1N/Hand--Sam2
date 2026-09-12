"""Evaluate an initialized or trained dual-hand Memory model."""

import logging

import torch

from ...dual_hand_memory.builder import (
    build_sam2_dual_hand_memory_tiny,
    load_sam2_dual_hand_memory_tiny,
)
from ...dual_hand_memory.losses import DualHandMemoryLoss
from ...dual_hand_memory.trainer import run_validation_epoch
from projects.framewise_sam2_modified.utils import configure_runtime, dump_json, set_seed
from projects.hardcases.evaluation_common import build_zero_shot_loader, parse_args



def main() -> None:
    args = parse_args()

    if args.multiview_checkpoint is not None:
        raise ValueError(
            "Memory 只能从 --sam-checkpoint、--framewise-checkpoint "
            "或完整的 --memory-checkpoint 加载"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)

    if args.memory_checkpoint is not None:
        model = load_sam2_dual_hand_memory_tiny(args.memory_checkpoint, device=device)
    else:
        model = build_sam2_dual_hand_memory_tiny(
            sam_checkpoint=args.sam_checkpoint,
            framewise_checkpoint=args.framewise_checkpoint,
            device=device,
            mode="eval",
            image_size=args.image_size,
        )

    # 评测协议独立于 checkpoint 中保存的训练配置。
    model.num_correction_pt_per_frame = args.num_correction_pt_per_frame
    model.add_all_frames_to_correct_as_cond = args.add_all_frames_to_correct_as_cond
    args.image_size = model.image_size

    val_loader = build_zero_shot_loader(args, device)
    loss_fn = DualHandMemoryLoss(
        mask_loss_weight=args.mask_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        iou_loss_weight=args.iou_loss_weight,
        class_loss_weight=args.class_loss_weight,
    ).to(device)

    validation_metrics = run_validation_epoch(
        model=model,
        loss_fn=loss_fn,
        loader=val_loader,
        device=device,
        args=args,
        epoch=0,
        metric_start_frame=args.metric_start_frame,
        correction_frame_indices=args.correction_frame_indices,
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
        "MEMORY EVALUATION COMPLETE | "
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
