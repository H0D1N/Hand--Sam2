"""Evaluate the unfinetuned dual-hand Memory model on the validation split."""

import logging

import torch

from .builder import build_sam2_dual_hand_memory_tiny
from .losses import DualHandMemoryLoss
from .trainer import run_validation_epoch
from projects.framewise_sam2_modified.utils import configure_runtime, dump_json, set_seed
from projects.zero_shot_common import build_zero_shot_loader, parse_args


def main() -> None:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)

    # 只加载官方 SAM2 checkpoint，并复制初始化左右手 Decoder 和 Memory。
    # 不加载任何双手训练 checkpoint，也不创建训练相关组件。
    model = build_sam2_dual_hand_memory_tiny(
        sam_checkpoint=args.sam_checkpoint,
        framewise_checkpoint=None,
        device=device,
        mode="eval",
        image_size=args.image_size,
    )
    args.image_size = model.image_size

    val_loader = build_zero_shot_loader(args, device)
    loss_fn = DualHandMemoryLoss(
        mask_loss_weight=args.mask_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        iou_loss_weight=args.iou_loss_weight,
        class_loss_weight=args.class_loss_weight,
    ).to(device)

    # 第 0 帧使用 GT 条件，只从第 1 帧开始统计 tracking 指标。
    validation_metrics = run_validation_epoch(
        model=model,
        loss_fn=loss_fn,
        loader=val_loader,
        device=device,
        args=args,
        epoch=0,
        metric_start_frame=1,
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
        "ZERO-SHOT MEMORY BASELINE COMPLETE | "
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
