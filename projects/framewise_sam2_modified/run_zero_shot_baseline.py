"""Evaluate the unfinetuned dual-decoder structure on the validation split."""

import logging

import torch

from .dataset import build_dataloaders, build_center_point_prompt
from .run_train import configure_model, parse_args
from .trainer import run_validation_epoch
from .utils import configure_runtime, dump_json, set_seed

def main() -> None:
    args = parse_args()
    args.skip_visualizations = True
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

    _, val_loader, _ = build_dataloaders(args, device)

    # 只加载官方 SAM2 checkpoint，并按训练配置创建双 Decoder 和 Adapter。
    # 不加载任何双手训练 checkpoint，也不创建 optimizer。
    model = configure_model(args)

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    point_prompt_fn = build_center_point_prompt if args.use_point_prompt else None
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
