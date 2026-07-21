import argparse
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .builder import build_sam2_modified_tiny, configure_finetune_stage
from .dataset import build_dataloaders
from .trainer import run_training_epoch, run_validation_epoch
from .utils import configure_runtime, dump_json, save_checkpoint, save_training_curves, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAM2Modified for framewise dual-hand segmentation.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=(REPO_ROOT/ "checkpoints"/ "sam2.1_hiera_tiny.pt"))
    parser.add_argument("--use-predict-mask", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--dataset-percent", type=int, default=10, choices=[10, 20, 50, 100])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:1" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--disable-augmentation", action="store_true")

    # dataset
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--frames-per-second", type=float, default=2.0)
    parser.add_argument("--dataset-names", nargs="+", default=None)
    parser.add_argument("--test-seq-count", type=int, default=2)

    # Loss Weights
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=0.1)
    parser.add_argument("--object-score-weight", type=float, default=0)

    # Optimizer & Scheduler
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "radam"], default="adamw")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    # Adapter Config
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-init-scale", type=float, default=1e-3)

    # Logging
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--skip-visualizations", action="store_true")

    args = parser.parse_args()
    if args.val_batch_size <= 0:
        args.val_batch_size = max(args.batch_size, 1)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")

    return args

def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found.")

    optimizer_map = {"adam": Adam, "adamw": AdamW, "radam": RAdam}
    optimizer_cls = optimizer_map[args.optimizer]
    optimizer_kwargs: dict[str, Any] = {"lr": args.lr, "weight_decay": args.weight_decay}

    if args.device.startswith("cuda") and args.optimizer in {"adam", "adamw"}:
        optimizer_kwargs["fused"] = True

    return optimizer_cls(trainable_parameters, **optimizer_kwargs)

def configure_model(
        args: argparse.Namespace,
) -> torch.nn.Module:
    model = build_sam2_modified_tiny(
        checkpoint_path=args.sam_checkpoint,
        device=args.device,
        mode="train",
        image_size=args.image_size,
    )

    logging.info(
        "Model built. Total parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    return model

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32, channels_last=args.channels_last)

    
    train_loader, val_loader, train_sampler = build_dataloaders(args, device)
    
    
    model = configure_model(args)

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    
    configure_finetune_stage(model)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    optimizer = build_optimizer(args, model)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
                                  min_lr=args.min_lr)

    best_val_iou = float("-inf")
    history: list[dict[str, float]] = []
    checkpoints_dir = args.output_dir / "checkpoints"

    epochs_without_improvement = 0
    early_stopped = False
    for epoch in range(args.epochs):
        logging.info("--- Epoch %d/%d ---",epoch + 1, args.epochs)

        # 每轮改变训练样本的排列顺序。
        train_sampler.set_epoch(epoch)

        train_metrics = run_training_epoch(model=model, loader=train_loader, optimizer=optimizer, scaler=scaler, device=device, args=args, epoch=epoch)

        logging.info("TRAINING | epoch=%d | train_loss=%.4f", epoch + 1, train_metrics["loss"])

        should_validate = ((epoch + 1) % max(args.val_interval, 1) == 0 or epoch == args.epochs - 1)
        if should_validate:
            val_metrics = run_validation_epoch(model=model, loader=val_loader, device=device, epoch=epoch, args=args)
            scheduler.step(val_metrics["loss"])

            if device.type == "cuda":
                torch.cuda.empty_cache()

            logging.info(
                "VALIDATION | epoch=%d | "
                "val_loss=%.4f | val_iou=%.4f | val_dice=%.4f",
                epoch + 1,
                val_metrics["loss"],
                val_metrics["iou"],
                val_metrics["dice"],
            )
        else:
            val_metrics = {
                "loss": float("nan"),
                "iou": float("nan"),
                "dice": float("nan"),
            }

        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_metrics = {
            "epoch": float(epoch + 1),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "val_iou": float(val_metrics["iou"]),
            "val_dice": float(val_metrics["dice"]),
        }

        history.append(epoch_metrics)

        improved = (
            math.isfinite(val_metrics["iou"])
            and val_metrics["iou"]
            > best_val_iou + args.early_stop_min_delta
        )

        if improved:
            best_val_iou = float(val_metrics["iou"])
            epochs_without_improvement = 0
        elif should_validate:
            epochs_without_improvement += 1

        save_checkpoint(
            output_path=checkpoints_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            scheduler=scheduler,
            epoch=epoch,
            best_val_iou=best_val_iou,
            args=args,
            metrics=epoch_metrics,
            history=history,
            include_optimizer_state=True,
        )

        if improved:
            save_checkpoint(
                output_path=checkpoints_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                scheduler=scheduler,
                epoch=epoch,
                best_val_iou=best_val_iou,
                args=args,
                metrics=epoch_metrics,
                history=history,
                include_optimizer_state=False,
            )

            logging.info(
                "New best checkpoint | epoch=%d | val_iou=%.4f",
                epoch + 1,
                best_val_iou,
            )

        dump_json(
            {
                "best_val_iou": best_val_iou,
                "early_stopped": False,
                "epochs": history,
            },
            args.output_dir / "metrics.json",
        )

        if not args.skip_visualizations:
            save_training_curves(
                history=history,
                output_path=(
                    args.output_dir
                    / "visualizations"
                    / "training_curves.png"
                ),
            )

        if (
            should_validate
            and args.early_stop_patience > 0
            and epochs_without_improvement
            >= args.early_stop_patience
        ):
            early_stopped = True

            logging.info("Early stopping triggered after %d ""validation epochs without improvement", epochs_without_improvement)
            break

    dump_json(
        {
            "best_val_iou": best_val_iou,
            "early_stopped": early_stopped,
            "epochs": history,
        },
        args.output_dir / "metrics.json",
    )

    logging.info(
        "TRAINING COMPLETE | best_val_iou=%.4f | early_stopped=%s",
        best_val_iou,
        early_stopped,
    )


if __name__ == "__main__":
    main()

