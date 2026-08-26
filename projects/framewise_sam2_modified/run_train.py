import argparse
from collections import defaultdict
import logging
import math
from pathlib import Path
from typing import Any

import torch
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .builder import build_sam2_modified_tiny, configure_finetune_stage
from .dataset import build_dataloaders, build_center_point_prompt
from .trainer import log_tensorboard_probe, run_training_epoch, run_validation_epoch
from .utils import configure_runtime, dump_json, save_checkpoint, save_training_curves, set_seed

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM_CHECKPOINT = REPO_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = "xingyi_4-5090_oak150-100output", "wuwen_4-5090_release-0623-compressed", "tencent_4-5090_7.5"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train SAM2Modified for framewise dual-hand segmentation.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--use-predict-mask", action="store_true")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--val-batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--multimask-output", action="store_true")
    parser.add_argument("--use-point-prompt", action="store_true")
    
    # Dataset
    parser.add_argument("--dataset", dest="dataset_mode", choices=("multiserver", "dexycb", "mixed"), default="mixed")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--frames-per-second", type=float, default=3.0)
    parser.add_argument("--test-seq-count", type=int, default=3)
    parser.add_argument("--dex-ycb-root", type=Path)

    # Loss Weights
    parser.add_argument("--bce-weight", type=float, default=1.0)
    parser.add_argument("--dice-weight", type=float, default=1.0)
    parser.add_argument("--iou-weight", type=float, default=0.1)
    parser.add_argument("--object-score-weight", type=float, default=1.0)

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
    parser.add_argument("--use-image-adapter", action="store_true")
    parser.add_argument("--use-decoder-adapter", action="store_true")
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-init-scale", type=float, default=1e-3)

    # Logging
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-log-interval", type=int, default=1)
    parser.add_argument("--tensorboard-probe-interval", type=int, default=500)
    parser.add_argument("--tensorboard-probe-image-size", type=int, default=384)
    parser.add_argument("--skip-visualizations", action="store_true")

    args = parser.parse_args()
    if args.val_batch_size <= 0:
        args.val_batch_size = max(args.batch_size, 1)
    if args.grad_accum_steps < 1:
        raise ValueError("--grad-accum-steps must be >= 1")
    if args.tensorboard_log_interval < 1:
        raise ValueError("--tensorboard-log-interval must be >= 1")
    if args.tensorboard_probe_image_size < 1:
        raise ValueError("--tensorboard-probe-image-size must be >= 1")
    if args.dataset_mode in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error("--dataset dexycb/mixed 需要提供 --dex-ycb-root")
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
        use_image_adapter=args.use_image_adapter,
        use_decoder_adapter=args.use_decoder_adapter,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
        adapter_init_scale=args.adapter_init_scale,
    )

    logging.info(
        "Model built. Total parameters: %s",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )

    return model


def build_tensorboard_probe(val_loader) -> dict | None:
    """从每个有效的 val dataset/camera 固定选择一张中间帧。"""
    dataset = val_loader.dataset
    selected_indices = set(val_loader.sampler)
    streams_by_group: dict[
        tuple[str, str],
        list[list[int]],
    ] = defaultdict(list)

    for stream_id, stream in dataset.streams.items():
        dataset_name = stream["dataset_name"]
        camera_name = stream_id.rsplit("/", 1)[-1]
        stream_candidates = [
            sample_index
            for sample_index in stream["sample_indices"]
            if sample_index in selected_indices
        ]

        if stream_candidates:
            streams_by_group[(dataset_name, camera_name)].append(
                stream_candidates
            )

    probe_items = []
    probe_labels = []
    probe_dataset_names = []

    for (dataset_name, camera_name), group_streams in sorted(
            streams_by_group.items()
    ):
        stream_candidates = group_streams[len(group_streams) // 2]
        sample_index = stream_candidates[len(stream_candidates) // 2]
        item = dataset[sample_index]

        probe_items.append(item)
        probe_labels.append(f"{dataset_name}/{camera_name}")
        probe_dataset_names.append(dataset_name)

        logging.info(
            "TensorBoard probe | %s/%s | %s",
            dataset_name,
            camera_name,
            item["sample_id"],
        )

    if not probe_items:
        logging.warning("TensorBoard probe 没有选出任何有效 val 样本")
        return None

    return {
        "image": torch.stack([item["image"] for item in probe_items]),
        "left_mask": torch.stack([item["left_mask"] for item in probe_items]),
        "right_mask": torch.stack([item["right_mask"] for item in probe_items]),
        "label": probe_labels,
        "dataset_name": probe_dataset_names,
    }

def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32, channels_last=args.channels_last)

    
    train_loader, val_loader, train_sampler = build_dataloaders(args, device)
    point_prompt_fn = (build_center_point_prompt if args.use_point_prompt else None)
    
    model = configure_model(args)

    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    
    configure_finetune_stage(model, args.use_decoder_adapter)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    optimizer = build_optimizer(args, model)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
                                  min_lr=args.min_lr)

    tensorboard_writer = None
    tensorboard_probe = None

    if not args.disable_tensorboard:
        tensorboard_writer = SummaryWriter(
            log_dir=str(args.output_dir / "tensorboard"),
            max_queue=100,
            flush_secs=10,
        )
        tensorboard_writer.add_text(
            "run/config",
            "  \n".join(
                f"{name}: {value}"
                for name, value in sorted(vars(args).items())
            ),
            global_step=0,
        )
        tensorboard_writer.add_text(
            "probe/panel_order",
            "original | ground truth | prediction",
            global_step=0,
        )
        tensorboard_probe = build_tensorboard_probe(val_loader)

        if tensorboard_probe is not None:
            log_tensorboard_probe(
                model=model,
                probe=tensorboard_probe,
                writer=tensorboard_writer,
                device=device,
                args=args,
                global_step=0,
                point_prompt_fn=point_prompt_fn,
            )

        logging.info(
            "TensorBoard logs: %s",
            args.output_dir / "tensorboard",
        )

    best_val_iou = float("-inf")
    history: list[dict] = []
    checkpoints_dir = args.output_dir / "checkpoints"

    epochs_without_improvement = 0
    early_stopped = False
    for epoch in range(args.epochs):
        logging.info("--- Epoch %d/%d ---",epoch + 1, args.epochs)

        # 每轮改变训练样本的排列顺序。
        train_sampler.set_epoch(epoch)

        train_metrics = run_training_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            args=args,
            epoch=epoch,
            tensorboard_writer=tensorboard_writer,
            tensorboard_probe=tensorboard_probe,
            point_prompt_fn=point_prompt_fn,
        )

        logging.info("TRAINING | epoch=%d | train_loss=%.4f", epoch + 1, train_metrics["loss"])

        if tensorboard_writer is not None:
            epoch_end_step = (epoch + 1) * len(train_loader)
            tensorboard_writer.add_scalar(
                "epoch/train_loss",
                train_metrics["loss"],
                epoch_end_step,
            )

        should_validate = ((epoch + 1) % max(args.val_interval, 1) == 0 or epoch == args.epochs - 1)
        if should_validate:
            val_metrics = run_validation_epoch(model=model, loader=val_loader, device=device, epoch=epoch, args=args, point_prompt_fn=point_prompt_fn,)
            overall_val = val_metrics["overall"]
            scheduler.step(overall_val["loss"])

            if tensorboard_writer is not None:
                for group, group_metrics in val_metrics.items():
                    for metric_name, metric_value in group_metrics.items():
                        if metric_value is not None:
                            tensorboard_writer.add_scalar(f"validation/{group}/{metric_name}", metric_value, epoch_end_step)
                tensorboard_writer.flush()

            if device.type == "cuda":
                torch.cuda.empty_cache()

            logging.info(
                "VALIDATION | epoch=%d | "
                "val_loss=%.4f | val_iou=%.4f | val_dice=%.4f | "
                "obj_acc=%.4f | obj_precision=%.4f | "
                "obj_recall=%.4f | obj_f1=%.4f",
                epoch + 1,
                overall_val["loss"],
                overall_val["iou"],
                overall_val["dice"],
                overall_val["object_accuracy"],
                overall_val["object_precision"],
                overall_val["object_recall"],
                overall_val["object_f1"],
            )
        else:
            empty_metrics = {name: float("nan") for name in ("loss", "iou", "dice", "object_accuracy", "object_precision", "object_recall", "object_f1")}
            val_metrics = {group: dict(empty_metrics) for group in ("overall", "multiserver", "dexycb")}
            overall_val = val_metrics["overall"]

        current_lr = float(optimizer.param_groups[0]["lr"])

        epoch_metrics = {
            "epoch": float(epoch + 1),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "validation": val_metrics,
        }

        history.append(epoch_metrics)

        improved = (
            math.isfinite(overall_val["iou"])
            and overall_val["iou"]
            > best_val_iou + args.early_stop_min_delta
        )

        if improved:
            best_val_iou = float(overall_val["iou"])
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
                history=[{
                    "epoch": row["epoch"], "train_loss": row["train_loss"],
                    "val_loss": row["validation"]["overall"]["loss"],
                    "val_iou": row["validation"]["overall"]["iou"],
                    "val_dice": row["validation"]["overall"]["dice"],
                } for row in history],
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

    if tensorboard_writer is not None:
        tensorboard_writer.flush()
        tensorboard_writer.close()


if __name__ == "__main__":
    main()
