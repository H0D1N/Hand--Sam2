"""训练多视角双手 Memory SAM2。"""

import argparse
import logging
from pathlib import Path

import torch
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from projects.framewise_sam2_modified.utils import configure_runtime, dump_json, save_checkpoint, save_training_curves, set_seed
from training.model.sam2_multiview_dual_hand_memory import PromptRequest
from .builder import build_sam2_multiview_dual_hand_memory_tiny, configure_multiview_training
from .dataset import build_dataloaders
from .losses import MultiViewDualHandMemoryLoss
from .trainer import run_training_epoch, run_validation_epoch


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM_CHECKPOINT = REPO_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = "xingyi_4-5090_oak150-100output", "wuwen_4-5090_release-0623-compressed", "tencent_4-5090_7.5"


def parse_args():
    parser = argparse.ArgumentParser(description="Train multiview dual-hand Memory SAM2.")

    # 运行环境
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")

    # 初始化与模型结构
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--sam-checkpoint", type=Path)
    init_group.add_argument("--memory-checkpoint", type=Path)
    parser.add_argument("--resume-checkpoint", type=Path)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--num-views", type=int, default=2)
    parser.add_argument("--num-latents", type=int, default=144)
    parser.add_argument("--num-aggregator-layers", type=int, default=2)
    parser.add_argument("--num-distributor-layers", type=int, default=1)
    parser.add_argument(
        "--finetune-mode",
        choices=("multiview-only", "multiview-memory", "multiview-memory-sam1"),
        default="multiview-only",
    )

    # Adapter
    parser.add_argument("--use-image-adapter", action="store_true")
    parser.add_argument("--use-decoder-adapter", action="store_true")
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-init-scale", type=float, default=1e-3)

    # PromptRequest；auto 只使用模型配置，point/mask 使用明确帧下标
    parser.add_argument("--prompt-mode", choices=("auto", "point", "mask"), default="auto")
    parser.add_argument("--start-frame-idx", type=int, default=0)
    parser.add_argument("--prompt-frame-indices", type=int, nargs="+", default=(0,))
    parser.add_argument("--correction-frame-indices", type=int, nargs="*")
    parser.add_argument("--num-init-cond-frames-for-train", type=int, default=2)
    parser.add_argument("--num-frames-to-correct-for-train", type=int, default=2)
    parser.add_argument("--add-all-frames-to-correct-as-cond", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-correction-pt-per-frame", type=int, default=7)

    # Dataset 与 Clip
    parser.add_argument("--dataset", dest="dataset_mode", choices=("multiserver", "dexycb", "mixed"), default="mixed")
    parser.add_argument("--disable-augmentation", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--test-seq-count", type=int, default=3)
    parser.add_argument("--dex-ycb-root", type=Path)
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=80)
    parser.add_argument("--val-clip-stride", type=int, default=8)
    parser.add_argument(
        "--encoder-chunk-size",
        type=int,
        default=1,
        help="每次送入 Image Encoder 的图像数；多视角仍在特征层同时融合",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    # Loss 与优化器
    parser.add_argument("--mask-loss-weight", type=float, default=20.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "radam"), default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # 日志、可视化与停止条件
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--tensorboard-log-interval", type=int, default=10)
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--max-vis-per-dataset", type=int, default=100)
    parser.add_argument("--metric-start-frame", type=int, default=0)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)

    args = parser.parse_args()
    if args.sam_checkpoint is None and args.memory_checkpoint is None:
        args.sam_checkpoint = DEFAULT_SAM_CHECKPOINT
    if args.dataset_mode in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error("--dataset dexycb/mixed 需要提供 --dex-ycb-root")
    if not 0 <= args.start_frame_idx < args.clip_length:
        parser.error("--start-frame-idx 必须位于 Clip 内")
    if args.prompt_mode != "auto":
        frame_indices = list(args.prompt_frame_indices) + list(args.correction_frame_indices or ())
        if any(index < args.start_frame_idx or index >= args.clip_length for index in frame_indices):
            parser.error("Prompt 或纠错帧下标超出有效范围")
    if not 1 <= args.num_init_cond_frames_for_train <= args.num_frames_to_correct_for_train <= args.clip_length:
        parser.error("必须满足 1 <= 初始条件帧数 <= 纠错帧数 <= clip-length")
    if args.encoder_chunk_size < 1:
        parser.error("--encoder-chunk-size 必须大于 0")
    return args


def build_optimizer(args, model):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise RuntimeError("没有可训练参数")

    optimizer_cls = {"adam": Adam, "adamw": AdamW, "radam": RAdam}[args.optimizer]
    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}
    if args.device.startswith("cuda") and args.optimizer in {"adam", "adamw"}:
        optimizer_kwargs["fused"] = True
    return optimizer_cls(parameters, **optimizer_kwargs)


def build_prompt_request(args):
    """auto 忽略明确帧下标；point/mask 完全使用这些下标。"""
    if args.prompt_mode == "auto":
        return PromptRequest(mode="auto", start_frame_idx=args.start_frame_idx)

    return PromptRequest(
        mode=args.prompt_mode,
        start_frame_idx=args.start_frame_idx,
        prompt_frame_indices=tuple(args.prompt_frame_indices),
        correction_frame_indices=None if args.correction_frame_indices is None else tuple(args.correction_frame_indices),
        num_correction_points_per_frame=args.num_correction_pt_per_frame,
        add_correction_frames_as_cond=args.add_all_frames_to_correct_as_cond,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)

    model = build_sam2_multiview_dual_hand_memory_tiny(
        sam_checkpoint=args.sam_checkpoint,
        memory_checkpoint=args.memory_checkpoint,
        device=device,
        mode="train",
        image_size=args.image_size,
        num_latents=args.num_latents,
        num_aggregator_layers=args.num_aggregator_layers,
        num_distributor_layers=args.num_distributor_layers,
        use_image_adapter=args.use_image_adapter,
        use_decoder_adapter=args.use_decoder_adapter,
        adapter_dim=args.adapter_dim,
        adapter_dropout=args.adapter_dropout,
        adapter_init_scale=args.adapter_init_scale,
        num_init_cond_frames_for_train=args.num_init_cond_frames_for_train,
        num_frames_to_correct_for_train=args.num_frames_to_correct_for_train,
        add_all_frames_to_correct_as_cond=args.add_all_frames_to_correct_as_cond,
        num_correction_pt_per_frame=args.num_correction_pt_per_frame,
    )
    args.image_size = model.image_size
    configure_multiview_training(model, args.finetune_mode)

    loss_fn = MultiViewDualHandMemoryLoss(
        mask_loss_weight=args.mask_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        iou_loss_weight=args.iou_loss_weight,
        class_loss_weight=args.class_loss_weight,
    ).to(device)
    train_loader, val_loader = build_dataloaders(args, device)
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience, min_lr=args.min_lr)
    prompt_request = build_prompt_request(args)
    writer = None if args.disable_tensorboard else SummaryWriter(str(args.output_dir / "tensorboard"))

    start_epoch = 0
    best_val_iou = float("-inf")
    epochs_without_improvement = 0
    history = []
    checkpoint_dir = args.output_dir / "checkpoints"

    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if "optimizer_state" not in checkpoint:
            raise ValueError("Resume 必须使用包含训练状态的 last.pt")
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_iou = float(checkpoint["best_val_iou"])
        history = list(checkpoint["history"])
        logging.info("Resume | checkpoint=%s | next_epoch=%d/%d", args.resume_checkpoint, start_epoch + 1, args.epochs)

    for epoch in range(start_epoch, args.epochs):
        logging.info("--- Epoch %d/%d ---", epoch + 1, args.epochs)
        train_metrics = run_training_epoch(
            model=model, loss_fn=loss_fn, loader=train_loader, optimizer=optimizer,
            scaler=scaler, device=device, args=args, epoch=epoch,
            prompt_request=prompt_request, tensorboard_writer=writer,
        )
        val_metrics = run_validation_epoch(
            model=model, loss_fn=loss_fn, loader=val_loader, device=device,
            args=args, epoch=epoch, prompt_request=prompt_request,
            metric_start_frame=args.metric_start_frame,
        )

        overall_val = val_metrics["overall"]
        scheduler.step(overall_val["loss"])
        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = overall_val["iou"] > best_val_iou + args.early_stop_min_delta
        if improved:
            best_val_iou = float(overall_val["iou"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_metrics = {
            "epoch": float(epoch + 1),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "validation": val_metrics,
        }
        history.append(epoch_metrics)
        logging.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | val_iou=%.4f | val_dice=%.4f | lr=%.2e",
            epoch + 1, train_metrics["loss"], overall_val["loss"], overall_val["iou"], overall_val["dice"], current_lr,
        )

        save_checkpoint(
            checkpoint_dir / "last.pt", model, optimizer, scaler, scheduler, epoch,
            best_val_iou, args, epoch_metrics, history, include_optimizer_state=True,
        )
        if improved:
            save_checkpoint(
                checkpoint_dir / "best.pt", model, optimizer, scaler, scheduler, epoch,
                best_val_iou, args, epoch_metrics, history, include_optimizer_state=False,
            )
            logging.info("New best checkpoint | epoch=%d | val_iou=%.4f", epoch + 1, best_val_iou)

        dump_json({"best_val_iou": best_val_iou, "epochs": history}, args.output_dir / "metrics.json")
        if not args.skip_visualizations:
            curves = [{
                "epoch": row["epoch"], "train_loss": row["train_loss"],
                "val_loss": row["validation"]["overall"]["loss"],
                "val_iou": row["validation"]["overall"]["iou"],
                "val_dice": row["validation"]["overall"]["dice"],
            } for row in history]
            save_training_curves(curves, args.output_dir / "visualizations/training_curves.png")

        if writer is not None:
            for name, value in train_metrics.items():
                writer.add_scalar(f"epoch_train/{name}", value, epoch + 1)
            for group, metrics in val_metrics.items():
                for name, value in metrics.items():
                    if value is not None:
                        writer.add_scalar(f"validation/{group}/{name}", value, epoch + 1)
            writer.flush()

        if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
            logging.info("Early stopping after %d epochs without improvement", epochs_without_improvement)
            break
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
