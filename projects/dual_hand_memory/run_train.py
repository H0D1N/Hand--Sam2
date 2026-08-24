"""训练双手独立 Memory 的 SAM2 模型。"""

import argparse
import logging
from pathlib import Path

import torch
from torch.optim import Adam, AdamW, RAdam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter

from .builder import build_sam2_dual_hand_memory_tiny, configure_memory_training
from .dataset import build_dataloaders
from .trainer import run_training_epoch, run_validation_epoch
from .losses import DualHandMemoryLoss
from projects.framewise_sam2_modified.utils import (
    configure_runtime, dump_json, save_checkpoint,
    save_training_curves, set_seed,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM_CHECKPOINT = REPO_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = "xingyi_4-5090_oak150-100output", "wuwen_4-5090_release-0623-compressed", "tencent_4-5090_7.5"


def parse_args() -> argparse.Namespace:
    """
    解析双手 Memory baseline 的训练参数。

    默认 auto 提示策略参考 SAM2：
    - 训练时约 50% mask、25% box、25% point；
    - 随机选择最多 2 个初始条件帧和 2 个纠错帧；
    - 纠错帧加入条件 Memory，每帧追加 7 个纠错点；
    - 验证时只在第 0 帧提供 GT mask，不模拟纠错。

    以下 SAM2 默认值固定在 Builder 中，不作为命令行参数：
    - point 输入概率 0.5，point 分支使用 box 的概率 0.5；
    - 从 GT 区域采样纠错点的概率 0.1；
    - 随机选择初始条件帧和纠错帧；
    - num_maskmem=7，Memory 网络结构保持官方配置。
    """
    parser = argparse.ArgumentParser(description="Train SAM2 with independent left/right hand Memory.")

    # 实验与运行环境
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--disable-tf32", action="store_true")

    # 模型初始化
    init_group = parser.add_mutually_exclusive_group()
    init_group.add_argument("--sam-checkpoint", type=Path)
    init_group.add_argument("--framewise-checkpoint", type=Path)

    # Finetune
    parser.add_argument("--finetune-mode", choices=("memory-only", "decoder-memory"), default="decoder-memory")

    # Prompt
    parser.add_argument("--image-size", type=int, default=768)

    # Adapter
    parser.add_argument("--use-image-adapter", action="store_true")
    parser.add_argument("--use-decoder-adapter", action="store_true")
    parser.add_argument("--adapter-dim", type=int, default=64)
    parser.add_argument("--adapter-dropout", type=float, default=0.1)
    parser.add_argument("--adapter-init-scale", type=float, default=1e-3)

    # Prompt
    parser.add_argument("--prompt-mode", choices=("point", "mask", "auto"), default="auto")
    parser.add_argument("--num-init-cond-frames-for-train", type=int, default=2)
    parser.add_argument("--num-frames-to-correct-for-train", type=int, default=2)
    parser.add_argument("--add-all-frames-to-correct-as-cond", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-correction-pt-per-frame", type=int, default=7)

    # Dataset
    dataset_group = parser.add_mutually_exclusive_group()
    dataset_group.add_argument("--multiserver", dest="dataset_mode", action="store_const", const="multiserver", help="只使用 MultiServer")
    dataset_group.add_argument("--dexycb", dest="dataset_mode", action="store_const", const="dexycb", help="只使用 DexYCB")
    dataset_group.add_argument("--mixed", dest="dataset_mode", action="store_const", const="mixed", help="同时使用 MultiServer 和 DexYCB")

    parser.set_defaults(dataset_mode="mixed")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--test-seq-count", type=int, default=3)
    parser.add_argument("--dex-ycb-root", type=Path)

    # Clip 与 DataLoader
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--clip-stride", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)

    # Loss
    parser.add_argument("--mask-loss-weight", type=float, default=20.0)
    parser.add_argument("--dice-loss-weight", type=float, default=1.0)
    parser.add_argument("--iou-loss-weight", type=float, default=1.0)
    parser.add_argument("--class-loss-weight", type=float, default=1.0)

    # Optimizer 与 Scheduler
    parser.add_argument("--optimizer", choices=("adam", "adamw", "radam"), default="adamw")
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience", type=int, default=2)
    parser.add_argument("--min-lr", type=float, default=1e-6)

    # 日志与停止条件
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-delta", type=float, default=1e-4)
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--disable-tensorboard", action="store_true")
    parser.add_argument("--tensorboard-log-interval", type=int, default=10)
    parser.add_argument("--debug-high-class-loss", action="store_true")

    args = parser.parse_args()

    if args.sam_checkpoint is None and args.framewise_checkpoint is None:
        args.sam_checkpoint = DEFAULT_SAM_CHECKPOINT

    if args.dataset_mode in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error("--dexycb/--mixed 需要提供 --dex-ycb-root")
    if args.epochs < 1:
        parser.error("--epochs 必须大于 0")
    if args.grad_accum_steps < 1:
        parser.error("--grad-accum-steps 必须大于 0")
    if args.max_grad_norm <= 0:
        parser.error("--max-grad-norm 必须大于 0")
    if args.log_interval < 1:
        parser.error("--log-interval 必须大于 0")
    if args.tensorboard_log_interval < 1:
        parser.error("--tensorboard-log-interval 必须大于 0")
    if args.num_correction_pt_per_frame < 0:
        parser.error("--num-correction-pt-per-frame 不能小于 0")
    if not (
        1
        <= args.num_init_cond_frames_for_train
        <= args.num_frames_to_correct_for_train
        <= args.clip_length
    ):
        parser.error("必须满足 1 <= 初始条件帧数 <= 纠错帧数 <= clip-length")

    return args

def build_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    trainable_parameters = [p for p in model.parameters() if p.requires_grad]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters found.")

    optimizer_map = {"adam": Adam, "adamw": AdamW, "radam": RAdam}
    optimizer_cls = optimizer_map[args.optimizer]
    optimizer_kwargs = {"lr": args.lr, "weight_decay": args.weight_decay}

    if args.device.startswith("cuda") and args.optimizer in {"adam", "adamw"}:
        optimizer_kwargs["fused"] = True

    return optimizer_cls(trainable_parameters, **optimizer_kwargs)

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

    # 模型
    model = build_sam2_dual_hand_memory_tiny(
        sam_checkpoint=args.sam_checkpoint,
        framewise_checkpoint=args.framewise_checkpoint,
        device=device,
        mode="train",
        image_size=args.image_size,
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
    configure_memory_training(model, args.finetune_mode)

    loss_fn = DualHandMemoryLoss(
        mask_loss_weight=args.mask_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        iou_loss_weight=args.iou_loss_weight,
        class_loss_weight=args.class_loss_weight,
    ).to(device)

    # 数据和优化器
    train_loader, val_loader = build_dataloaders(args, device)

    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and args.amp)
    optimizer = build_optimizer(args, model)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=args.lr_factor, patience=args.lr_patience,
                                  min_lr=args.min_lr)

    writer = None if args.disable_tensorboard else SummaryWriter(
        log_dir=str(args.output_dir / "tensorboard")
    )

    best_val_iou = float("-inf")
    epochs_without_improvement = 0
    history = []
    checkpoint_dir = args.output_dir / "checkpoints"

    for epoch in range(args.epochs):
        logging.info("--- Epoch %d/%d ---", epoch + 1, args.epochs)

        train_metrics = run_training_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            scaler=scaler, device=device, args=args, epoch=epoch,
            tensorboard_writer=writer, loss_fn=loss_fn,
        )
        val_metrics = run_validation_epoch(
            model=model, loader=val_loader, device=device,
            args=args, epoch=epoch, loss_fn=loss_fn,
        )
        scheduler.step(val_metrics["loss"])

        current_lr = float(optimizer.param_groups[0]["lr"])
        improved = (
            val_metrics["iou"]
            > best_val_iou + args.early_stop_min_delta
        )

        if improved:
            best_val_iou = float(val_metrics["iou"])
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_metrics = {
            "epoch": float(epoch + 1),
            "lr": current_lr,
            "train_loss": float(train_metrics["loss"]),
            "val_loss": float(val_metrics["loss"]),
            "val_iou": float(val_metrics["iou"]),
            "val_dice": float(val_metrics["dice"]),
            "val_object_accuracy": float(val_metrics["object_accuracy"]),
            "val_object_precision": float(val_metrics["object_precision"]),
            "val_object_recall": float(val_metrics["object_recall"]),
            "val_object_f1": float(val_metrics["object_f1"]),
        }
        history.append(epoch_metrics)

        logging.info(
            "Epoch %d | train_loss=%.4f | val_loss=%.4f | "
            "val_iou=%.4f | val_dice=%.4f | lr=%.2e",
            epoch + 1, train_metrics["loss"], val_metrics["loss"],
            val_metrics["iou"], val_metrics["dice"], current_lr,
        )

        save_checkpoint(
            output_path=checkpoint_dir / "last.pt",
            model=model, optimizer=optimizer, scaler=scaler,
            scheduler=scheduler, epoch=epoch,
            best_val_iou=best_val_iou, args=args,
            metrics=epoch_metrics, history=history,
            include_optimizer_state=True,
        )

        if improved:
            save_checkpoint(
                output_path=checkpoint_dir / "best.pt",
                model=model, optimizer=optimizer, scaler=scaler,
                scheduler=scheduler, epoch=epoch,
                best_val_iou=best_val_iou, args=args,
                metrics=epoch_metrics, history=history,
                include_optimizer_state=False,
            )
            logging.info(
                "New best checkpoint | epoch=%d | val_iou=%.4f",
                epoch + 1, best_val_iou,
            )

        dump_json(
            {"best_val_iou": best_val_iou, "epochs": history},
            args.output_dir / "metrics.json",
        )

        if not args.skip_visualizations:
            save_training_curves(
                history,
                args.output_dir / "visualizations" / "training_curves.png",
            )

        # val 的tensorboard 写入
        if writer is not None:
            for name, value in train_metrics.items():
                writer.add_scalar(
                    f"epoch_train/{name}",
                    value,
                    epoch + 1,
                )

            for name, value in val_metrics.items():
                writer.add_scalar(
                    f"validation/{name}",
                    value,
                    epoch + 1,
                )

            writer.flush()

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            logging.info(
                "Early stopping after %d epochs without improvement",
                epochs_without_improvement,
            )
            break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
