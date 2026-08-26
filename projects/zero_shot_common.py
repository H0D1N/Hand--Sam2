"""Framewise 与 Memory zero-shot 共用的数据参数和验证 Clip。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from projects.dual_hand_memory.dataset import ConsecutiveClipDataset, collate_clip_batch
from projects.framewise_sam2_modified.dataset import DexYCBDataset, MultiServerDualHandDataset, CombinedStreamDataset


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAM_CHECKPOINT = REPO_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = "xingyi_4-5090_oak150-100output", "wuwen_4-5090_release-0623-compressed", "tencent_4-5090_7.5"


def parse_args() -> argparse.Namespace:
    """解析两种模型完全相同的 zero-shot 数据与运行参数。"""
    parser = argparse.ArgumentParser(description="Compare Framewise and Memory zero-shot baselines on shared clips.")

    # 公共参数
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sam-checkpoint", type=Path, default=DEFAULT_SAM_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--image-size", type=int, default=768)

    # Dataset
    parser.add_argument("--dataset", dest="dataset_mode", choices=("multiserver", "dexycb", "mixed"), default="mixed")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--test-seq-count", type=int, default=3)
    parser.add_argument("--dex-ycb-root", type=Path)

    # Dataloader
    parser.add_argument("--clip-length", type=int, default=8)
    parser.add_argument("--val-clip-stride", type=int, default=8)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--log-interval", type=int, default=50)

    # framewise 专有参数
    parser.add_argument("--channels-last", action="store_true")

    args = parser.parse_args()
    if args.dataset_mode in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error("--dataset dexycb/mixed 需要提供 --dex-ycb-root")
    if args.val_clip_stride < args.clip_length:
        parser.error("--val-clip-stride 不能小于 --clip-length")

    # 公共固定参数
    args.skip_visualizations = True

    # Framewise 固定参数
    args.multimask_output = False
    args.use_point_prompt = True
    args.bce_weight, args.dice_weight = 1.0, 1.0
    args.iou_weight, args.object_score_weight = 0.1, 1.0

    # Dual-hand Memory 固定参数
    args.prompt_mode = "auto"
    args.mask_loss_weight, args.dice_loss_weight = 20.0, 1.0
    args.iou_loss_weight, args.class_loss_weight = 1.0, 1.0
    return args


def build_zero_shot_loader(args: argparse.Namespace, device: torch.device, collate_fn=collate_clip_batch) -> DataLoader:
    """只加载共享的验证集，保证两种模型评测完全相同的连续帧。"""
    frame_datasets = []
    if args.dataset_mode in {"multiserver", "mixed"}:
        frame_datasets.append(MultiServerDualHandDataset(
            dataset_root=args.dataset_root, split="val", test_seq_count=args.test_seq_count,
            image_size=args.image_size, use_augmentation=False, dataset_names=args.dataset_names,
        ))
    if args.dataset_mode in {"dexycb", "mixed"}:
        frame_datasets.append(DexYCBDataset(
            dataset_root=args.dex_ycb_root, split="val", setup="s0",
            image_size=args.image_size, use_augmentation=False,
        ))

    frame_dataset = CombinedStreamDataset(frame_datasets) if args.dataset_mode == "mixed" else frame_datasets[0]
    dataset = ConsecutiveClipDataset(frame_dataset, clip_length=args.clip_length, clip_stride=args.val_clip_stride)
    if len(dataset) == 0:
        raise ValueError("验证集没有生成任何连续 Clip")

    loader_kwargs = {
        "batch_size": args.val_batch_size, "shuffle": False, "drop_last": False,
        "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    return DataLoader(dataset, **loader_kwargs)
