"""Framewise、Memory 与 MultiView 共用的数据参数和验证 Clip。"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from projects.dual_hand_memory.dataset import ConsecutiveClipDataset, collate_clip_batch
from projects.dual_hand_multiview.dataset import MultiViewConsecutiveClipDataset, collate_multiview_clip_batch
from projects.framewise_sam2_modified.dataset import DexYCBDataset, MultiServerDualHandDataset, CombinedStreamDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SAM_CHECKPOINT = REPO_ROOT / "checkpoints/sam2.1_hiera_tiny.pt"
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = "xingyi_4-5090_oak150-100output", "wuwen_4-5090_release-0623-compressed", "tencent_4-5090_7.5"

def add_dataloader_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Dataloader")
    group.add_argument("--clip-length", type=int, default=8)
    group.add_argument("--val-clip-stride", type=int, default=8)
    group.add_argument("--val-batch-size", type=int, default=1)
    group.add_argument("--num-workers", type=int, default=4)
    group.add_argument("--prefetch-factor", type=int, default=2)
    group.add_argument("--log-interval", type=int, default=50)
    group.add_argument("--save-visualizations", action="store_true")

    return parser

def add_dataset_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Dataset")
    group.add_argument("--dataset", dest="dataset_mode", choices=("multiserver", "dexycb", "mixed"), default="mixed")
    group.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    group.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    group.add_argument("--test-seq-count", type=int, default=3)
    group.add_argument("--dex-ycb-root", type=Path)
    group.add_argument(
        "--num-views",
        type=int,
        default=2,
        help="MultiView 每个 Clip 中对齐的视角数",
    )

    return parser

def add_runtime_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Runtime")

    group.add_argument("--output-dir", type=Path, required=True)
    group.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    group.add_argument("--seed", type=int, default=42)
    group.add_argument("--disable-tf32", action="store_true")
    group.add_argument("--image-size", type=int, default=768)
    group.add_argument(
        "--views-per-encode",
        type=int,
        default=1,
        help="MultiView 每次送入 Image Encoder 的视角数",
    )
    group.add_argument("--channels-last", action="store_true")

    return parser

def add_model_arguments(parser):
    checkpoint = parser.add_argument_group("Checkpoint")
    source = checkpoint.add_mutually_exclusive_group()
    source.add_argument("--sam-checkpoint", type=Path)
    source.add_argument("--framewise-checkpoint", type=Path)
    source.add_argument("--memory-checkpoint", type=Path)
    source.add_argument("--multiview-checkpoint", type=Path)

    # MultiView 从 SAM / Memory 权重初始化时所需的结构参数。
    structure = parser.add_argument_group("MultiView model structure")
    structure.add_argument("--num-latents", type=int, default=144)
    structure.add_argument("--num-aggregator-layers", type=int, default=2)
    structure.add_argument("--num-distributor-layers", type=int, default=1)
    return parser

def add_loss_arguments(parser):
    group = parser.add_argument_group("Loss")

    # Framewise
    group.add_argument("--bce-weight", type=float, default=1.0)
    group.add_argument("--dice-weight", type=float, default=1.0)
    group.add_argument("--iou-weight", type=float, default=0.1)
    group.add_argument("--object-score-weight", type=float, default=1.0)

    # Memory
    group.add_argument("--mask-loss-weight", type=float, default=20.0)
    group.add_argument("--dice-loss-weight", type=float, default=1.0)
    group.add_argument("--iou-loss-weight", type=float, default=1.0)
    group.add_argument("--class-loss-weight", type=float, default=1.0)

    return parser

def add_prompt_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Prompt")
    group.add_argument("--prompt-mode", choices=("auto", "point", "mask"), default="point")
    group.add_argument("--correction-frame-indices", type=int, nargs="*", default=[])
    group.add_argument("--num-correction-pt-per-frame", type=int, default=0)
    group.add_argument("--add-all-frames-to-correct-as-cond", action=argparse.BooleanOptionalAction, default=True)
    return parser

def add_evaluation_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Evaluation")
    group.add_argument("--metric-start-frame", type=int, default=0)
    group.add_argument("--multimask-output", action=argparse.BooleanOptionalAction, default=False)
    return parser

def parse_args() -> argparse.Namespace:
    """解析三种模型共用的验证数据与运行参数。"""
    parser = argparse.ArgumentParser(
        description="Evaluate Framewise, Memory, or MultiView models on shared validation clips."
    )

    add_runtime_arguments(parser)
    add_model_arguments(parser)
    add_dataset_arguments(parser)
    add_dataloader_arguments(parser)
    add_loss_arguments(parser)
    add_prompt_arguments(parser)
    add_evaluation_arguments(parser)

    args = parser.parse_args()
    args.skip_visualizations = not args.save_visualizations


    if args.dataset_mode in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error("--dataset dexycb/mixed 需要提供 --dex-ycb-root")

    if all(
        checkpoint is None
        for checkpoint in (
            args.sam_checkpoint,
            args.framewise_checkpoint,
            args.memory_checkpoint,
            args.multiview_checkpoint,
        )
    ):
        args.sam_checkpoint = DEFAULT_SAM_CHECKPOINT

    if args.clip_length < 2:
        parser.error("--clip-length 必须至少为 2")

    if args.val_clip_stride < args.clip_length:
        parser.error("--val-clip-stride 不能小于 --clip-length")

    if args.num_views < 2:
        parser.error("--num-views 必须至少为 2")

    if args.views_per_encode < 1:
        parser.error("--views-per-encode 必须大于 0")

    if args.num_latents < 1 or args.num_aggregator_layers < 1 or args.num_distributor_layers < 1:
        parser.error("MultiView 结构参数必须大于 0")

    if args.num_correction_pt_per_frame < 0:
        parser.error("--num-correction-pt-per-frame 不能小于 0")

    has_frames = bool(args.correction_frame_indices)
    has_points = args.num_correction_pt_per_frame > 0
    if has_frames != has_points:
        parser.error("纠错帧和每帧纠错点数必须同时设置")

    if not 0 <= args.metric_start_frame < args.clip_length:
        parser.error("--metric-start-frame 必须位于 Clip 范围内")

    for frame_idx in args.correction_frame_indices:
        if not 0 <= frame_idx < args.clip_length:
            parser.error("纠错帧下标超出 Clip 范围")

    if args.correction_frame_indices and args.num_correction_pt_per_frame == 0:
        parser.error("指定纠错帧时，纠错点数量必须大于 0")

    return args


def _build_validation_frame_dataset(args: argparse.Namespace):
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

    return CombinedStreamDataset(frame_datasets) if len(frame_datasets) > 1 else frame_datasets[0]


def _build_validation_loader(args, device, dataset, collate_fn):
    loader_kwargs = {
        "batch_size": args.val_batch_size, "shuffle": False, "drop_last": False,
        "num_workers": args.num_workers, "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
    }
    if args.num_workers > 0:
        loader_kwargs.update({"persistent_workers": True, "prefetch_factor": args.prefetch_factor})
    return DataLoader(dataset, **loader_kwargs)


def build_zero_shot_loader(args: argparse.Namespace, device: torch.device, collate_fn=collate_clip_batch) -> DataLoader:
    """加载共享验证集中的单视角连续 Clip。"""
    dataset = ConsecutiveClipDataset(
        _build_validation_frame_dataset(args),
        clip_length=args.clip_length,
        clip_stride=args.val_clip_stride,
    )
    if len(dataset) == 0:
        raise ValueError("验证集没有生成任何连续 Clip")
    return _build_validation_loader(args, device, dataset, collate_fn)


def build_multiview_zero_shot_loader(args: argparse.Namespace, device: torch.device) -> DataLoader:
    """加载共享验证集中对齐的多视角连续 Clip。"""
    dataset = MultiViewConsecutiveClipDataset(
        _build_validation_frame_dataset(args),
        num_views=args.num_views,
        clip_length=args.clip_length,
        clip_stride=args.val_clip_stride,
        use_augmentation=False,
    )
    if len(dataset) == 0:
        raise ValueError("验证集没有生成多视角连续 Clip")
    return _build_validation_loader(
        args,
        device,
        dataset,
        collate_multiview_clip_batch,
    )
