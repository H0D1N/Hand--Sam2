"""复用现有 Dataset 和 stream 索引，不切分视频。"""

import argparse

import torch
from torch.utils.data import DataLoader, Subset

from projects.framewise_sam2_modified.dataset import (
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
    collate_batch,
)


def build_frame_dataset(args: argparse.Namespace):
    datasets = []
    if args.dataset in {"multiserver", "mixed"}:
        datasets.append(MultiServerDualHandDataset(
            dataset_root=args.dataset_root,
            split="val",
            test_seq_count=args.test_seq_count,
            image_size=args.image_size,
            use_augmentation=False,
            dataset_names=args.dataset_names,
        ))
    if args.dataset in {"dexycb", "mixed"}:
        datasets.append(DexYCBDataset(
            dataset_root=args.dex_ycb_root,
            split="val",
            setup=args.dex_ycb_setup,
            image_size=args.image_size,
            use_augmentation=False,
        ))
    return datasets[0] if len(datasets) == 1 else CombinedStreamDataset(datasets)


def build_loader(dataset, args: argparse.Namespace, sample_indices=None) -> DataLoader:
    if sample_indices is not None:
        dataset = Subset(dataset, sample_indices)
    if args.model == "memory" and args.batch_size != 1:
        raise ValueError("memory 推理仅支持 batch_size=1")

    loader_args = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
        collate_fn=collate_batch,
    )
    if args.num_workers > 0:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**loader_args)
