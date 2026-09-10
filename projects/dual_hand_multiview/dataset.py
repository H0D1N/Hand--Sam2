"""将单视角数据组织成同步的多视角时序 Clip。"""

from itertools import combinations

import torch
from torch.utils.data import DataLoader, Dataset

from projects.framewise_sam2_modified.dataset import (
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
    augment_dual_hand_clip,
)


class MultiViewConsecutiveClipDataset(Dataset):
    """
    将单帧 Dataset 包装成多视角连续 Clip。

    输出：
        image:      [V, T, 3, H, W]
        left_mask:  [V, T, 1, H, W]
        right_mask: [V, T, 1, H, W]
    """

    def __init__(
        self,
        frame_dataset,
        num_views=2,
        clip_length=8,
        clip_stride=8,
        use_augmentation=False,
    ):
        if num_views < 2:
            raise ValueError("num_views 必须至少为 2")
        if clip_length < 2:
            raise ValueError("clip_length 必须至少为 2")
        if clip_stride < 1:
            raise ValueError("clip_stride 必须大于 0")

        self.frame_dataset = frame_dataset
        self.use_augmentation = use_augmentation
        self.clips = []

        # 1. 按 sequence 聚合各个 view
        sequences = {}

        for stream_id, stream in frame_dataset.streams.items():
            sequence_id, view_name = stream_id.rsplit("/", 1)
            sequences.setdefault(sequence_id, {})[view_name] = stream

        # 2. 对齐每组视角的共同帧
        for sequence_id, streams in sequences.items():
            view_names = sorted(streams)

            if len(view_names) < num_views:
                continue

            for selected_views in combinations(view_names, num_views):
                frame_maps = {
                    view: dict(zip(streams[view]["frame_numbers"], streams[view]["sample_indices"]))
                    for view in selected_views
                }
                common_frames = sorted(set.intersection(*(set(frame_map) for frame_map in frame_maps.values())))

                # 3. 从共同帧中寻找连续 T 帧
                clip_start = 0

                for clip_end in range(len(common_frames)):
                    if clip_end > 0 and common_frames[clip_end] != common_frames[clip_end - 1] + 1:
                        clip_start = clip_end

                    if clip_end - clip_start + 1 != clip_length:
                        continue

                    frame_numbers = common_frames[clip_start:clip_end + 1]
                    sample_indices = [
                        [frame_maps[view][frame] for frame in frame_numbers]
                        for view in selected_views
                    ]

                    self.clips.append({
                        "dataset_name": streams[selected_views[0]]["dataset_name"],
                        "sequence_id": sequence_id,
                        "view_names": selected_views,
                        "frame_numbers": frame_numbers,
                        "sample_indices": sample_indices,
                    })
                    clip_start += clip_stride

    def __getitem__(self, index):
        clip = self.clips[index]
        views = [
            [self.frame_dataset[i] for i in indices]
            for indices in clip["sample_indices"]
        ]

        images = torch.stack([
            torch.stack([frame["image"] for frame in frames])
            for frames in views
        ])
        left_masks = torch.stack([
            torch.stack([frame["left_mask"] for frame in frames])
            for frames in views
        ])
        right_masks = torch.stack([
            torch.stack([frame["right_mask"] for frame in frames])
            for frames in views
        ])

        if self.use_augmentation:
            images, left_masks, right_masks = augment_dual_hand_clip(
                images, left_masks, right_masks
            )

        return {
            "image": images,
            "left_mask": left_masks,
            "right_mask": right_masks,

            "original_size": [[frame["original_size"] for frame in frames] for frames in views],
            "original_image": [[frame["original_image"] for frame in frames] for frames in views],
            "original_left_mask": [[frame["original_left_mask"] for frame in frames] for frames in views],
            "original_right_mask": [[frame["original_right_mask"] for frame in frames] for frames in views],

            "image_path": [[frame["image_path"] for frame in frames] for frames in views],
            "mask_path": [[frame["mask_path"] for frame in frames] for frames in views],
            "sample_id": [[frame["sample_id"] for frame in frames] for frames in views],

            "dataset_name": clip["dataset_name"],
            "sequence_id": clip["sequence_id"],
            "view_names": clip["view_names"],
            "frame_numbers": clip["frame_numbers"],
            "sample_indices": clip["sample_indices"],
        }

    def __len__(self):
        return len(self.clips)


def collate_multiview_clip_batch(batch):
    """组成 [B,V,T,...] batch，并保留不同尺寸的验证原图。"""

    return {
        "image": torch.stack([item["image"] for item in batch]),
        "left_mask": torch.stack([item["left_mask"] for item in batch]),
        "right_mask": torch.stack([item["right_mask"] for item in batch]),

        "original_size": [item["original_size"] for item in batch],
        "original_image": [item["original_image"] for item in batch],
        "original_left_mask": [item["original_left_mask"] for item in batch],
        "original_right_mask": [item["original_right_mask"] for item in batch],

        "image_path": [item["image_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],

        "dataset_name": [item["dataset_name"] for item in batch],
        "sequence_id": [item["sequence_id"] for item in batch],
        "view_names": [item["view_names"] for item in batch],
        "frame_numbers": [item["frame_numbers"] for item in batch],
        "sample_indices": [item["sample_indices"] for item in batch],
    }


def build_dataloaders(args, device):
    """创建 MultiServer、DexYCB 或 mixed 多视角 DataLoader。"""

    frame_datasets = {}

    for split in ("train", "val"):
        datasets = []

        if args.dataset_mode in {"multiserver", "mixed"}:
            datasets.append(MultiServerDualHandDataset(
                dataset_root=args.dataset_root,
                split=split,
                test_seq_count=args.test_seq_count,
                image_size=args.image_size,
                use_augmentation=False,
                dataset_names=args.dataset_names,
            ))

        if args.dataset_mode in {"dexycb", "mixed"}:
            datasets.append(DexYCBDataset(
                dataset_root=args.dex_ycb_root,
                split=split,
                setup="s0",
                image_size=args.image_size,
                use_augmentation=False,
            ))

        if not datasets:
            raise ValueError(f"不支持的 dataset_mode: {args.dataset_mode}")

        frame_datasets[split] = CombinedStreamDataset(datasets) if len(datasets) > 1 else datasets[0]

    train_dataset = MultiViewConsecutiveClipDataset(
        frame_datasets["train"],
        num_views=args.num_views,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
        use_augmentation=not args.disable_augmentation,
    )
    val_dataset = MultiViewConsecutiveClipDataset(
        frame_datasets["val"],
        num_views=args.num_views,
        clip_length=args.clip_length,
        clip_stride=args.val_clip_stride,
        use_augmentation=False,
    )

    if len(train_dataset) == 0:
        raise ValueError("训练集没有生成多视角连续 Clip")
    if len(val_dataset) == 0:
        raise ValueError("验证集没有生成多视角连续 Clip")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_multiview_clip_batch,
    }

    if args.num_workers > 0:
        loader_kwargs.update(
            persistent_workers=True,
            prefetch_factor=args.prefetch_factor,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        drop_last=False,
        **loader_kwargs,
    )
    return train_loader, val_loader
