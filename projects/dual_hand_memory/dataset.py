"""将单帧 Dataset 组织成严格连续的时序 Clip。"""

import torch
from torch.utils.data import Dataset, DataLoader

from projects.framewise_sam2_modified.dataset import DexYCBDataset, MultiServerDualHandDataset, CombinedStreamDataset

FrameDataset = (
    MultiServerDualHandDataset
    | DexYCBDataset
    | CombinedStreamDataset
)

class ConsecutiveClipDataset(Dataset):
    def __init__(
        self,
        frame_dataset: FrameDataset,
        clip_length: int = 2,
        clip_stride: int = 2,
    ):
        """把已有单帧 Dataset 包成 Clip"""

        if clip_length < 2:
            raise ValueError("clip_length 必须至少为 2")
        if clip_stride < 1:
            raise ValueError("clip_stride 必须大于 0")

        self.frame_dataset = frame_dataset
        self.clips = []

        for stream_id, stream in frame_dataset.streams.items():
            # sample_index:   这一帧在 Dataset 中的位置, 用于读取 dataset[index]
            # frame_number： 原视频文件名里的真实帧号, 用于判断原视频是否连续
            sample_indices = stream["sample_indices"]
            frame_numbers = stream["frame_numbers"]

            assert len(sample_indices) == len(frame_numbers)


            # 找 clip 类似一个算法题
            # 本质是“在有序帧号中寻找固定长度的连续窗口”，可以用单次扫描解决
            next_clip_start = 0
            for clip_end in range(len(frame_numbers)):
                # 遇到缺帧，从当前帧重新开始。首帧(clip_end=0)因为没有前驱要跳过
                if (
                    clip_end > 0
                    and frame_numbers[clip_end] != frame_numbers[clip_end - 1] + 1
                ):
                    next_clip_start = clip_end

                # 当前连续帧恰好可以组成一个 Clip。
                if clip_end - next_clip_start + 1 == clip_length:
                    end = clip_end + 1
                    self.clips.append({
                        "stream_id": stream_id,
                        "sample_indices": sample_indices[next_clip_start:end],
                        "frame_numbers": frame_numbers[next_clip_start:end],
                    })
                    next_clip_start += clip_stride

    def __getitem__(self, index):
        clip = self.clips[index]

        frames = [
            self.frame_dataset[sample_index]
            for sample_index in clip["sample_indices"]
        ]

        return {
            "image": torch.stack([frame["image"] for frame in frames]), # [T,3,H,W] 还没有 Batch collate
            "left_mask": torch.stack([frame["left_mask"] for frame in frames]),
            "right_mask": torch.stack([frame["right_mask"] for frame in frames]),

            "original_size": [frame["original_size"] for frame in frames],
            "original_image": [frame["original_image"] for frame in frames],
            "original_left_mask": [frame["original_left_mask"] for frame in frames],
            "original_right_mask": [frame["original_right_mask"] for frame in frames],

            "image_path": [frame["image_path"] for frame in frames],
            "mask_path": [frame["mask_path"] for frame in frames],
            "sample_id": [frame["sample_id"] for frame in frames],
            "dataset_name": frames[0]["dataset_name"],

            "stream_id": clip["stream_id"],
            "sample_indices": clip["sample_indices"],
            "frame_numbers": clip["frame_numbers"],
        }
    def __len__(self):
        return len(self.clips)

def collate_clip_batch(batch: list[dict]) -> dict:

    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "left_mask": torch.stack([item["left_mask"] for item in batch], dim=0),
        "right_mask": torch.stack([item["right_mask"] for item in batch], dim=0),

        "original_size": [item["original_size"] for item in batch],
        "original_image": [item["original_image"] for item in batch],
        "original_left_mask": [item["original_left_mask"] for item in batch],
        "original_right_mask": [item["original_right_mask"] for item in batch],

        "image_path": [item["image_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],
        "sample_id": [item["sample_id"] for item in batch],
        "dataset_name": [item["dataset_name"] for item in batch],

        "stream_id": [item["stream_id"] for item in batch],
        "sample_indices": [item["sample_indices"] for item in batch],
        "frame_numbers": [item["frame_numbers"] for item in batch],
    }

def build_dataloaders(args, device):
    train_frame_datasets = []
    val_frame_datasets = []

    if args.dataset_mode in {"multiserver", "mixed"}:
        train_frame_datasets.append(MultiServerDualHandDataset(
            dataset_root=args.dataset_root, split="train",
            test_seq_count=args.test_seq_count, image_size=args.image_size,
            use_augmentation=False, dataset_names=args.dataset_names,
        ))
        val_frame_datasets.append(MultiServerDualHandDataset(
            dataset_root=args.dataset_root, split="val",
            test_seq_count=args.test_seq_count, image_size=args.image_size,
            use_augmentation=False, dataset_names=args.dataset_names,
        ))

    if args.dataset_mode in {"dexycb", "mixed"}:
        train_frame_datasets.append(DexYCBDataset(
            dataset_root=args.dex_ycb_root, split="train", setup="s0",
            image_size=args.image_size, use_augmentation=False,
        ))
        val_frame_datasets.append(DexYCBDataset(
            dataset_root=args.dex_ycb_root, split="val", setup="s0",
            image_size=args.image_size, use_augmentation=False,
        ))

    if args.dataset_mode == "mixed":
        train_frame_dataset = CombinedStreamDataset(train_frame_datasets)
        val_frame_dataset = CombinedStreamDataset(val_frame_datasets)
    else:
        train_frame_dataset = train_frame_datasets[0]
        val_frame_dataset = val_frame_datasets[0]

    train_dataset = ConsecutiveClipDataset(
        train_frame_dataset,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
    )
    val_dataset = ConsecutiveClipDataset(
        val_frame_dataset,
        clip_length=args.clip_length,
        clip_stride=args.clip_stride,
    )

    if len(train_dataset) == 0:
        raise ValueError("训练集没有生成任何连续 Clip")
    if len(val_dataset) == 0:
        raise ValueError("验证集没有生成任何连续 Clip")

    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_clip_batch,
    }
    if args.num_workers > 0:
        loader_kwargs.update({
            "persistent_workers": True,
            "prefetch_factor": args.prefetch_factor,
        })

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, drop_last=True, **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size,
        shuffle=False, drop_last=False, **loader_kwargs,
    )

    return train_loader, val_loader