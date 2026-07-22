"""多数据源左右手单帧 Dataset 与 batch collate。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import logging
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from .frame_sampler import FramesPerSecondSampler

try:
    import albumentations as A
except ModuleNotFoundError:
    A = None


SAM2_MEAN = [0.485, 0.456, 0.406]
SAM2_STD = [0.229, 0.224, 0.225]


def build_train_augmentation(image_size: int):
    """创建同时作用于图片、左手 mask 和右手 mask 的数据增强。"""
    if A is None:
        raise ModuleNotFoundError(
            "使用数据增强需要安装 albumentations"
        )

    return A.Compose(
        [
            A.RandomResizedCrop(
                size=(image_size, image_size),
                scale=(0.8, 1.0),
                ratio=(0.9, 1.1),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                p=0.6,
            ),
            A.RandomRotate90(p=0.5),
            A.Affine(
                scale=(0.95, 1.05),
                translate_percent=(-0.05, 0.05),
                rotate=(-20, 20),
                shear=(-8, 8),
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.4,
            ),
            A.RandomBrightnessContrast(p=0.3),
            A.GaussNoise(
                std_range=(0.01, 0.05),
                p=0.15,
            ),
            A.Resize(
                height=image_size,
                width=image_size,
                interpolation=cv2.INTER_LINEAR,
                mask_interpolation=cv2.INTER_NEAREST,
            ),
        ],
        additional_targets={
            "right_mask": "mask",
        },
    )


class MultiServerDualHandDataset(Dataset):
    """读取完整下载到本地的多数据源左右手单帧数据。"""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "val",
        test_seq_count: int = 2,
        image_size: int = 1024,
        use_augmentation: bool = False,
        dataset_names: list[str] | None = None,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"split 必须是 'train' 或 'val'，当前是 {split!r}")

        if test_seq_count <= 0:
            raise ValueError("test_seq_count 必须大于 0")

        if image_size <= 0:
            raise ValueError("image_size 必须大于 0")

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.streams = {}
        self.samples = []
        self.mask_values = {}

        self.augmentation = (
            build_train_augmentation(image_size)
            if use_augmentation and split == "train"
            else None
        )

        # pull_remote_dataset.py 会在缓存根目录生成 datasets.json。
        catalog_path = self.dataset_root / "datasets.json"

        if not catalog_path.is_file():
            raise FileNotFoundError(f"找不到本地数据集配置: {catalog_path}")

        with catalog_path.open("r", encoding="utf-8") as handle:
            catalog = json.load(handle)

        dataset_config_by_name = {
            config["dataset_name"]: config
            for config in catalog["datasets"]
        }

        if dataset_names is None:
            dataset_names = list(dataset_config_by_name)
        else:
            missing_names = [
                name
                for name in dataset_names
                if name not in dataset_config_by_name
            ]

            if missing_names:
                raise ValueError(f"datasets.json 中不存在: {missing_names}")

        # 3. 得到每个 dataset 对应的 ROM 列表和 cam 名称列表
        seq_dirs_list = []
        cam_names_list = []

        for dataset_name in dataset_names:
            dataset_config = dataset_config_by_name[dataset_name]

            self.mask_values[dataset_name] = {
                "left": dataset_config["mask_values"]["left"][0],
                "right": dataset_config["mask_values"]["right"][0],
            }

            # 获得序列列表
            local_dataset_root = self.dataset_root / "sources" / dataset_name

            if not local_dataset_root.is_dir():
                raise FileNotFoundError(f"本地数据集目录不存在: {local_dataset_root}")

            sequence_dirs = sorted([
                path for path in local_dataset_root.iterdir()
                if path.is_dir() and path.match(dataset_config["sequence_glob"])
            ])

            if len(sequence_dirs) <= test_seq_count:
                raise ValueError(
                    f"{dataset_name} 只有 {len(sequence_dirs)} 条序列，"
                    f"必须大于 test_seq_count={test_seq_count}"
                )

            if split == "train":
                current_sequence_dirs = sequence_dirs[:-test_seq_count]
            else:
                current_sequence_dirs = sequence_dirs[-test_seq_count:]

            # 获得视角列表
            view_glob = dataset_config["view_glob"]
            view_globs = view_glob if isinstance(view_glob, list) else [view_glob]
            first_sequence_dir = current_sequence_dirs[0]
            cam_names = sorted([
                path.name for path in first_sequence_dir.iterdir()
                if path.is_dir() and any(
                    path.match(pattern) for pattern in view_globs
                )
            ])

            seq_dirs_list.append(current_sequence_dirs)
            cam_names_list.append(cam_names)
            print(
                f"[{split.upper()}] {dataset_name}: "
                f"找到 {len(current_sequence_dirs)} 条序列，cam={cam_names}",
                flush=True,
            )

        # 4. 按 dataset、序列和 cam 收集同名的图片与 mask
        for dataset_index, dataset_name in enumerate(dataset_names):
            dataset_config = dataset_config_by_name[dataset_name]
            sample_count_before = len(self.samples)

            for sequence_dir in seq_dirs_list[dataset_index]:
                for cam_name in cam_names_list[dataset_index]:
                    image_dir = sequence_dir / dataset_config["rgb_dir"].format(
                        view=cam_name
                    )
                    mask_dir = sequence_dir / dataset_config["mask_dir"].format(
                        view=cam_name
                    )

                    if not image_dir.is_dir():
                        print("找不到图片文件夹:", image_dir)
                        continue

                    if not mask_dir.is_dir():
                        print("找不到 mask 文件夹:", mask_dir)
                        continue

                    image_paths = sorted(image_dir.glob("*.png"))

                    stream_id = (
                        f"{dataset_name}/"
                        f"{sequence_dir.name}/"
                        f"{cam_name}"
                    )
                    stream_sample_indices = []

                    for image_path in image_paths:
                        mask_path = mask_dir / image_path.name

                        if not mask_path.is_file():
                            continue

                        sample_index = len(self.samples)

                        sample_id = (
                            image_path
                            .relative_to(self.dataset_root / "sources")
                            .with_suffix("")
                            .as_posix()
                            .replace("/", "__")
                        )

                        self.samples.append(
                            {
                                "image_path": image_path,
                                "mask_path": mask_path,
                                "dataset_name": dataset_name,
                                "sample_id": sample_id,
                            }
                        )

                        stream_sample_indices.append(sample_index)

                    if stream_sample_indices:
                        self.streams[stream_id] = {
                            "dataset_name": dataset_name,
                            "fps": dataset_config.get("fps"),
                            "sample_indices": stream_sample_indices,
                        }

            dataset_sample_count = len(self.samples) - sample_count_before

            print(
                f"[{split.upper()}] {dataset_name}: "
                f"{dataset_sample_count} 对图片和 mask"
            )

        print(f"[{split.upper()}] 共加载 {len(self.samples)} 个样本")

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]

        image_path = sample["image_path"]
        mask_path = sample["mask_path"]

        image_pil = Image.open(image_path).convert("RGB")
        raw_mask_pil = Image.open(mask_path).convert("L")

        original_width, original_height = image_pil.size
        original_size = (original_height, original_width)

        image_np = np.array(image_pil)
        raw_mask_np = np.array(raw_mask_pil)

        dataset_name = sample["dataset_name"]

        left_mask_np = np.isin(
            raw_mask_np,
            self.mask_values[dataset_name]["left"],
        ).astype(np.uint8)

        right_mask_np = np.isin(
            raw_mask_np,
            self.mask_values[dataset_name]["right"],
        ).astype(np.uint8)

        # 如果 mask 和图片尺寸不同，先把 mask 对齐到原图尺寸。
        if left_mask_np.shape != image_np.shape[:2]:
            left_mask_np = cv2.resize(
                left_mask_np,
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

            right_mask_np = cv2.resize(
                right_mask_np,
                (image_np.shape[1], image_np.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )

        # 验证时保留原图和原始 mask，用来计算指标与可视化。
        original_image = (
            torch.from_numpy(image_np.copy()).permute(2, 0, 1)
            if self.split == "val" else None)

        original_left_mask = (
            torch.from_numpy(left_mask_np.astype(np.float32)).unsqueeze(0)
            if self.split == "val" else None)
        original_right_mask = (
            torch.from_numpy(right_mask_np.astype(np.float32)).unsqueeze(0)
            if self.split == "val" else None)

        if self.augmentation is not None:
            transformed = self.augmentation(
                image=image_np,
                mask=left_mask_np,
                right_mask=right_mask_np,
            )

            image_np = transformed["image"]
            left_mask_np = transformed["mask"]
            right_mask_np = transformed["right_mask"]
        else:
            image_np = cv2.resize(
                image_np,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR,
            )

            left_mask_np = cv2.resize(
                left_mask_np,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

            right_mask_np = cv2.resize(
                right_mask_np,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_NEAREST,
            )

        image_tensor = (torch.from_numpy(image_np.copy()).permute(2, 0, 1).float() / 255.0)
        image_tensor = TF.normalize(image_tensor, mean=SAM2_MEAN, std=SAM2_STD)

        left_mask_tensor = torch.from_numpy(left_mask_np.astype(np.float32)).unsqueeze(0)
        right_mask_tensor = torch.from_numpy(right_mask_np.astype(np.float32)).unsqueeze(0)

        # SAM 使用标签 2、3 表示 box 的左上角和右下角。
        bbox = torch.tensor(
            [
                [0.0, 0.0],
                [self.image_size - 1.0, self.image_size - 1.0],
            ],
            dtype=torch.float32,
        )

        box_labels = torch.tensor(
            [2, 3],
            dtype=torch.int64,
        )


        return {
            "image": image_tensor,
            "left_mask": left_mask_tensor,
            "right_mask": right_mask_tensor,
            "bbox": bbox,
            "box_labels": box_labels,
            "original_size": original_size,
            "original_image": original_image,
            "original_left_mask": original_left_mask,
            "original_right_mask": original_right_mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
            "sample_id": sample["sample_id"],
            "dataset_name": sample["dataset_name"],
        }

    def __len__(self) -> int:
        return len(self.samples)


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """组成 batch，同时把不同尺寸的验证原图保存在列表中。"""
    return {
        "image": torch.stack([item["image"] for item in batch], dim=0),
        "left_mask": torch.stack([item["left_mask"] for item in batch], dim=0),
        "right_mask": torch.stack([item["right_mask"] for item in batch], dim=0),

        "bbox": torch.stack([item["bbox"] for item in batch], dim=0),
        "box_labels": torch.stack([item["box_labels"] for item in batch], dim=0),

        "original_size": [item["original_size"] for item in batch],
        "original_image": [item["original_image"] for item in batch],
        "original_left_mask": [item["original_left_mask"] for item in batch],
        "original_right_mask": [item["original_right_mask"] for item in batch],

        "image_path": [item["image_path"] for item in batch],
        "mask_path": [item["mask_path"] for item in batch],

        "sample_id": [item["sample_id"] for item in batch],
        "dataset_name": [item["dataset_name"] for item in batch],
    }


def build_dataloaders(
        args: Any,
        device: torch.device,
) -> tuple[DataLoader, DataLoader, FramesPerSecondSampler]:
    """创建训练/验证 Dataset、FPS Sampler 和 DataLoader。"""
    train_dataset = MultiServerDualHandDataset(
        dataset_root=args.dataset_root,
        split="train",
        test_seq_count=args.test_seq_count,
        image_size=args.image_size,
        use_augmentation=not args.disable_augmentation,
        dataset_names=args.dataset_names,
    )

    val_dataset = MultiServerDualHandDataset(
        dataset_root=args.dataset_root,
        split="val",
        test_seq_count=args.test_seq_count,
        image_size=args.image_size,
        use_augmentation=False,
        dataset_names=args.dataset_names,
    )

    train_sampler = FramesPerSecondSampler(
        dataset=train_dataset,
        frames_per_second=args.frames_per_second,
        shuffle=True,
        seed=args.seed,
    )

    val_sampler = FramesPerSecondSampler(
        dataset=val_dataset,
        frames_per_second=args.frames_per_second,
        shuffle=False,
        seed=args.seed,
    )

    if len(train_sampler) == 0:
        raise ValueError("训练 Sampler 没有选出任何样本")

    if len(val_sampler) == 0:
        raise ValueError("验证 Sampler 没有选出任何样本")

    train_loader_kwargs: dict[str, object] = {
        "dataset": train_dataset,
        "batch_size": args.batch_size,
        "sampler": train_sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_batch,
    }

    val_loader_kwargs: dict[str, object] = {
        "dataset": val_dataset,
        "batch_size": args.val_batch_size,
        "sampler": val_sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_batch,
    }

    if args.num_workers > 0:
        train_loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": max(2, args.prefetch_factor,),
            }
        )

        val_loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": max(2, args.prefetch_factor,),
            }
        )

    train_loader = DataLoader(
        **train_loader_kwargs
    )

    val_loader = DataLoader(
        **val_loader_kwargs
    )

    logging.info(
    "Dataloaders built | Train: %d | Val: %d | Batch: %d",
    len(train_loader.sampler),
    len(val_loader.sampler),
    args.batch_size,
    )

    return train_loader, val_loader, train_sampler

