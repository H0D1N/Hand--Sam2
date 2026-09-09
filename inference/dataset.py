"""推理 Dataset：memory 模式只要求每条视频流的第一帧有 mask。"""

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.transforms import functional as TF

from projects.framewise_sam2_modified.dataset import (
    SAM2_MEAN,
    SAM2_STD,
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
    collate_batch,
)


def _build_frame_item(
    *,
    image_path: str | Path,
    image_size: int,
    dataset_root: str | Path,
    dataset_name: str,
    sample_id: str,
    mask_path: str | Path | None = None,
    left_mask_np: np.ndarray | None = None,
    right_mask_np: np.ndarray | None = None,
) -> dict:
    """构造推理样本；无 GT 的后续帧使用零 mask 占位。"""
    image_path = Path(image_path)
    image_np = np.array(Image.open(image_path).convert("RGB"))
    original_height, original_width = image_np.shape[:2]
    original_image = torch.from_numpy(image_np.copy()).permute(2, 0, 1)

    if left_mask_np is None or right_mask_np is None:
        left_mask_np = np.zeros(image_np.shape[:2], dtype=np.uint8)
        right_mask_np = np.zeros(image_np.shape[:2], dtype=np.uint8)
    elif left_mask_np.shape != image_np.shape[:2]:
        left_mask_np = cv2.resize(
            left_mask_np,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )
        right_mask_np = cv2.resize(
            right_mask_np,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

    original_left_mask = torch.from_numpy(left_mask_np.astype(np.float32)).unsqueeze(0)
    original_right_mask = torch.from_numpy(right_mask_np.astype(np.float32)).unsqueeze(0)
    resized_image = cv2.resize(
        image_np,
        (image_size, image_size),
        interpolation=cv2.INTER_LINEAR,
    )
    left_mask_np = cv2.resize(
        left_mask_np,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )
    right_mask_np = cv2.resize(
        right_mask_np,
        (image_size, image_size),
        interpolation=cv2.INTER_NEAREST,
    )
    image = torch.from_numpy(resized_image.copy()).permute(2, 0, 1).float() / 255.0
    image = TF.normalize(image, mean=SAM2_MEAN, std=SAM2_STD)

    bbox = torch.tensor(
        [[0.0, 0.0], [image_size - 1.0, image_size - 1.0]],
        dtype=torch.float32,
    )

    return {
        "image": image,
        "left_mask": torch.from_numpy(left_mask_np.astype(np.float32)).unsqueeze(0),
        "right_mask": torch.from_numpy(right_mask_np.astype(np.float32)).unsqueeze(0),
        "bbox": bbox,
        "box_labels": torch.tensor([2, 3], dtype=torch.int64),
        "original_size": (original_height, original_width),
        "original_image": original_image,
        "original_left_mask": original_left_mask,
        "original_right_mask": original_right_mask,
        "image_path": str(image_path),
        "mask_path": None if mask_path is None else str(mask_path),
        "dataset_root": str(dataset_root),
        "sample_id": sample_id,
        "dataset_name": dataset_name,
    }


def _natural_sort_key(path: Path) -> tuple:
    parts = re.split(r"(\d+)", path.name.casefold())
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
    ) + ((2, path.name.casefold()),)


def _find_images(image_dir: Path, config: dict) -> list[Path]:
    extensions = {suffix.lower() for suffix in config.get("image_extensions", [".png"])}
    return [
        path for path in sorted(image_dir.iterdir(), key=_natural_sort_key)
        if path.is_file() and path.suffix.lower() in extensions
    ]


def _find_mask(image_path: Path, mask_dir: Path, config: dict) -> Path | None:
    extensions = {suffix.lower() for suffix in config.get("mask_extensions", [".png"])}
    candidates = [
        path for path in sorted(mask_dir.iterdir(), key=_natural_sort_key)
        if path.is_file()
        and path.stem == image_path.stem
        and path.suffix.lower() in extensions
    ]
    if not candidates:
        return None
    return next(
        (path for path in candidates if path.suffix.lower() == image_path.suffix.lower()),
        candidates[0],
    )


class FirstFrameMaskMultiServerDataset(Dataset):
    """读取全部 RGB，每条 dataset/sequence/camera 仅要求首帧有 mask。"""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "val",
        test_seq_count: int = 2,
        image_size: int = 1024,
        use_augmentation: bool = False,
        dataset_names: list[str] | None = None,
    ) -> None:
        if split != "val" or use_augmentation:
            raise ValueError("首帧 mask Dataset 仅用于无增强的 val 推理")
        if test_seq_count <= 0:
            raise ValueError("test_seq_count 必须大于 0")
        if image_size <= 0:
            raise ValueError("image_size 必须大于 0")

        self.dataset_root = Path(dataset_root)
        self.split = split
        self.image_size = image_size
        self.samples = []
        self.streams = {}
        self.mask_values = {}
        catalog_path = self.dataset_root / "datasets.json"
        if not catalog_path.is_file():
            raise FileNotFoundError(f"找不到本地数据集配置: {catalog_path}")
        with catalog_path.open("r", encoding="utf-8") as handle:
            configs = {
                config["dataset_name"]: config
                for config in json.load(handle)["datasets"]
            }

        selected_names = list(configs) if dataset_names is None else dataset_names
        missing_names = [name for name in selected_names if name not in configs]
        if missing_names:
            raise ValueError(f"datasets.json 中不存在: {missing_names}")

        for dataset_name in selected_names:
            config = configs[dataset_name]
            self.mask_values[dataset_name] = config["mask_values"]
            configured_root = Path(config.get("data_root", f"sources/{dataset_name}"))
            local_root = (
                configured_root
                if configured_root.is_absolute()
                else self.dataset_root / configured_root
            )
            if not local_root.is_dir():
                raise FileNotFoundError(f"本地数据集目录不存在: {local_root}")

            sequence_dirs = sorted(
                (
                    path for path in local_root.glob(config["sequence_glob"])
                    if path.is_dir()
                ),
                key=_natural_sort_key,
            )
            valid_sequences = []
            cameras_by_sequence = {}
            raw_view_glob = config["view_glob"]
            view_globs = raw_view_glob if isinstance(raw_view_glob, list) else [raw_view_glob]

            for sequence_dir in sequence_dirs:
                valid_cameras = []
                camera_dirs = sorted(
                    (path for path in sequence_dir.iterdir() if path.is_dir()),
                    key=_natural_sort_key,
                )
                for camera_dir in camera_dirs:
                    if not any(camera_dir.match(pattern) for pattern in view_globs):
                        continue
                    image_dir = sequence_dir / config["rgb_dir"].format(view=camera_dir.name)
                    mask_dir = sequence_dir / config["mask_dir"].format(view=camera_dir.name)
                    if not image_dir.is_dir() or not mask_dir.is_dir():
                        continue
                    image_paths = _find_images(image_dir, config)
                    if image_paths and _find_mask(image_paths[0], mask_dir, config) is not None:
                        valid_cameras.append(camera_dir.name)
                    else:
                        print(
                            f"[VAL] 跳过首帧没有 mask 的视频流: "
                            f"{dataset_name}/{sequence_dir.name}/{camera_dir.name}",
                            flush=True,
                        )
                if valid_cameras:
                    valid_sequences.append(sequence_dir)
                    cameras_by_sequence[sequence_dir] = valid_cameras

            if len(valid_sequences) < test_seq_count:
                raise ValueError(
                    f"{dataset_name} 只有 {len(valid_sequences)} 条首帧带 mask 的有效序列，"
                    f"必须大于 test_seq_count={test_seq_count}"
                )

            selected_sequences = valid_sequences[-test_seq_count:]
            print(
                f"[VAL] {dataset_name}: "
                f"选择序列={[path.name for path in selected_sequences]}",
                flush=True,
            )

            for sequence_dir in selected_sequences:
                for camera_name in cameras_by_sequence[sequence_dir]:
                    image_dir = sequence_dir / config["rgb_dir"].format(view=camera_name)
                    mask_dir = sequence_dir / config["mask_dir"].format(view=camera_name)
                    image_paths = _find_images(image_dir, config)
                    first_mask_path = _find_mask(image_paths[0], mask_dir, config)
                    stream_id = f"{dataset_name}/{sequence_dir.name}/{camera_name}"
                    sample_indices = []
                    frame_numbers = []

                    for frame_idx, image_path in enumerate(image_paths):
                        relative_image_path = image_path.relative_to(sequence_dir.parent)
                        sample_id = (
                            (Path(dataset_name) / relative_image_path)
                            .with_suffix("")
                            .as_posix()
                            .replace("/", "__")
                        )
                        sample_indices.append(len(self.samples))
                        frame_numbers.append(int(image_path.stem))
                        self.samples.append({
                            "image_path": image_path,
                            "mask_path": first_mask_path if frame_idx == 0 else None,
                            "dataset_root": local_root,
                            "dataset_name": dataset_name,
                            "sample_id": sample_id,
                        })

                    self.streams[stream_id] = {
                        "dataset_name": dataset_name,
                        "fps": config.get("fps"),
                        "sample_indices": sample_indices,
                        "frame_numbers": frame_numbers,
                    }

        print(
            f"[VAL] 推理数据集: {len(self.streams)} 条视频流，{len(self.samples)} 帧",
            flush=True,
        )

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        left_mask_np = right_mask_np = None
        if sample["mask_path"] is not None:
            raw_mask_np = np.array(Image.open(sample["mask_path"]).convert("L"))
            values = self.mask_values[sample["dataset_name"]]
            left_mask_np = np.isin(raw_mask_np, values["left"]).astype(np.uint8)
            right_mask_np = np.isin(raw_mask_np, values["right"]).astype(np.uint8)

        return _build_frame_item(
            image_path=sample["image_path"],
            image_size=self.image_size,
            dataset_root=sample["dataset_root"],
            dataset_name=sample["dataset_name"],
            sample_id=sample["sample_id"],
            mask_path=sample["mask_path"],
            left_mask_np=left_mask_np,
            right_mask_np=right_mask_np,
        )

    def __len__(self) -> int:
        return len(self.samples)


class FirstFrameMaskDexYCBDataset(Dataset):
    """DexYCB 推理只在每条 stream 的首帧读取 label_file。"""

    def __init__(
        self,
        dataset_root: str | Path,
        split: str = "val",
        setup: str = "s0",
        image_size: int = 1024,
        use_augmentation: bool = False,
    ) -> None:
        if split != "val" or use_augmentation:
            raise ValueError("首帧 mask Dataset 仅用于无增强的 val 推理")
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.image_size = image_size
        self.dataset_name = "DexYCB"
        self.streams = {}
        if not self.dataset_root.is_dir():
            raise FileNotFoundError(f"DexYCB 数据集目录不存在: {self.dataset_root}")

        import os
        os.environ["DEX_YCB_DIR"] = str(self.dataset_root)
        from dex_ycb_toolkit.factory import get_dataset

        self.official_dataset = get_dataset(f"{setup}_{split}")
        self.samples = list(range(len(self.official_dataset)))
        for index in self.samples:
            image_path = Path(self.official_dataset[index]["color_file"])
            stream_id = (
                f"DexYCB/{image_path.parents[2].name}/"
                f"{image_path.parents[1].name}/{image_path.parent.name}"
            )
            stream = self.streams.setdefault(stream_id, {
                "dataset_name": self.dataset_name,
                "fps": 30,
                "sample_indices": [],
                "frame_numbers": [],
            })
            stream["sample_indices"].append(index)
            stream["frame_numbers"].append(int(image_path.stem.split("_")[-1]))

        self.first_frame_indices = {
            stream["sample_indices"][0]
            for stream in self.streams.values()
        }

    def __getitem__(self, index: int) -> dict:
        sample = self.official_dataset[self.samples[index]]
        image_path = Path(sample["color_file"])
        mask_path = None
        left_mask_np = right_mask_np = None
        if index in self.first_frame_indices:
            mask_path = Path(sample["label_file"])
            with np.load(mask_path) as label:
                hand_mask_np = (label["seg"] == 255).astype(np.uint8)
            if sample["mano_side"] == "left":
                left_mask_np = hand_mask_np
                right_mask_np = np.zeros_like(hand_mask_np)
            elif sample["mano_side"] == "right":
                left_mask_np = np.zeros_like(hand_mask_np)
                right_mask_np = hand_mask_np
            else:
                raise ValueError(f"未知的 mano_side={sample['mano_side']!r}: {image_path}")

        sample_id = (
            f"DexYCB__{image_path.parents[2].name}__"
            f"{image_path.parents[1].name}__{image_path.parent.name}__{image_path.stem}"
        )
        return _build_frame_item(
            image_path=image_path,
            image_size=self.image_size,
            dataset_root=self.dataset_root,
            dataset_name=self.dataset_name,
            sample_id=sample_id,
            mask_path=mask_path,
            left_mask_np=left_mask_np,
            right_mask_np=right_mask_np,
        )

    def __len__(self) -> int:
        return len(self.samples)


def build_frame_dataset(args: argparse.Namespace):
    datasets = []
    model_kind = getattr(args, "model", "framewise")
    multiserver_cls = (
        FirstFrameMaskMultiServerDataset
        if model_kind == "memory"
        else MultiServerDualHandDataset
    )
    dexycb_cls = (
        FirstFrameMaskDexYCBDataset
        if model_kind == "memory"
        else DexYCBDataset
    )
    if args.dataset in {"multiserver", "mixed"}:
        datasets.append(multiserver_cls(
            dataset_root=args.dataset_root,
            split="val",
            test_seq_count=args.test_seq_count,
            image_size=args.image_size,
            use_augmentation=False,
            dataset_names=args.dataset_names,
        ))
    if args.dataset in {"dexycb", "mixed"}:
        datasets.append(dexycb_cls(
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
