"""把完整 GT 单帧数据组织成长视频或同步多视角长视频。"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import json
from pathlib import Path
import re

import torch

from projects.framewise_sam2_modified.dataset import (
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
    _find_image_mask_pairs,
    _find_valid_cameras,
)


def natural_sort_key(value) -> tuple:
    text = str(value).casefold()
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part)
        for part in re.split(r"(\d+)", text)
    ) + ((2, text),)


class NaturalValMultiServerDataset(MultiServerDualHandDataset):
    """评估专用的全 GT Dataset；按自然数字顺序选择最后 N 个序列。"""

    def __init__(
        self,
        dataset_root,
        test_seq_count: int,
        image_size: int,
        dataset_names=None,
    ):
        if test_seq_count <= 0:
            raise ValueError("test_seq_count 必须大于 0")
        if image_size <= 0:
            raise ValueError("image_size 必须大于 0")

        self.dataset_root = Path(dataset_root)
        self.split = "val"
        self.image_size = image_size
        self.streams = {}
        self.samples = []
        self.mask_values = {}
        self.augmentation = None

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
            self.mask_values[dataset_name] = {
                "left": config["mask_values"]["left"],
                "right": config["mask_values"]["right"],
            }
            configured_root = Path(
                config.get("data_root", f"sources/{dataset_name}")
            )
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
                key=natural_sort_key,
            )
            cameras_by_sequence = {
                sequence_dir: camera_names
                for sequence_dir in sequence_dirs
                if (camera_names := _find_valid_cameras(sequence_dir, config))
            }
            valid_sequences = list(cameras_by_sequence)
            if len(valid_sequences) < test_seq_count:
                raise ValueError(
                    f"{dataset_name} 只有 {len(valid_sequences)} 条有效序列，"
                    f"少于 test_seq_count={test_seq_count}"
                )
            selected_sequences = valid_sequences[-test_seq_count:]
            print(
                f"[VAL] {dataset_name}: 自然数字顺序选择序列="
                f"{[path.name for path in selected_sequences]}",
                flush=True,
            )

            for sequence_dir in selected_sequences:
                for camera_name in sorted(
                    cameras_by_sequence[sequence_dir], key=natural_sort_key
                ):
                    image_dir = sequence_dir / config["rgb_dir"].format(
                        view=camera_name
                    )
                    mask_dir = sequence_dir / config["mask_dir"].format(
                        view=camera_name
                    )
                    stream_id = f"{dataset_name}/{sequence_dir.name}/{camera_name}"
                    sample_indices = []
                    frame_numbers = []

                    for image_path, mask_path in _find_image_mask_pairs(
                        image_dir, mask_dir, config
                    ):
                        relative_image_path = image_path.relative_to(
                            sequence_dir.parent
                        )
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
                            "mask_path": mask_path,
                            "dataset_root": local_root,
                            "dataset_name": dataset_name,
                            "sample_id": sample_id,
                        })

                    if sample_indices:
                        self.streams[stream_id] = {
                            "dataset_name": dataset_name,
                            "fps": config.get("fps"),
                            "sample_indices": sample_indices,
                            "frame_numbers": frame_numbers,
                        }


@dataclass(frozen=True)
class LongVideoSequence:
    dataset_name: str
    sequence_id: str
    segment_index: int
    view_names: tuple[str, ...]
    frame_numbers: tuple[int, ...]
    sample_indices: tuple[tuple[int, ...], ...]

    @property
    def num_frames(self) -> int:
        return len(self.frame_numbers)

    @property
    def evaluation_id(self) -> str:
        views = "+".join(self.view_names)
        return f"{self.sequence_id}/{views}#segment-{self.segment_index}"


def _split_contiguous_frames(frame_numbers: list[int]) -> list[list[int]]:
    if not frame_numbers:
        return []

    runs = [[frame_numbers[0]]]
    for frame_number in frame_numbers[1:]:
        if frame_number != runs[-1][-1] + 1:
            runs.append([])
        runs[-1].append(frame_number)
    return runs


class LongVideoDataset:
    """
    保留完整 GT，并按真实帧号拆成连续的长序列。

    ``num_views=1`` 时每条 camera stream 独立评估；``num_views>1`` 时先
    按 sequence 聚合 camera，再取所选视角的共同帧。
    """

    def __init__(self, frame_dataset, num_views: int = 1, min_sequence_length: int = 2):
        if num_views < 1:
            raise ValueError("num_views 必须大于 0")
        if min_sequence_length < 1:
            raise ValueError("min_sequence_length 必须大于 0")

        self.frame_dataset = frame_dataset
        self.num_views = num_views
        self.min_sequence_length = min_sequence_length
        self.sequences: list[LongVideoSequence] = []

        if num_views == 1:
            self._build_single_view_sequences()
        else:
            self._build_multiview_sequences()

    def _append_runs(
        self,
        *,
        dataset_name: str,
        sequence_id: str,
        view_names: tuple[str, ...],
        frame_maps: dict[str, dict[int, int]],
        common_frames: list[int],
    ) -> None:
        for segment_index, frame_numbers in enumerate(
            _split_contiguous_frames(common_frames)
        ):
            if len(frame_numbers) < self.min_sequence_length:
                continue
            self.sequences.append(LongVideoSequence(
                dataset_name=dataset_name,
                sequence_id=sequence_id,
                segment_index=segment_index,
                view_names=view_names,
                frame_numbers=tuple(frame_numbers),
                sample_indices=tuple(
                    tuple(frame_maps[view][frame] for frame in frame_numbers)
                    for view in view_names
                ),
            ))

    def _build_single_view_sequences(self) -> None:
        for stream_id, stream in sorted(
            self.frame_dataset.streams.items(), key=lambda item: natural_sort_key(item[0])
        ):
            sequence_id, view_name = stream_id.rsplit("/", 1)
            frame_map = dict(zip(stream["frame_numbers"], stream["sample_indices"]))
            self._append_runs(
                dataset_name=stream["dataset_name"],
                sequence_id=sequence_id,
                view_names=(view_name,),
                frame_maps={view_name: frame_map},
                common_frames=sorted(frame_map),
            )

    def _build_multiview_sequences(self) -> None:
        grouped_streams: dict[str, dict[str, dict]] = {}
        for stream_id, stream in self.frame_dataset.streams.items():
            sequence_id, view_name = stream_id.rsplit("/", 1)
            grouped_streams.setdefault(sequence_id, {})[view_name] = stream

        for sequence_id, streams in sorted(
            grouped_streams.items(), key=lambda item: natural_sort_key(item[0])
        ):
            view_names = sorted(streams, key=natural_sort_key)
            if len(view_names) < self.num_views:
                continue

            for selected_views in combinations(view_names, self.num_views):
                frame_maps = {
                    view: dict(zip(
                        streams[view]["frame_numbers"],
                        streams[view]["sample_indices"],
                    ))
                    for view in selected_views
                }
                common_frames = sorted(
                    set.intersection(*(set(frame_map) for frame_map in frame_maps.values()))
                )
                self._append_runs(
                    dataset_name=streams[selected_views[0]]["dataset_name"],
                    sequence_id=sequence_id,
                    view_names=selected_views,
                    frame_maps=frame_maps,
                    common_frames=common_frames,
                )

    def load_frame(self, sequence: LongVideoSequence, frame_index: int) -> dict:
        frames = [
            self.frame_dataset[view_indices[frame_index]]
            for view_indices in sequence.sample_indices
        ]
        for frame in frames:
            if frame["original_left_mask"] is None or frame["original_right_mask"] is None:
                raise ValueError("长视频评估要求每一帧都有原始 GT mask")

        return {
            "image": torch.stack([frame["image"] for frame in frames]),
            "left_mask": torch.stack([frame["left_mask"] for frame in frames]),
            "right_mask": torch.stack([frame["right_mask"] for frame in frames]),
            "original_left_mask": [frame["original_left_mask"] for frame in frames],
            "original_right_mask": [frame["original_right_mask"] for frame in frames],
            "image_path": [frame["image_path"] for frame in frames],
        }

    def __len__(self) -> int:
        return len(self.sequences)


def build_full_gt_frame_dataset(
    *,
    dataset_mode: str,
    image_size: int,
    dataset_root,
    dataset_names,
    test_seq_count: int,
    dex_ycb_root,
):
    """使用训练数据读取器的 val split，但不再切成固定长度 Clip。"""
    datasets = []
    if dataset_mode in {"multiserver", "mixed"}:
        datasets.append(NaturalValMultiServerDataset(
            dataset_root=dataset_root,
            test_seq_count=test_seq_count,
            image_size=image_size,
            dataset_names=dataset_names,
        ))
    if dataset_mode in {"dexycb", "mixed"}:
        if dex_ycb_root is None:
            raise ValueError("dataset_mode=dexycb/mixed 时必须提供 dex_ycb_root")
        datasets.append(DexYCBDataset(
            dataset_root=dex_ycb_root,
            split="val",
            setup="s0",
            image_size=image_size,
            use_augmentation=False,
        ))
    if not datasets:
        raise ValueError(f"不支持的 dataset_mode: {dataset_mode}")
    return CombinedStreamDataset(datasets) if len(datasets) > 1 else datasets[0]
