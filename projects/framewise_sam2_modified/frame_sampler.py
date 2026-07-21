"""按每条视频流的原始 FPS 均匀抽取单帧样本。"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

import torch
from torch.utils.data import Sampler


class FramesPerSecondSampler(Sampler[int]):
    """
    从 Dataset 的每条 stream 中按目标帧率均匀选择样本。

    Dataset 需要提供 ``streams`` 字典，每条 stream 的结构为：

    {
        "dataset/sequence/camera": {
            "dataset_name": "dataset",
            "fps": 30,
            "sample_indices": [0, 1, 2, ...],
        }
    }

    ``sample_indices`` 必须已经按时间顺序排列，因此不需要额外保存
    ``frame_position``。Sampler 只返回 Dataset 索引，DataLoader 会再调用
    ``dataset[index]`` 读取图片和 mask。
    """

    def __init__(
        self,
        dataset: Any,
        frames_per_second: float,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if (
            isinstance(frames_per_second, bool)
            or not isinstance(frames_per_second, (int, float))
            or not math.isfinite(frames_per_second)
            or frames_per_second <= 0
        ):
            raise ValueError("frames_per_second 必须是有限的正数")

        streams = getattr(dataset, "streams", None)
        if not isinstance(streams, Mapping):
            raise TypeError("dataset 必须提供 streams 字典")

        self.dataset = dataset
        self.frames_per_second = float(frames_per_second)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.epoch = 0
        self._selected_indices = self._select_indices(streams)

    def _select_indices(self, streams: Mapping[str, Any]) -> list[int]:
        selected_indices = []
        dataset_size = len(self.dataset)

        for stream_id, stream in streams.items():
            if not isinstance(stream, Mapping):
                raise TypeError(f"stream {stream_id!r} 的配置必须是字典")

            fps = stream.get("fps")
            if (
                isinstance(fps, bool)
                or not isinstance(fps, (int, float))
                or not math.isfinite(fps)
                or fps <= 0
            ):
                raise ValueError(f"stream {stream_id!r} 没有有效的 fps")

            fps = float(fps)
            if self.frames_per_second > fps:
                raise ValueError(
                    f"stream {stream_id!r} 的 fps={fps:g} 小于目标帧率 "
                    f"{self.frames_per_second:g}"
                )

            sample_indices = stream.get("sample_indices")
            if not isinstance(sample_indices, Sequence) or isinstance(
                sample_indices, (str, bytes)
            ):
                raise TypeError(
                    f"stream {stream_id!r} 必须提供有序的 sample_indices"
                )

            for sample_index in sample_indices:
                if (
                    isinstance(sample_index, bool)
                    or not isinstance(sample_index, int)
                    or not 0 <= sample_index < dataset_size
                ):
                    raise ValueError(
                        f"stream {stream_id!r} 包含无效样本索引 "
                        f"{sample_index!r}"
                    )

            step = fps / self.frames_per_second
            sample_number = 0

            while True:
                stream_position = round(sample_number * step)
                if stream_position >= len(sample_indices):
                    break

                selected_indices.append(sample_indices[stream_position])
                sample_number += 1

        return selected_indices

    def set_epoch(self, epoch: int) -> None:
        """设置当前 epoch，使 shuffle 顺序能够按 epoch 确定性变化。"""
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        if not self.shuffle or len(self._selected_indices) < 2:
            return iter(self._selected_indices)

        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        order = torch.randperm(
            len(self._selected_indices),
            generator=generator,
        ).tolist()

        return iter([self._selected_indices[index] for index in order])

    def __len__(self) -> int:
        return len(self._selected_indices)
