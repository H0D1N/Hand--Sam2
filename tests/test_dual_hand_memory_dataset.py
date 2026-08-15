import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.dataset import (
    ConsecutiveClipDataset,
    collate_clip_batch,
)
from projects.dual_hand_memory import dataset as memory_dataset
from projects.framewise_sam2_modified.dataset import CombinedStreamDataset


IMAGE_SIZE = 16


class FakeFrameDataset(Dataset):
    def __init__(self, dataset_name, stream_id, frame_numbers):
        self.dataset_name = dataset_name
        self.frame_numbers = frame_numbers
        self.streams = {
            stream_id: {
                "dataset_name": dataset_name,
                "fps": 30,
                "sample_indices": list(range(len(frame_numbers))),
                "frame_numbers": frame_numbers,
            }
        }

    def __len__(self):
        return len(self.frame_numbers)

    def __getitem__(self, index):
        frame_number = self.frame_numbers[index]
        image = torch.full((3, IMAGE_SIZE, IMAGE_SIZE), frame_number, dtype=torch.float32)
        mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)

        return {
            "image": image,
            "left_mask": mask.clone(),
            "right_mask": mask.clone(),
            "original_size": (IMAGE_SIZE, IMAGE_SIZE),
            "original_image": image.clone(),
            "original_left_mask": mask.clone(),
            "original_right_mask": mask.clone(),
            "image_path": f"{self.dataset_name}/{frame_number}.png",
            "mask_path": f"{self.dataset_name}/{frame_number}.png",
            "sample_id": f"{self.dataset_name}-{frame_number}",
            "dataset_name": self.dataset_name,
        }


def check_consecutive_clips():
    frame_dataset = FakeFrameDataset(
        dataset_name="multiserver",
        stream_id="multiserver/sequence/cam-a",
        frame_numbers=[1, 2, 4, 5, 6, 7, 9],
    )
    clip_dataset = ConsecutiveClipDataset(
        frame_dataset,
        clip_length=2,
        clip_stride=2,
    )

    assert [clip["frame_numbers"] for clip in clip_dataset.clips] == [
        [1, 2],
        [4, 5],
        [6, 7],
    ]
    assert [clip["sample_indices"] for clip in clip_dataset.clips] == [
        [0, 1],
        [2, 3],
        [4, 5],
    ]


def check_mixed_dataset_and_collate():
    multiserver = FakeFrameDataset(
        "multiserver",
        "multiserver/sequence/cam-a",
        [10, 11],
    )
    dexycb = FakeFrameDataset(
        "DexYCB",
        "DexYCB/sequence/cam-a",
        [20, 21],
    )
    frame_dataset = CombinedStreamDataset([multiserver, dexycb])
    clip_dataset = ConsecutiveClipDataset(frame_dataset, clip_length=2, clip_stride=2)

    assert clip_dataset.clips[1]["sample_indices"] == [2, 3]
    assert clip_dataset.clips[1]["frame_numbers"] == [20, 21]

    batch = collate_clip_batch([clip_dataset[0], clip_dataset[1]])
    assert batch["image"].shape == (2, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["left_mask"].shape == (2, 2, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["right_mask"].shape == (2, 2, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert batch["dataset_name"] == ["multiserver", "DexYCB"]
    assert batch["frame_numbers"] == [[10, 11], [20, 21]]
    assert batch["sample_indices"] == [[0, 1], [2, 3]]


def check_build_dataloaders():
    def build_multiserver(**kwargs):
        split = kwargs["split"]
        return FakeFrameDataset("multiserver", f"multiserver/{split}", [1, 2])

    def build_dexycb(**kwargs):
        split = kwargs["split"]
        return FakeFrameDataset("DexYCB", f"DexYCB/{split}", [1, 2])

    args = SimpleNamespace(
        dataset_root="multiserver", dex_ycb_root="dexycb",
        dataset_names=None, test_seq_count=2, image_size=IMAGE_SIZE,
        clip_length=2, clip_stride=2, batch_size=1, val_batch_size=1,
        num_workers=0, prefetch_factor=2,
    )

    with (
        patch.object(memory_dataset, "MultiServerDualHandDataset", side_effect=build_multiserver),
        patch.object(memory_dataset, "DexYCBDataset", side_effect=build_dexycb),
    ):
        for dataset_mode, expected_clips in (
            ("multiserver", 1),
            ("dexycb", 1),
            ("mixed", 2),
        ):
            args.dataset_mode = dataset_mode
            train_loader, val_loader = memory_dataset.build_dataloaders(
                args,
                torch.device("cpu"),
            )
            assert len(train_loader.dataset) == expected_clips
            assert len(val_loader.dataset) == expected_clips
            assert next(iter(val_loader))["image"].shape == (
                1,
                2,
                3,
                IMAGE_SIZE,
                IMAGE_SIZE,
            )


def main():
    check_consecutive_clips()
    check_mixed_dataset_and_collate()
    check_build_dataloaders()
    print("Dual-hand Memory clip dataset: OK")


if __name__ == "__main__":
    main()
