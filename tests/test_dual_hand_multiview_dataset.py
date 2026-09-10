import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.utils.data import Dataset


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_multiview import dataset as multiview_dataset
from projects.dual_hand_multiview.dataset import MultiViewConsecutiveClipDataset
from projects.framewise_sam2_modified.dataset import SAM2_MEAN, SAM2_STD


IMAGE_SIZE = 16


class FakeMultiViewFrameDataset(Dataset):
    def __init__(self, split="train"):
        self.streams = {
            f"dataset/{split}/cam-a": {
                "dataset_name": "dataset",
                "sample_indices": [0, 1],
                "frame_numbers": [1, 2],
            },
            f"dataset/{split}/cam-b": {
                "dataset_name": "dataset",
                "sample_indices": [2, 3],
                "frame_numbers": [1, 2],
            },
        }

    def __len__(self):
        return 4

    def __getitem__(self, index):
        mean = torch.tensor(SAM2_MEAN).view(3, 1, 1)
        std = torch.tensor(SAM2_STD).view(3, 1, 1)
        image = (torch.full((3, IMAGE_SIZE, IMAGE_SIZE), 0.5) - mean) / std
        left_mask = torch.zeros(1, IMAGE_SIZE, IMAGE_SIZE)
        left_mask[:, 4:12, 5:11] = 1
        right_mask = torch.zeros_like(left_mask)

        return {
            "image": image,
            "left_mask": left_mask,
            "right_mask": right_mask,
            "original_size": (IMAGE_SIZE, IMAGE_SIZE),
            "original_image": None,
            "original_left_mask": None,
            "original_right_mask": None,
            "image_path": f"{index}.png",
            "mask_path": f"{index}.png",
            "sample_id": str(index),
            "dataset_name": "dataset",
        }


def check_multiview_clip_augmentation():
    dataset = MultiViewConsecutiveClipDataset(
        FakeMultiViewFrameDataset(),
        num_views=2,
        clip_length=2,
        clip_stride=2,
        use_augmentation=True,
    )

    torch.manual_seed(3)
    sample = dataset[0]
    assert sample["image"].shape == (2, 2, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["left_mask"].shape == (2, 2, 1, IMAGE_SIZE, IMAGE_SIZE)
    assert torch.allclose(sample["image"][0, 0], sample["image"][0, 1])
    assert torch.allclose(sample["image"][1, 0], sample["image"][1, 1])
    assert torch.equal(sample["left_mask"][0, 0], sample["left_mask"][1, 1])
    assert set(torch.unique(sample["left_mask"]).tolist()) <= {0.0, 1.0}


def check_build_dataloaders_augmentation_switch():
    def build_multiserver(**kwargs):
        assert not kwargs["use_augmentation"]
        return FakeMultiViewFrameDataset(kwargs["split"])

    args = SimpleNamespace(
        dataset_mode="multiserver",
        dataset_root="multiserver",
        dex_ycb_root=None,
        dataset_names=None,
        test_seq_count=2,
        image_size=IMAGE_SIZE,
        num_views=2,
        clip_length=2,
        clip_stride=2,
        val_clip_stride=2,
        batch_size=1,
        val_batch_size=1,
        disable_augmentation=False,
        num_workers=0,
        prefetch_factor=2,
    )

    with patch.object(
        multiview_dataset,
        "MultiServerDualHandDataset",
        side_effect=build_multiserver,
    ):
        train_loader, val_loader = multiview_dataset.build_dataloaders(
            args,
            torch.device("cpu"),
        )
        assert train_loader.dataset.use_augmentation
        assert not val_loader.dataset.use_augmentation

        args.disable_augmentation = True
        train_loader, val_loader = multiview_dataset.build_dataloaders(
            args,
            torch.device("cpu"),
        )
        assert not train_loader.dataset.use_augmentation
        assert not val_loader.dataset.use_augmentation


def main():
    check_multiview_clip_augmentation()
    check_build_dataloaders_augmentation_switch()
    print("Dual-hand multiview clip dataset: OK")


if __name__ == "__main__":
    main()
