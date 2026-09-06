from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import Dataset, Subset

from inference import dataset as prediction_data


class FrameDataset(Dataset):
    def __init__(self, num_frames=3, image_size=128, prefix="video"):
        self.num_frames, self.image_size, self.prefix = num_frames, image_size, prefix
        self.streams = {prefix: {"sample_indices": list(range(num_frames))}}
        self.reads = []
        self.change_later_gt = False

    def __len__(self):
        return self.num_frames

    def __getitem__(self, index):
        self.reads.append(index)
        size = self.image_size
        image = torch.randn(3, size, size, generator=torch.Generator().manual_seed(index + 100))
        left, right = torch.zeros(1, size, size), torch.zeros(1, size, size)
        left[:, size // 8:size // 2, size // 8:size // 2] = 1
        right[:, size // 2:7 * size // 8, size // 2:7 * size // 8] = 1
        if self.change_later_gt and index > 0:
            left, right = 1 - left, 1 - right
        return {
            "image": image, "left_mask": left, "right_mask": right,
            "bbox": torch.zeros(2, 4), "box_labels": torch.zeros(2),
            "original_size": (size + 3, size + 5), "original_image": None,
            "original_left_mask": None, "original_right_mask": None,
            "image_path": f"{self.prefix}/rgb/{index * 3}.jpg", "mask_path": "unused",
            "sample_id": str(index), "dataset_name": self.prefix,
        }


def test_memory_loader_preserves_complete_stream_order():
    dataset = FrameDataset(5, image_size=16)
    indices = [2, 0, 4, 1, 3]
    dataset.streams["video"]["sample_indices"] = indices
    args = SimpleNamespace(model="memory", batch_size=1, num_workers=0, device="cpu")
    loader = prediction_data.build_loader(dataset, args, indices)
    assert isinstance(loader.dataset, Subset)
    assert dataset.reads == []
    assert [batch["sample_id"][0] for batch in loader] == [str(i) for i in indices]
    assert dataset.reads == indices
    args.batch_size = 2
    with pytest.raises(ValueError, match="batch_size=1"):
        prediction_data.build_loader(dataset, args, indices)
    args.model = "framewise"
    assert [batch["image"].shape[0] for batch in prediction_data.build_loader(dataset, args)] == [2, 2, 1]


@pytest.mark.parametrize("selection", ["multiserver", "dexycb", "mixed"])
def test_dataset_selection_unchanged(monkeypatch, selection):
    calls = []

    def create_dataset(**kwargs):
        calls.append(kwargs)
        return FrameDataset(2, prefix=kwargs["dataset_root"])

    monkeypatch.setattr(prediction_data, "MultiServerDualHandDataset", create_dataset)
    monkeypatch.setattr(prediction_data, "DexYCBDataset", create_dataset)
    args = SimpleNamespace(dataset=selection, dataset_root="multi", dex_ycb_root="dex", test_seq_count=3,
                           image_size=128, dataset_names=["selected"], dex_ycb_setup="s0")
    dataset = prediction_data.build_frame_dataset(args)
    assert len(dataset) == (4 if selection == "mixed" else 2)
    assert all(call["split"] == "val" and not call["use_augmentation"] and call["image_size"] == 128 for call in calls)
    if selection != "dexycb":
        assert calls[0]["test_seq_count"] == 3 and calls[0]["dataset_names"] == ["selected"]
    if selection != "multiserver":
        assert calls[-1]["setup"] == "s0"
    if selection == "mixed":
        assert dataset.streams["multi"]["sample_indices"] == [0, 1]
        assert dataset.streams["dex"]["sample_indices"] == [2, 3]
