import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]

if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.framewise_sam2_modified.frame_sampler import (
    FramesPerSecondSampler,
)


class FakeDataset:
    def __init__(self):
        self.samples = [{} for _ in range(80)]
        self.streams = {
            "dataset_a/sequence_1/cam_0": {
                "dataset_name": "dataset_a",
                "fps": 30,
                "sample_indices": list(range(60)),
            },
            "dataset_b/sequence_2/cam_1": {
                "dataset_name": "dataset_b",
                "fps": 10,
                "sample_indices": list(range(60, 80)),
            },
        }

    def __len__(self):
        return len(self.samples)


def check_uniform_sampling():
    dataset = FakeDataset()
    sampler = FramesPerSecondSampler(
        dataset=dataset,
        frames_per_second=2,
        shuffle=False,
    )

    selected_indices = list(sampler)

    assert selected_indices == [
        0,
        15,
        30,
        45,
        60,
        65,
        70,
        75,
    ]
    assert len(sampler) == len(selected_indices)


def check_shuffle_changes_between_epochs():
    sampler = FramesPerSecondSampler(
        dataset=FakeDataset(),
        frames_per_second=2,
        shuffle=True,
        seed=42,
    )

    sampler.set_epoch(0)
    epoch_0_indices = list(sampler)

    sampler.set_epoch(1)
    epoch_1_indices = list(sampler)

    assert set(epoch_0_indices) == set(epoch_1_indices)
    assert epoch_0_indices != epoch_1_indices


def check_missing_fps_is_rejected():
    dataset = FakeDataset()
    dataset.streams["dataset_a/sequence_1/cam_0"]["fps"] = None

    try:
        FramesPerSecondSampler(
            dataset=dataset,
            frames_per_second=2,
        )
    except ValueError as error:
        assert "没有有效的 fps" in str(error)
    else:
        raise AssertionError("缺少 fps 时应该抛出 ValueError")


def check_target_fps_is_rejected_when_too_high():
    try:
        FramesPerSecondSampler(
            dataset=FakeDataset(),
            frames_per_second=11,
        )
    except ValueError as error:
        assert "小于目标帧率" in str(error)
    else:
        raise AssertionError("目标帧率超过原始帧率时应该抛出 ValueError")


def main():
    check_uniform_sampling()
    check_shuffle_changes_between_epochs()
    check_missing_fps_is_rejected()
    check_target_fps_is_rejected_when_too_high()

    print("FramesPerSecondSampler: OK")


if __name__ == "__main__":
    main()
