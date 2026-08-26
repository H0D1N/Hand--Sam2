import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.make_visualization_video import find_frame_paths, make_video, parse_frame_path


def main():
    paths = [
        Path("sequence_0001_frame_00_step_00.png"),
        Path("sequence_0000_frame_07_step_00.png"),
        Path("sequence_0000_frame_00_step_01.png"),
        Path("sequence_0000_frame_00_step_00.png"),
    ]
    assert [path.name for path in sorted(paths, key=parse_frame_path)] == [
        "sequence_0000_frame_00_step_00.png",
        "sequence_0000_frame_00_step_01.png",
        "sequence_0000_frame_07_step_00.png",
        "sequence_0001_frame_00_step_00.png",
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "frames"
        input_dir.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        for name in ("sequence_0000_frame_00_step_00.png", "sequence_0000_frame_00_step_01.png", "sequence_0001_frame_00_step_00.png"):
            cv2.imwrite(str(input_dir / name), image)
        output = Path(temp_dir) / "visualization.mp4"
        make_video(input_dir, output, fps=2.0, sequence_gap_seconds=1.0)
        video = cv2.VideoCapture(str(output))
        assert output.stat().st_size > 0
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        video.release()

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "frames"
        (input_dir / "dataset_b").mkdir(parents=True)
        (input_dir / "dataset_a").mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        frame_names = ("sequence_0000_frame_00_step_01.png", "sequence_0000_frame_00_step_00.png")
        for dataset_name in ("dataset_b", "dataset_a"):
            for name in frame_names:
                cv2.imwrite(str(input_dir / dataset_name / name), image)

        assert [path.relative_to(input_dir).as_posix() for path in find_frame_paths(input_dir)] == [
            "dataset_a/sequence_0000_frame_00_step_00.png",
            "dataset_a/sequence_0000_frame_00_step_01.png",
            "dataset_b/sequence_0000_frame_00_step_00.png",
            "dataset_b/sequence_0000_frame_00_step_01.png",
        ]

        output = Path(temp_dir) / "visualization.mp4"
        make_video(input_dir, output, fps=2.0, sequence_gap_seconds=1.0)
        video = cv2.VideoCapture(str(output))
        assert output.stat().st_size > 0
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 6
        video.release()
    print("SAM2DualHandMemory visualization video ordering: OK")


if __name__ == "__main__":
    main()
