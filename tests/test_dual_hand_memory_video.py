import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


from projects.dual_hand_memory.make_validation_video import make_video, parse_frame_path


def main():
    paths = [
        Path("clip_0001_frame_00.png"),
        Path("clip_0000_frame_07.png"),
        Path("clip_0000_frame_00.png"),
    ]
    assert [path.name for path in sorted(paths, key=parse_frame_path)] == [
        "clip_0000_frame_00.png",
        "clip_0000_frame_07.png",
        "clip_0001_frame_00.png",
    ]

    with tempfile.TemporaryDirectory() as temp_dir:
        input_dir = Path(temp_dir) / "frames"
        input_dir.mkdir()
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        for name in ("clip_0000_frame_00.png", "clip_0000_frame_01.png", "clip_0001_frame_00.png"):
            cv2.imwrite(str(input_dir / name), image)
        output = Path(temp_dir) / "validation.mp4"
        make_video(input_dir, output, fps=2.0, clip_gap_seconds=1.0)
        video = cv2.VideoCapture(str(output))
        assert output.stat().st_size > 0
        assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == 5
        video.release()
    print("SAM2DualHandMemory validation video ordering: OK")


if __name__ == "__main__":
    main()
