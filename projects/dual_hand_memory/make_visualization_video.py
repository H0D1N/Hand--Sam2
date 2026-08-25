"""把一个 dataset 的可视化 PNG 按 sequence、frame、step 合成 MP4。"""

import argparse
import re
from pathlib import Path


FRAME_NAME = re.compile(r"sequence_(\d+)_frame_(\d+)_step_(\d+)\.png$")


def parse_frame_path(path: Path) -> tuple[int, int, int]:
    """从文件名解析 sequence、frame 和 step 编号。"""

    match = FRAME_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"无法解析可视化图片名: {path.name}")
    return tuple(int(value) for value in match.groups())


def make_video(input_dir: Path, output: Path, fps: float = 2.0, sequence_gap_seconds: float = 2.0) -> None:
    """连续播放所有帧，并在相邻 sequences 之间插入黑屏。"""

    import cv2

    image_paths = sorted(input_dir.glob("sequence_*_frame_*_step_*.png"), key=parse_frame_path)
    if not image_paths:
        raise ValueError(f"没有找到可视化 PNG: {input_dir}")

    first_image = cv2.imread(str(image_paths[0]))
    if first_image is None:
        raise ValueError(f"无法读取图片: {image_paths[0]}")
    height, width = first_image.shape[:2]
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频: {output}")

    blank = first_image.copy()
    blank.fill(0)
    gap_frames = round(fps * sequence_gap_seconds)
    previous_sequence = None
    try:
        for image_path in image_paths:
            sequence_idx, _, _ = parse_frame_path(image_path)
            if previous_sequence is not None and sequence_idx != previous_sequence:
                for _ in range(gap_frames):
                    writer.write(blank)

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
            previous_sequence = sequence_idx
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine visualization PNGs into one video.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--sequence-gap-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_video(args.input_dir, args.output, args.fps, args.sequence_gap_seconds)


if __name__ == "__main__":
    main()
