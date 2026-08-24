"""把一个 dataset 的 validation PNG 按 clip 顺序合成 MP4。"""

import argparse
import re
from pathlib import Path


FRAME_NAME = re.compile(r"clip_(\d+)_frame_(\d+)\.png$")


def parse_frame_path(path: Path) -> tuple[int, int]:
    """从文件名解析用于排序和插入 clip 间隔的编号。"""

    match = FRAME_NAME.fullmatch(path.name)
    if match is None:
        raise ValueError(f"无法解析 validation 图片名: {path.name}")
    return int(match.group(1)), int(match.group(2))


def make_video(input_dir: Path, output: Path, fps: float = 2.0, clip_gap_seconds: float = 2.0) -> None:
    """连续播放所有帧，并在相邻 clips 之间插入黑屏。"""

    import cv2

    image_paths = sorted(input_dir.glob("clip_*_frame_*.png"), key=parse_frame_path)
    if not image_paths:
        raise ValueError(f"没有找到 validation PNG: {input_dir}")

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
    gap_frames = round(fps * clip_gap_seconds)
    previous_clip = None
    try:
        for image_path in image_paths:
            clip_idx, _ = parse_frame_path(image_path)
            if previous_clip is not None and clip_idx != previous_clip:
                for _ in range(gap_frames):
                    writer.write(blank)

            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"无法读取图片: {image_path}")
            if image.shape[:2] != (height, width):
                image = cv2.resize(image, (width, height))
            writer.write(image)
            previous_clip = clip_idx
    finally:
        writer.release()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine validation PNGs into one video.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--clip-gap-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    make_video(args.input_dir, args.output, args.fps, args.clip_gap_seconds)


if __name__ == "__main__":
    main()
