from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cv2


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将可视化图片按文件名排序并合成为视频。")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--format", choices=["avi", "mp4"], default="avi")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=2560)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def sorted_image_paths(input_dir: Path) -> list[Path]:
    return sorted(
        path for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def resize_with_padding(frame, frame_size: tuple[int, int]):
    target_w, target_h = frame_size
    h, w = frame.shape[:2]
    scale = min(target_w / w, target_h / h)

    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

    left = (target_w - new_w) // 2
    right = target_w - new_w - left
    top = (target_h - new_h) // 2
    bottom = target_h - new_h - top

    return cv2.copyMakeBorder(
        frame, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(0, 0, 0),
    )


def make_video(
    input_dir: Path,
    output_dir: Path,
    fps: float,
    video_format: str,
    stride: int,
    max_frames: int,
    frame_size: tuple[int, int],
) -> Path:
    if not input_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在：{input_dir}")

    image_paths = sorted_image_paths(input_dir)[::max(1, stride)]
    if max_frames > 0:
        image_paths = image_paths[:max_frames]
    if not image_paths:
        raise RuntimeError(f"目录中没有可用图片：{input_dir}")

    width, height = frame_size
    frame_size = (width - width % 2, height - height % 2)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_dir.name}.{video_format}"

    codec = "MJPG" if video_format == "avi" else "mp4v"
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"无法创建视频：{output_path}")

    written = 0
    try:
        for image_path in image_paths:
            frame = cv2.imread(str(image_path))
            if frame is None:
                logging.warning("跳过无法读取的图片：%s", image_path)
                continue

            writer.write(resize_with_padding(frame, frame_size))
            written += 1
    finally:
        writer.release()

    if written == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("没有成功写入任何帧。")

    logging.info("已生成 %s（%d 帧）", output_path, written)
    return output_path


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    if args.fps <= 0 or args.width <= 0 or args.height <= 0:
        raise SystemExit("fps、width 和 height 必须大于 0。")

    try:
        make_video(
            args.input_dir,
            args.output_dir,
            args.fps,
            args.format,
            args.stride,
            args.max_frames,
            (args.width, args.height),
        )
    except (FileNotFoundError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()