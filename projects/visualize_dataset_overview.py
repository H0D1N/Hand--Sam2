"""遍历预测 mask 生成"原图 | mask 叠加图"左右拼接的总览视频。

只以 mask-root 下已有的预测 mask 为基准：按相同 stem 回查 RGB 原图，
拼接成帧后按"序列 x 相机"输出视频与逐帧 PNG；尚未推理的序列、相机和
图片自动跳过，所有跳过与异常统一记录到问题报告 report.txt。

mask-root 与 dataset-root 以下必须保持相同的相对层级：

    dataset-root/subject1/ROM_right_bare_ball_1/camera0/rgb/000001.png
    mask-root/subject1/ROM_right_bare_ball_1/camera0/masks-test/000001.png

输出目录结构：

    output-dir/videos/<subject>/<ROM>/<camera>.mp4
    output-dir/frames/<subject>/<ROM>/<camera>/<stem>.png
    output-dir/report.txt
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MASK_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff"}

# 预测 mask 像素值 -> BGR 叠加色：1/2 为双手分割的两类，其余非零值按未知类别处理。
OVERLAY_COLORS = {1: (0, 0, 255), 2: (255, 0, 0)}
UNKNOWN_COLOR = (0, 255, 255)
OVERLAY_ALPHA = 0.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成原图与预测 mask 叠加的总览视频")

    io = parser.add_argument_group("io")
    io.add_argument("--dataset-root", type=Path, required=True, help="RGB 数据根目录")
    io.add_argument("--mask-root", type=Path, help="预测 mask 根目录；默认与 --dataset-root 相同")
    io.add_argument("--output-dir", type=Path, required=True, help="视频、帧和问题报告输出目录")

    layout = parser.add_argument_group("layout")
    layout.add_argument("--seq-glob", default="subject*/ROM*", help="序列目录 glob，相对数据根目录")
    layout.add_argument("--cam-name", default="cam*", help="相机目录 glob")
    layout.add_argument("--image-name", default="rgb", help="RGB 文件夹名称")
    layout.add_argument("--mask-name", default="masks-test", help="预测 mask 文件夹名称")

    video = parser.add_argument_group("video")
    video.add_argument("--fps", type=float, default=10.0, help="输出视频帧率")

    args = parser.parse_args()
    args.dataset_root = args.dataset_root.resolve()
    args.mask_root = (args.mask_root or args.dataset_root).resolve()
    args.output_dir = args.output_dir.resolve()
    return args


def natural_sort_key(text: str) -> tuple:
    """按数字大小而不是字典序排序，保证 2.png 排在 10.png 前面。"""
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", text)
    )


def find_rgb(rgb_dir: Path, stem: str) -> Path | None:
    """按相同 stem 在 RGB 目录中查找原图，自动尝试常见扩展名。"""
    for extension in IMAGE_EXTENSIONS:
        candidate = rgb_dir / f"{stem}{extension}"
        if candidate.is_file():
            return candidate
    return None


def read_mask(mask_path: Path) -> np.ndarray | None:
    """读取预测 mask 为单通道灰度图。"""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
    if mask is None:
        return None
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask


def build_overlay(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """把类别 mask 半透明叠加到原图上。"""
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    color_block = np.zeros_like(rgb)
    known_values = list(OVERLAY_COLORS)
    for value, color in OVERLAY_COLORS.items():
        color_block[mask == value] = color
    color_block[(mask != 0) & ~np.isin(mask, known_values)] = UNKNOWN_COLOR
    return cv2.addWeighted(rgb, 1.0 - OVERLAY_ALPHA, color_block, OVERLAY_ALPHA, 0.0)


def process_camera(
    *,
    rel_cam: Path,
    dataset_cam_dir: Path,
    mask_dir: Path,
    image_name: str,
    videos_dir: Path,
    frames_dir: Path,
    fps: float,
    problems: list[str],
    stats: dict[str, int],
) -> None:
    """为单个相机生成左右拼接的总览视频与逐帧 PNG。"""
    mask_paths = sorted(
        (path for path in mask_dir.iterdir() if path.suffix.lower() in MASK_EXTENSIONS),
        key=lambda path: natural_sort_key(path.stem),
    )
    if not mask_paths:
        problems.append(f"[未推理] {rel_cam}: mask 目录为空")
        return

    rgb_dir = dataset_cam_dir / image_name
    if not rgb_dir.is_dir():
        problems.append(f"[无RGB] {rel_cam}: dataset 下没有 {image_name} 目录")
        return

    frames = []
    missing_rgb = []
    for mask_path in mask_paths:
        stem = mask_path.stem
        rgb_path = find_rgb(rgb_dir, stem)
        if rgb_path is None:
            missing_rgb.append(mask_path.name)
            continue
        rgb = cv2.imread(str(rgb_path))
        if rgb is None:
            problems.append(f"[坏图] {rel_cam}/{mask_path.name}: 无法读取 RGB {rgb_path}")
            continue
        mask = read_mask(mask_path)
        if mask is None:
            problems.append(f"[坏mask] {rel_cam}/{mask_path.name}: 无法读取")
            continue
        if mask.shape[:2] != rgb.shape[:2]:
            problems.append(
                f"[尺寸不符] {rel_cam}/{mask_path.name}: mask {mask.shape[:2]} vs RGB {rgb.shape[:2]}"
            )
        frames.append((stem, np.hstack([rgb, build_overlay(rgb, mask)])))

    if missing_rgb:
        examples = ", ".join(missing_rgb[:10])
        if len(missing_rgb) > 10:
            examples += " ..."
        problems.append(f"[缺RGB] {rel_cam}: {len(missing_rgb)} 个 mask 找不到同 stem 原图: {examples}")

    # 有 RGB 但没有对应 mask 的图片，以及不连续的帧号
    rgb_stems = {path.stem for path in rgb_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
    unpredicted_count = len(rgb_stems - {path.stem for path in mask_paths})
    if unpredicted_count:
        problems.append(f"[未推理] {rel_cam}: {unpredicted_count} 帧没有预测 mask")

    numeric_stems = [int(stem) for stem, _ in frames if stem.isdigit()]
    gaps = [f"{a}->{b}" for a, b in zip(numeric_stems, numeric_stems[1:]) if b != a + 1]
    if gaps:
        problems.append(f"[帧间隔] {rel_cam}: 帧号不连续 {', '.join(gaps[:10])}")

    if not frames:
        problems.append(f"[无有效帧] {rel_cam}: 所有 mask 都找不到可读的 RGB")
        return

    video_path = videos_dir / rel_cam.parent / f"{rel_cam.name}.mp4"
    video_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0][1].shape[:2]
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        problems.append(f"[写视频失败] {rel_cam}: {video_path}")
        return

    cam_frames_dir = frames_dir / rel_cam
    cam_frames_dir.mkdir(parents=True, exist_ok=True)
    try:
        for stem, frame in frames:
            cv2.imwrite(str(cam_frames_dir / f"{stem}.png"), frame)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
    finally:
        writer.release()

    stats["videos"] += 1
    stats["frames"] += len(frames)
    logging.info("%s | %d 帧 | %s", rel_cam, len(frames), video_path)


def write_report(output_dir: Path, stats: dict[str, int], problems: list[str]) -> None:
    """把所有跳过与异常记录写入 report.txt。"""
    lines = [
        "# 数据集总览问题报告",
        "",
        f"视频 {stats['videos']} 个 | 帧 {stats['frames']} 张 | 相机 {stats['cameras']} 个 | 问题/跳过 {len(problems)} 条",
        "",
    ]
    lines += sorted(problems)
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = args.output_dir / "videos"
    frames_dir = args.output_dir / "frames"
    problems: list[str] = []
    stats = {"cameras": 0, "videos": 0, "frames": 0}

    mask_seq_dirs = sorted(
        (path for path in args.mask_root.glob(args.seq_glob) if path.is_dir()),
        key=lambda path: natural_sort_key(path.name),
    )
    processed_seqs = set()
    for seq_dir in mask_seq_dirs:
        rel_seq = seq_dir.relative_to(args.mask_root)
        processed_seqs.add(rel_seq.as_posix())
        dataset_seq_dir = args.dataset_root / rel_seq
        if not dataset_seq_dir.is_dir():
            problems.append(f"[无RGB序列] {rel_seq}: dataset-root 下不存在")
            continue

        cam_dirs = sorted(
            (path for path in seq_dir.glob(args.cam_name) if path.is_dir()),
            key=lambda path: natural_sort_key(path.name),
        )
        if not cam_dirs:
            problems.append(f"[未推理] {rel_seq}: 没有匹配 {args.cam_name!r} 的相机目录")
            continue

        # dataset 侧有、但 mask 侧没有预测目录的相机
        dataset_cam_names = {
            path.name for path in dataset_seq_dir.glob(args.cam_name) if path.is_dir()
        }
        mask_cam_names = {cam_dir.name for cam_dir in cam_dirs}
        for cam_name in sorted(dataset_cam_names - mask_cam_names, key=natural_sort_key):
            problems.append(f"[未推理] {rel_seq / cam_name}: 没有 {args.mask_name} 目录")

        for cam_dir in cam_dirs:
            rel_cam = cam_dir.relative_to(args.mask_root)
            mask_dir = cam_dir / args.mask_name
            if not mask_dir.is_dir():
                problems.append(f"[未推理] {rel_cam}: 没有 {args.mask_name} 目录")
                continue
            stats["cameras"] += 1
            process_camera(
                rel_cam=rel_cam,
                dataset_cam_dir=dataset_seq_dir / cam_dir.name,
                mask_dir=mask_dir,
                image_name=args.image_name,
                videos_dir=videos_dir,
                frames_dir=frames_dir,
                fps=args.fps,
                problems=problems,
                stats=stats,
            )

    # dataset-root 下有、但 mask-root 下完全没有预测的序列
    for seq_dir in args.dataset_root.glob(args.seq_glob):
        if seq_dir.is_dir() and seq_dir.relative_to(args.dataset_root).as_posix() not in processed_seqs:
            problems.append(f"[未推理] 序列 {seq_dir.relative_to(args.dataset_root)}")

    write_report(args.output_dir, stats, problems)
    logging.info(
        "完成 | 视频 %d 个 | 帧 %d 张 | 相机 %d 个 | 问题 %d 条 | 报告: %s",
        stats["videos"], stats["frames"], stats["cameras"], len(problems),
        args.output_dir / "report.txt",
    )


if __name__ == "__main__":
    main()
