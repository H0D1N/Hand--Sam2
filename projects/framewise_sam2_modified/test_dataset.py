"""逐个读取 Dataset 样本，并把读取失败的图片写入 bad_samples.txt。"""

from __future__ import annotations

import argparse
from pathlib import Path

from .dataset import MultiServerDualHandDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 framewise Dataset 坏样本")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-names", nargs="+", default=None)
    parser.add_argument("--test-seq-count", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="默认输出到 <dataset-root>/bad_samples.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = args.output or args.dataset_root / "bad_samples.txt"
    bad_image_paths: set[str] = set()
    checked_count = 0

    for split in ["train", "val"]:
        dataset = MultiServerDualHandDataset(
            dataset_root=args.dataset_root,
            split=split,
            test_seq_count=args.test_seq_count,
            image_size=args.image_size,
            use_augmentation=False,
            dataset_names=args.dataset_names,
        )

        print(f"checking {split}: {len(dataset)} samples", flush=True)

        for index in range(len(dataset)):
            checked_count += 1
            try:
                dataset[index]
            except Exception as exc:
                sample = dataset.samples[index]
                image_path = str(sample["image_path"])
                bad_image_paths.add(image_path)

                print(f"BAD [{split}] index={index}", flush=True)
                print(f"  image: {image_path}", flush=True)
                print(f"  mask:  {sample['mask_path']}", flush=True)
                print(
                    f"  error: {type(exc).__name__}: {exc}",
                    flush=True,
                )

            if (index + 1) % 1000 == 0:
                print(
                    f"  checked {index + 1}/{len(dataset)}",
                    flush=True,
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(f"{path}\n" for path in sorted(bad_image_paths)),
        encoding="utf-8",
    )

    print(
        f"finished, checked samples: {checked_count}, "
        f"bad samples: {len(bad_image_paths)}",
        flush=True,
    )
    print(f"saved to: {output_path}", flush=True)


if __name__ == "__main__":
    main()
