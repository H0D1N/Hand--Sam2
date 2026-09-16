"""共享的双手预测 mask 编码与输出路径。"""

from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image


def prediction_path(
    image_path: str,
    directory_name: str,
    output_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> Path:
    """把 dataset_root 替换为 output_dir，并保留 sequence/camera 层级。"""
    image_path = Path(image_path)
    image_dir = image_path.parent
    image_dir_name = image_dir.name.lower()
    if image_dir_name.startswith("rgb") or image_dir_name in {"color", "images"}:
        image_dir = image_dir.parent
    output_root = image_dir
    if output_dir is not None:
        if dataset_root is None:
            raise ValueError("指定 output_dir 时需要 dataset_root")
        output_root = Path(output_dir) / image_dir.relative_to(dataset_root)
    return output_root / directory_name / image_path.with_suffix(".png").name


def save_prediction(
    *,
    image_path: str,
    original_size: tuple[int, int],
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    prediction_dir_name: str,
    output_dir: str | Path | None = None,
    dataset_root: str | Path | None = None,
) -> None:
    size = tuple(int(value) for value in original_size)
    left_logits = F.interpolate(
        left_logits.float(), size=size, mode="bilinear", align_corners=False
    )[0, 0]
    right_logits = F.interpolate(
        right_logits.float(), size=size, mode="bilinear", align_corners=False
    )[0, 0]
    left_mask = left_logits > 0
    right_mask = right_logits > 0

    label = torch.zeros(size, dtype=torch.uint8, device=left_logits.device)
    label[left_mask] = 2
    label[right_mask] = 1
    overlap = left_mask & right_mask
    label[overlap & (left_logits >= right_logits)] = 1

    mask_path = prediction_path(
        image_path, prediction_dir_name, output_dir, dataset_root
    )
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label.cpu().numpy()).save(mask_path)
