"""在现有 Dataset 上运行 Framewise 或 Memory 双手分割推理。"""

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from projects.dual_hand_memory.dataset import ConsecutiveClipDataset, collate_clip_batch
from projects.framewise_sam2_modified.builder import (
    create_sam2_modified_tiny,
    inject_sam2_modified_adapters,
)
from projects.framewise_sam2_modified.dataset import (
    CombinedStreamDataset,
    DexYCBDataset,
    MultiServerDualHandDataset,
    build_center_point_prompt,
    collate_batch,
)
from projects.framewise_sam2_modified.utils import configure_runtime
from training.model.sam2_dual_hand_memory import SAM2DualHandMemory
from training.model.sam2_modified import SAM2Modified


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = REPO_ROOT / "framewise_data/dataset"
DEFAULT_DATASET_NAMES = (
    "xingyi_4-5090_oak150-100output",
    "wuwen_4-5090_release-0623-compressed",
    "tencent_4-5090_7.5",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="双手分割推理")

    common = parser.add_argument_group("common")
    common.add_argument("--model", choices=("framewise", "memory"), required=True)
    common.add_argument("--model-checkpoint", type=Path, required=True)
    common.add_argument("--prediction-dir-name", help="每个图像目录对应的推理文件夹名；默认 <model>_prediction。")
    common.add_argument("--batch-size", type=int, default=2)
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    common.add_argument("--amp", action="store_true")
    common.add_argument("--log-interval", type=int, default=50)

    data = parser.add_argument_group("dataset")
    data.add_argument("--dataset", choices=("multiserver", "dexycb", "mixed"), default="multiserver")
    data.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    data.add_argument("--dataset-names", nargs="+", default=DEFAULT_DATASET_NAMES)
    data.add_argument("--test-seq-count", type=int, default=3)
    data.add_argument("--dex-ycb-root", type=Path)
    data.add_argument("--dex-ycb-setup", default="s0")

    framewise = parser.add_argument_group("framewise")
    framewise.add_argument("--use-point-prompt", action="store_true")

    memory = parser.add_argument_group("memory")
    memory.add_argument("--prompt-mode", choices=("mask", "point", "auto"), default="mask")
    memory.add_argument("--clip-length", type=int, default=8)
    memory.add_argument("--clip-stride", type=int)

    args = parser.parse_args()
    args.prediction_dir_name = args.prediction_dir_name or f"{args.model}_prediction"
    args.clip_stride = args.clip_stride or args.clip_length
    if args.dataset in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error(f"--dataset {args.dataset} 需要 --dex-ycb-root")
    return args


def build_model(args: argparse.Namespace) -> torch.nn.Module:
    checkpoint = torch.load(args.model_checkpoint, map_location="cpu", weights_only=False)
    saved_args = checkpoint["args"]
    model_cls = SAM2DualHandMemory if args.model == "memory" else SAM2Modified
    model = create_sam2_modified_tiny(
        image_size=saved_args["image_size"],
        model_cls=model_cls,
    )
    inject_sam2_modified_adapters(
        model=model,
        use_image_adapter=saved_args["use_image_adapter"],
        use_decoder_adapter=saved_args["use_decoder_adapter"],
        adapter_dim=saved_args["adapter_dim"],
        adapter_dropout=saved_args["adapter_dropout"],
        adapter_init_scale=saved_args["adapter_init_scale"],
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model = model.to(args.device).eval()
    args.image_size = saved_args["image_size"]
    logging.info("Loaded %s checkpoint: %s", args.model, args.model_checkpoint)
    return model


def build_frame_dataset(args: argparse.Namespace):
    datasets = []
    if args.dataset in {"multiserver", "mixed"}:
        datasets.append(MultiServerDualHandDataset(
            dataset_root=args.dataset_root,
            split="val",
            test_seq_count=args.test_seq_count,
            image_size=args.image_size,
            use_augmentation=False,
            dataset_names=args.dataset_names,
        ))
    if args.dataset in {"dexycb", "mixed"}:
        datasets.append(DexYCBDataset(
            dataset_root=args.dex_ycb_root,
            split="val",
            setup=args.dex_ycb_setup,
            image_size=args.image_size,
            use_augmentation=False,
        ))
    return datasets[0] if len(datasets) == 1 else CombinedStreamDataset(datasets)


def build_loader(args: argparse.Namespace) -> DataLoader:
    dataset = build_frame_dataset(args)
    collate_fn = collate_batch
    if args.model == "memory":
        dataset = ConsecutiveClipDataset(
            dataset,
            clip_length=args.clip_length,
            clip_stride=args.clip_stride,
        )
        collate_fn = collate_clip_batch

    loader_args = dict(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.device(args.device).type == "cuda",
        collate_fn=collate_fn,
    )
    if args.num_workers > 0:
        loader_args.update(persistent_workers=True, prefetch_factor=2)
    return DataLoader(**loader_args)


def prediction_path(image_path: str, directory_name: str) -> Path:
    """例如 cam/rgb_undistort/0001.png -> cam/<directory_name>/0001.png。"""
    image_path = Path(image_path)
    image_dir = image_path.parent
    image_dir_name = image_dir.name.lower()
    if image_dir_name.startswith("rgb") or image_dir_name in {"color", "images"}:
        image_dir = image_dir.parent
    return image_dir / directory_name / image_path.with_suffix(".png").name


def save_prediction(
    *,
    image_path: str,
    original_size: tuple[int, int],
    left_logits: torch.Tensor,
    right_logits: torch.Tensor,
    prediction_dir_name: str,
) -> None:
    size = tuple(int(value) for value in original_size)
    left_logits = F.interpolate(left_logits.float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    right_logits = F.interpolate(right_logits.float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    left_mask = left_logits > 0
    right_mask = right_logits > 0

    label = torch.zeros(size, dtype=torch.uint8, device=left_logits.device)
    label[left_mask] = 1
    label[right_mask] = 2
    overlap = left_mask & right_mask
    label[overlap & (left_logits >= right_logits)] = 1

    mask_path = prediction_path(image_path, prediction_dir_name)
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(label.cpu().numpy()).save(mask_path)


def _autocast(args: argparse.Namespace):
    enabled = torch.device(args.device).type == "cuda" and args.amp
    return torch.amp.autocast("cuda") if enabled else nullcontext()


@torch.inference_mode()
def predict_framewise(model, loader, args) -> int:
    count = 0
    device = torch.device(args.device)
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)
        with _autocast(args):
            outputs = model.forward_single_image(
                images=images,
                left_point_inputs=build_center_point_prompt(left_masks) if args.use_point_prompt else None,
                right_point_inputs=build_center_point_prompt(right_masks) if args.use_point_prompt else None,
                mask_inputs=None,
                multimask_output=False,
            )
        for index in range(images.size(0)):
            save_prediction(
                image_path=batch["image_path"][index],
                original_size=batch["original_size"][index],
                left_logits=outputs["left"]["high_res_masks"][index:index + 1],
                right_logits=outputs["right"]["high_res_masks"][index:index + 1],
                prediction_dir_name=args.prediction_dir_name,
            )
            count += 1
        if step % args.log_interval == 0 or step == len(loader):
            logging.info("framewise | %d/%d batches | %d images", step, len(loader), count)
    return count


@torch.inference_mode()
def predict_memory(model, loader, args) -> int:
    count = 0
    device = torch.device(args.device)
    for step, batch in enumerate(loader, start=1):
        images = batch["image"].to(device, non_blocking=True)
        left_masks = batch["left_mask"].to(device, non_blocking=True)
        right_masks = batch["right_mask"].to(device, non_blocking=True)
        with _autocast(args):
            frame_outputs = model(
                images=images,
                left_masks=left_masks,
                right_masks=right_masks,
                prompt_mode=args.prompt_mode,
            )
        for sample_index in range(images.size(0)):
            for frame_index, outputs in enumerate(frame_outputs):
                save_prediction(
                    image_path=batch["image_path"][sample_index][frame_index],
                    original_size=batch["original_size"][sample_index][frame_index],
                    left_logits=outputs["left"]["pred_masks_high_res"][sample_index:sample_index + 1],
                    right_logits=outputs["right"]["pred_masks_high_res"][sample_index:sample_index + 1],
                    prediction_dir_name=args.prediction_dir_name,
                )
                count += 1
        if step % args.log_interval == 0 or step == len(loader):
            logging.info("memory | %d/%d batches | %d images", step, len(loader), count)
    return count


def run_prediction(model, loader, args) -> int:
    """统一推理入口；模型差异只在这里分发。"""
    if args.model == "framewise":
        return predict_framewise(model, loader, args)
    return predict_memory(model, loader, args)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    configure_runtime(torch.device(args.device), use_tf32=True)
    model = build_model(args)
    loader = build_loader(args)
    count = run_prediction(model, loader, args)
    logging.info("Prediction complete | %d images", count)


if __name__ == "__main__":
    main()
