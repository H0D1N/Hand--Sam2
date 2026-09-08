"""在现有 Dataset 上运行 Framewise 或 Memory 双手分割推理。"""

from __future__ import annotations

import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from inference.builder import build_model
from inference.dataset import build_frame_dataset, build_loader
from projects.framewise_sam2_modified.dataset import build_center_point_prompt
from projects.framewise_sam2_modified.utils import configure_runtime
from sam2.modeling.sam2_utils import get_next_point


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
    common.add_argument("--output-dir", type=Path, help="输出根目录；默认写回原图对应的相机目录。")
    common.add_argument("--batch-size", type=int)
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

    args = parser.parse_args()
    args.prediction_dir_name = args.prediction_dir_name or f"{args.model}_prediction"
    if args.batch_size is None:
        args.batch_size = 1 if args.model == "memory" else 2
    if args.batch_size < 1 or (args.model == "memory" and args.batch_size != 1):
        parser.error("batch-size 必须为正整数，memory 推理仅支持 --batch-size 1")
    if args.dataset in {"dexycb", "mixed"} and args.dex_ycb_root is None:
        parser.error(f"--dataset {args.dataset} 需要 --dex-ycb-root")
    return args


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
    left_logits = F.interpolate(left_logits.float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    right_logits = F.interpolate(right_logits.float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    left_mask = left_logits > 0
    right_mask = right_logits > 0

    label = torch.zeros(size, dtype=torch.uint8, device=left_logits.device)
    label[left_mask] = 2
    label[right_mask] = 1
    overlap = left_mask & right_mask
    label[overlap & (left_logits >= right_logits)] = 1

    mask_path = prediction_path(image_path, prediction_dir_name, output_dir, dataset_root)
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
                output_dir=getattr(args, "output_dir", None),
                dataset_root=batch["dataset_root"][index],
            )
            count += 1
        if step % args.log_interval == 0 or step == len(loader):
            logging.info("framewise | %d/%d batches | %d images", step, len(loader), count)
    return count


@torch.inference_mode()
def predict_memory(model, dataset, args) -> int:
    count = 0
    for stream_id, stream in dataset.streams.items():
        loader = build_loader(dataset, args, sample_indices=stream["sample_indices"])
        state = model.init_state(loader)
        for hand in ("left", "right"):
            mask = state["first_frame"][f"{hand}_mask"]
            if args.prompt_mode == "point":
                coords, labels = get_next_point(gt_masks=mask.bool(), pred_masks=None, method=model.pt_sampling_for_eval)
                model.add_points(state, hand, {"point_coords": coords, "point_labels": labels})
                del coords, labels
            else:
                model.add_mask(state, hand, mask)
        del mask
        with _autocast(args):
            for frame_idx, outputs, frame_info in model.propagate_in_video(state):
                save_prediction(
                    image_path=frame_info["image_path"], original_size=frame_info["original_size"],
                    left_logits=outputs["left"]["pred_masks_high_res"],
                    right_logits=outputs["right"]["pred_masks_high_res"],
                    prediction_dir_name=args.prediction_dir_name,
                    output_dir=getattr(args, "output_dir", None),
                    dataset_root=frame_info["dataset_root"],
                )
                count += 1
                if (frame_idx + 1) % args.log_interval == 0 or frame_idx + 1 == state["num_frames"]:
                    logging.info("memory | %s | %d/%d frames | %d images", stream_id, frame_idx + 1, state["num_frames"], count)
                del outputs
        del state, loader
    return count


def run_prediction(model, dataset, args) -> int:
    """统一推理入口；memory 按 stream 逐帧传播。"""
    if args.model == "framewise":
        return predict_framewise(model, build_loader(dataset, args), args)
    return predict_memory(model, dataset, args)


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    configure_runtime(torch.device(args.device), use_tf32=True)
    model = build_model(args)
    dataset = build_frame_dataset(args)
    count = run_prediction(model, dataset, args)
    logging.info("Prediction complete | %d images", count)


if __name__ == "__main__":
    main()
