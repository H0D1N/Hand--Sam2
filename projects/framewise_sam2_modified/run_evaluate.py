"""Evaluate a trained framewise model on MultiServer and DexYCB separately."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from .builder import create_sam2_modified_tiny, inject_sam2_modified_adapters
from .dataset import (
    DexYCBDataset,
    MultiServerDualHandDataset,
    build_center_point_prompt,
    collate_batch,
)
from .frame_sampler import FramesPerSecondSampler
from .trainer import run_validation_epoch
from .utils import configure_runtime, dump_json, set_seed


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained framewise dual-hand model separately on "
            "MultiServer and DexYCB."
        )
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset-names", nargs="+", required=True)
    parser.add_argument(
        "--dex-ycb-root",
        "--dex_ycb_root",
        dest="dex_ycb_root",
        type=Path,
        required=True,
    )
    parser.add_argument("--test-seq-count", type=int, default=3)
    parser.add_argument("--frames-per-second", type=float, default=3.0)
    parser.add_argument(
        "--prompt-mode",
        choices=("point", "none"),
        required=True,
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable-tf32", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--skip-visualizations", action="store_true")
    parser.add_argument("--log-interval", type=int, default=50)

    args = parser.parse_args(argv)

    if args.test_seq_count < 1:
        parser.error("--test-seq-count 必须大于 0")
    if args.frames_per_second <= 0:
        parser.error("--frames-per-second 必须大于 0")
    if args.batch_size < 1:
        parser.error("--batch-size 必须大于 0")
    if args.num_workers < 0:
        parser.error("--num-workers 不能小于 0")
    if args.prefetch_factor < 1:
        parser.error("--prefetch-factor 必须大于 0")
    if args.log_interval < 1:
        parser.error("--log-interval 必须大于 0")

    return args


def load_trained_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, dict, dict]:
    """Rebuild the model from the configuration stored in framewise best.pt."""

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state" not in checkpoint or "args" not in checkpoint:
        raise KeyError(
            "Framewise checkpoint must contain 'model_state' and 'args'."
        )

    train_args = checkpoint["args"]
    if not isinstance(train_args, dict):
        raise TypeError("checkpoint['args'] 必须是字典")

    model = create_sam2_modified_tiny(
        image_size=train_args.get("image_size", 768),
    )
    inject_sam2_modified_adapters(
        model=model,
        use_image_adapter=train_args.get("use_image_adapter", False),
        use_decoder_adapter=train_args.get("use_decoder_adapter", False),
        adapter_dim=train_args.get("adapter_dim", 64),
        adapter_dropout=train_args.get("adapter_dropout", 0.1),
        adapter_init_scale=train_args.get("adapter_init_scale", 1e-3),
    )
    model.load_state_dict(checkpoint["model_state"], strict=True)
    model.to(device).eval()

    checkpoint_metadata = {
        "epoch": checkpoint.get("epoch"),
        "best_val_iou": checkpoint.get("best_val_iou"),
    }
    del checkpoint

    return model, checkpoint_metadata, train_args


def build_eval_datasets(args: argparse.Namespace, image_size: int) -> dict:
    """Build one combined MultiServer domain and one DexYCB domain."""

    return {
        "MultiServer": MultiServerDualHandDataset(
            dataset_root=args.dataset_root,
            split="val",
            test_seq_count=args.test_seq_count,
            image_size=image_size,
            use_augmentation=False,
            dataset_names=args.dataset_names,
        ),
        "DexYCB": DexYCBDataset(
            dataset_root=args.dex_ycb_root,
            split="val",
            setup="s0",
            image_size=image_size,
            use_augmentation=False,
        ),
    }


def build_eval_loader(
    dataset,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, FramesPerSecondSampler]:
    sampler = FramesPerSecondSampler(
        dataset=dataset,
        frames_per_second=args.frames_per_second,
        shuffle=False,
        seed=args.seed,
    )

    if len(sampler) == 0:
        raise ValueError("评估 Sampler 没有选出任何样本")

    loader_kwargs = {
        "dataset": dataset,
        "batch_size": args.batch_size,
        "sampler": sampler,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_batch,
        "drop_last": False,
    }
    if args.num_workers > 0:
        loader_kwargs.update(
            {
                "persistent_workers": True,
                "prefetch_factor": args.prefetch_factor,
            }
        )

    return DataLoader(**loader_kwargs), sampler


def build_validation_args(
    args: argparse.Namespace,
    train_args: dict,
    output_dir: Path,
) -> SimpleNamespace:
    return SimpleNamespace(
        output_dir=output_dir,
        skip_visualizations=args.skip_visualizations,
        channels_last=args.channels_last,
        multimask_output=train_args.get("multimask_output", False),
        bce_weight=train_args.get("bce_weight", 1.0),
        dice_weight=train_args.get("dice_weight", 1.0),
        iou_weight=train_args.get("iou_weight", 0.1),
        object_score_weight=train_args.get("object_score_weight", 1.0),
        log_interval=args.log_interval,
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(
        device,
        use_tf32=not args.disable_tf32,
        channels_last=args.channels_last,
    )

    model, checkpoint_metadata, train_args = load_trained_model(
        args.checkpoint,
        device,
    )
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    trained_with_point_prompt = bool(
        train_args.get("use_point_prompt", False)
    )
    if args.prompt_mode == "none" and trained_with_point_prompt:
        logging.warning(
            "Checkpoint was trained with point prompts; "
            "the no-prompt result is an ablation."
        )

    point_prompt_fn = (
        build_center_point_prompt
        if args.prompt_mode == "point"
        else None
    )
    datasets = build_eval_datasets(
        args,
        image_size=train_args.get("image_size", 768),
    )

    results = {}
    output_names = {
        "MultiServer": "multiserver",
        "DexYCB": "dexycb",
    }
    dataset_metadata = {
        "MultiServer": {
            "split": "val",
            "split_protocol": "last_sequences_per_dataset",
            "test_seq_count": args.test_seq_count,
            "dataset_names": list(args.dataset_names),
        },
        "DexYCB": {
            "split": "val",
            "setup": "s0",
        },
    }

    for dataset_name, dataset in datasets.items():
        loader, sampler = build_eval_loader(dataset, args, device)
        dataset_output_dir = args.output_dir / output_names[dataset_name]
        validation_args = build_validation_args(
            args,
            train_args,
            dataset_output_dir,
        )

        logging.info(
            "%s | raw_samples=%d | sampled_samples=%d | streams=%d | "
            "prompt=%s",
            dataset_name,
            len(dataset),
            len(sampler),
            len(dataset.streams),
            args.prompt_mode,
        )

        metrics = run_validation_epoch(
            model=model,
            loader=loader,
            device=device,
            epoch=0,
            args=validation_args,
            point_prompt_fn=point_prompt_fn,
        )
        results[dataset_name] = {
            **dataset_metadata[dataset_name],
            "raw_samples": len(dataset),
            "sampled_samples": len(sampler),
            "num_streams": len(dataset.streams),
            "metrics": {
                name: float(value)
                for name, value in metrics.items()
            },
        }

        logging.info(
            "%s complete | iou=%.4f | dice=%.4f",
            dataset_name,
            metrics["iou"],
            metrics["dice"],
        )

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint_metadata["epoch"],
        "checkpoint_best_val_iou": checkpoint_metadata["best_val_iou"],
        "frames_per_second": args.frames_per_second,
        "prompt_mode": args.prompt_mode,
        "prompt_source": (
            "ground_truth_mask_center"
            if args.prompt_mode == "point"
            else None
        ),
        "trained_with_point_prompt": trained_with_point_prompt,
        "datasets": results,
    }
    dump_json(summary, args.output_dir / "metrics.json")
    logging.info("Evaluation results: %s", args.output_dir / "metrics.json")


if __name__ == "__main__":
    main()
