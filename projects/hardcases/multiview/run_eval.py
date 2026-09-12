"""Evaluate an initialized or trained multi-view dual-hand Memory model."""

import logging

import torch

from evaluation.model_loader import load_evaluation_model
from projects.dual_hand_multiview.builder import build_sam2_multiview_dual_hand_memory_tiny
from projects.dual_hand_multiview.losses import MultiViewDualHandMemoryLoss
from projects.dual_hand_multiview.trainer import run_validation_epoch
from projects.framewise_sam2_modified.utils import configure_runtime, dump_json, set_seed
from projects.hardcases.evaluation_common import build_multiview_zero_shot_loader, parse_args
from training.model.sam2_multiview_dual_hand_memory import PromptRequest


def build_prompt_request(args) -> PromptRequest:
    """将共享评测协议转换为 MultiView PromptRequest。"""
    if args.prompt_mode == "auto":
        return PromptRequest(mode="auto")

    return PromptRequest(
        mode=args.prompt_mode,
        start_frame_idx=0,
        prompt_frame_indices=(0,),
        correction_frame_indices=tuple(args.correction_frame_indices),
        num_correction_points_per_frame=args.num_correction_pt_per_frame,
        add_correction_frames_as_cond=args.add_all_frames_to_correct_as_cond,
    )


def main() -> None:
    args = parse_args()

    if args.framewise_checkpoint is not None:
        raise ValueError(
            "MultiView 只能从 --sam-checkpoint、--memory-checkpoint "
            "或完整的 --multiview-checkpoint 加载"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    set_seed(args.seed)
    device = torch.device(args.device)
    configure_runtime(device, use_tf32=not args.disable_tf32)

    if args.multiview_checkpoint is not None:
        model = load_evaluation_model(
            "multiview", args.multiview_checkpoint, device
        )
    else:
        model = build_sam2_multiview_dual_hand_memory_tiny(
            sam_checkpoint=args.sam_checkpoint,
            memory_checkpoint=args.memory_checkpoint,
            device=device,
            mode="eval",
            image_size=args.image_size,
            num_latents=args.num_latents,
            num_aggregator_layers=args.num_aggregator_layers,
            num_distributor_layers=args.num_distributor_layers,
            num_correction_pt_per_frame=args.num_correction_pt_per_frame,
            add_all_frames_to_correct_as_cond=args.add_all_frames_to_correct_as_cond,
        )

    # 数据缩放必须使用 checkpoint 重建出的模型尺寸。
    args.image_size = model.image_size
    val_loader = build_multiview_zero_shot_loader(args, device)
    loss_fn = MultiViewDualHandMemoryLoss(
        mask_loss_weight=args.mask_loss_weight,
        dice_loss_weight=args.dice_loss_weight,
        iou_loss_weight=args.iou_loss_weight,
        class_loss_weight=args.class_loss_weight,
    ).to(device)

    validation_metrics = run_validation_epoch(
        model=model,
        loss_fn=loss_fn,
        loader=val_loader,
        device=device,
        args=args,
        epoch=0,
        prompt_request=build_prompt_request(args),
        metric_start_frame=args.metric_start_frame,
    )
    overall = validation_metrics["overall"]
    epoch_metrics = {
        "epoch": 0.0,
        "lr": None,
        "train_loss": None,
        "validation": validation_metrics,
    }

    dump_json(
        {
            "best_val_iou": float(overall["iou"]),
            "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
            "epochs": [epoch_metrics],
        },
        args.output_dir / "metrics.json",
    )

    logging.info(
        "MULTIVIEW EVALUATION COMPLETE | "
        "val_loss=%.4f | val_iou=%.4f | val_dice=%.4f | "
        "obj_acc=%.4f | obj_precision=%.4f | "
        "obj_recall=%.4f | obj_f1=%.4f",
        overall["loss"],
        overall["iou"],
        overall["dice"],
        overall["object_accuracy"],
        overall["object_precision"],
        overall["object_recall"],
        overall["object_f1"],
    )


if __name__ == "__main__":
    main()
