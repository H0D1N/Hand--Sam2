import torch.nn as nn

from training.loss_fns import MultiStepMultiMasksAndIous
from training.trainer import CORE_LOSS_KEY


LOSS_NAMES = ("loss_mask", "loss_dice", "loss_iou", "loss_class",)

class DualHandMemoryLoss(nn.Module):
    """使用 SAM2 原版多轮 loss 计算左右手序列 loss。"""

    def __init__(
        self,
        mask_loss_weight=20.0,
        dice_loss_weight=1.0,
        iou_loss_weight=1.0,
        class_loss_weight=1.0,
    ):
        """创建一次原版 loss, 只在这里设置原版参数"""

        super().__init__()

        self.sam2_loss = MultiStepMultiMasksAndIous(
            weight_dict={
                "loss_mask": mask_loss_weight,
                "loss_dice": dice_loss_weight,
                "loss_iou": iou_loss_weight,
                "loss_class": class_loss_weight,
            },
            supervise_all_iou=True,
            iou_use_l1_loss=True,
            pred_obj_scores=True,
            focal_gamma_obj_score=0.0,
            focal_alpha_obj_score=-1.0,
        )

    def _compute_hand_loss(
        self,
        frame_outputs,
        target_masks,
        hand,
    ):
        # list[T]，每项是当前手在一帧上的完整 multistep 输出。
        hand_outputs = [
            frame_output[hand]
            for frame_output in frame_outputs
        ]

        # [B, T, 1, H, W] -> [T, B, H, W]
        hand_targets = target_masks.transpose(0, 1).squeeze(2)

        return self.sam2_loss(hand_outputs, hand_targets)

    def forward(self, frame_outputs, left_masks, right_masks):
        """计算双手序列的 SAM2 多轮 loss，并对左右手取平均。"""

        num_frames = left_masks.size(1)

        if num_frames == 0:
            raise ValueError("序列不能为空")

        if len(frame_outputs) != num_frames:
            raise ValueError(f"模型输出 {len(frame_outputs)} 帧，GT 包含 {num_frames} 帧")


        left_losses = self._compute_hand_loss(frame_outputs, left_masks, "left",)
        right_losses = self._compute_hand_loss(frame_outputs, right_masks, "right",)

        total_loss = (left_losses[CORE_LOSS_KEY] + right_losses[CORE_LOSS_KEY]) / 2.0

        loss_details = {
            "left": {
                name: left_losses[name]
                for name in LOSS_NAMES
            },
            "right": {
                name: right_losses[name]
                for name in LOSS_NAMES
            },
        }

        return total_loss, loss_details
