"""多视角双手 Memory 模型的序列 Loss。"""

import torch

from projects.dual_hand_memory.losses import LOSS_NAMES, DualHandMemoryLoss

class MultiViewDualHandMemoryLoss(DualHandMemoryLoss):
    """
    计算多视角双手序列 Loss。

    输入：
        frame_outputs:
            长度为 T 的 list，每帧左右手预测的 batch 维均为 B*V。
        left_masks:
            Tensor[B,V,T,1,H,W]。
        right_masks:
            Tensor[B,V,T,1,H,W]。
        sample_indices:
            validation 时需要计算哪些 batch 样本的 loss；训练时为 None。

    因此先将 GT 转换为：
        [B*V, T, 1, H, W]

    之后完全复用 DualHandMemoryLoss。
    """
    def forward(
        self,
        frame_outputs, 
        left_masks,
        right_masks,
        sample_indices=None,
    ):
        num_views = left_masks.size(1)

        # [B,V,T,1,H,W] -》 [B*V, T, 1, H, W]
        left_masks = left_masks.flatten(0, 1)
        right_masks = right_masks.flatten(0, 1)

        if sample_indices is not None:
            # 当前 batch 中第几个 clip
            clip_indices = torch.as_tensor(
                sample_indices,
                dtype=torch.long,
                device=left_masks.device,
            )
            # 这个 clip 中第几个视角
            view_indices = torch.arange(
                num_views,
                device=left_masks.device,
            )

            clip_indices = clip_indices.unsqueeze(1)  # [B] -> [B,1]
            view_indices = view_indices.unsqueeze(0)    # [V] -> [1,V]

            # [B, 1] + [1, V] -> [B, V] -flatten-> [B*V]
            sample_indices = (
                clip_indices * num_views
                + view_indices
            ).flatten()


        return super().forward(
            frame_outputs,
            left_masks,
            right_masks,
            sample_indices,
        )
