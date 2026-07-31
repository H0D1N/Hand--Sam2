"""计算 SAM2Modified 左右手分支的 mask、IoU 和object_score_logits。"""
import torch
import torch.nn.functional as F



def object_targets_from_masks(
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """
    根据每张 GT mask 是否非空，返回 [B] bool。

    Output: tensor([True, False, True, ...])
    """
    return target_masks.gt(0.5).flatten(1).any(dim=1)

def masked_batch_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    只保留 mask=True 样本的 loss，然后按整个 batch 平均。
    达到 GT 为空，只计算 obj_loss , 有值时再计算 dice_loss...
    """
    return (values * mask.to(values.dtype)).mean()

def dice_loss_from_logits(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
    reduction: str = "mean",
) -> torch.Tensor:
    """
    计算每个样本、每个候选 mask 的 Dice loss。

    input:
        logits: [B, M, H(768), W(768)]
        targets:[B, 1, H(768), W(768)] 每个点为： 0.0， 1.0

    output: 
    reduction=none: [B, M] 大小的 tensor
    reduction=mean: [] 大小的 标量
    """
    # probs: [B, M, H(768), W(768)]
    # 每个点在 0-1 之间
    probs = torch.sigmoid(logits)

    # [B, M, H(768), W(768)] 
    # 每个点为： 0.0， 1.0
    # 维度: [B, 1, H, W] -> [B, M, H, W]
    targets = targets.float().expand_as(logits) 

    intersection = (probs * targets).sum(dim=(-2, -1)) # [B, M]
    total_area = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
    
    dice = (2.0 * intersection + 1e-6) / (total_area + 1e-6)
    dice_loss = 1.0 - dice

    if reduction == "none":
        return dice_loss
    return dice_loss.mean()

def iou_target_from_logits(
    logits: torch.Tensor, 
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    计算每个候选 mask 与 GT 的真实 IoU。

    logits:  [B, M, H, W]
    targets: [B, 1, H, W]

    返回: [B, M]
    """

    predicted_masks = (logits > 0).float()
    targets = targets.float().expand_as(logits)

    intersection = (predicted_masks * targets).sum(dim=(-2, -1))
    union = predicted_masks.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1)) - intersection
    
    return (intersection + 1e-6) / (union + 1e-6)

def object_score_loss_from_logits(
    object_score_logits: torch.Tensor,
    target_masks: torch.Tensor,
) -> torch.Tensor:
    """
    根据 GT mask 是否为空，监督模型预测目标是否存在。

    object_score_logits: [B, 1] 或 [B]
    target_masks: [B, 1, H, W]
    """

    object_exists = object_targets_from_masks(
        target_masks
    ).to(object_score_logits.dtype)

    return F.binary_cross_entropy_with_logits(
        object_score_logits.reshape(-1),
        object_exists,
    )

def one_hand_loss(
    hand_outputs: dict[str, torch.Tensor],
    target_masks: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    iou_weight: float = 0.1,
    object_score_weight: float = 1.0,
):
    """
    计算单只手的全部loss

    Input:
    hand_outputs: {
                "low_res_multimasks": left_outputs[0],
                "high_res_multimasks": left_outputs[1],     # [B, M, H, W]
                "ious": left_outputs[2],                    # [B, M]
                "low_res_masks": left_outputs[3],
                "high_res_masks": left_outputs[4],
                "obj_ptr": left_outputs[5],
                "object_score_logits": left_outputs[6],
            }
    
    target_masks: [B, 1, 768, 768]

    Outputs:
    total_loss: tensor []

    loss_details: dict，包含 BCE、Dice、IoU 和 object score loss。
    """

    logits = hand_outputs["high_res_multimasks"]
    predicted_ious = hand_outputs["ious"]
    target_masks = target_masks.float()

    # 计算 dice和 bce
        # B * M张预测掩码，每张图的每个候选分别计算 BCE
        # [B, M]
    bce_per_mask = F.binary_cross_entropy_with_logits(
        logits,
        target_masks.expand_as(logits),
        reduction="none",
    ).mean(dim=(-2, -1))

        # B * M张预测掩码，每张图的每个候选分别计算 Dice
        # [B, M]
    dice_per_mask = dice_loss_from_logits(
        logits,
        target_masks,
        reduction="none",
    )

        # take the mask indices with the smallest bce + dice loss for back propagation
    loss_combo = (
        bce_weight * bce_per_mask
        + dice_weight * dice_per_mask
    )
        # 每张图片各自选出 loss 最小的候选
        # shape: [B]
    best_loss_inds = torch.argmin(loss_combo, dim=-1)
    batch_inds = torch.arange(loss_combo.size(0), device=loss_combo.device)
    
        # BCE 和 Dice 必须取同一个候选
        # shape: [B]
    bce_per_sample = bce_per_mask[batch_inds, best_loss_inds]
    dice_per_sample = dice_per_mask[batch_inds, best_loss_inds]



    # 计算 iou_loss
        # [B, M]：三个候选各自的真实 IoU
    target_ious = iou_target_from_logits(logits.detach(), target_masks)

        # [B, M]：三个预测 IoU 分别拟合各自的真实 IoU
    iou_per_mask = F.mse_loss(
        predicted_ious.reshape_as(target_ious),
        target_ious,
        reduction="none",
    )

        # [B]：三个候选的 IoU loss 取平均后都参与训练
    iou_per_sample = iou_per_mask.mean(dim=1)

    # tensor([True, False, True, ...])
    # GT 为空时，不计算 mask、Dice 和 IoU loss
    object_targets = object_targets_from_masks(target_masks)

    bce_loss = masked_batch_mean(bce_per_sample, object_targets)
    dice_loss = masked_batch_mean(dice_per_sample, object_targets)
    iou_loss = masked_batch_mean(iou_per_sample, object_targets)

    # Object score 无论 GT 是否为空都需要训练
    object_score_loss = object_score_loss_from_logits(
        object_score_logits=hand_outputs["object_score_logits"],
        target_masks=target_masks,
    )

    total_loss = (
        bce_weight * bce_loss
        + dice_weight * dice_loss
        + iou_weight * iou_loss
        + object_score_weight * object_score_loss
    )

    loss_details = {
        "bce": bce_loss,
        "dice": dice_loss,
        "iou": iou_loss,
        "object_score": object_score_loss,
    }

    return total_loss, loss_details

def dual_hand_loss(
    model_output,
    left_masks: torch.Tensor,
    right_masks: torch.Tensor,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    iou_weight: float = 0.1,
    object_score_weight: float = 1.0,
):
    """
    分别计算双手的loss，之后取平均
    """

    left_loss, left_details = one_hand_loss(
        hand_outputs=model_output["left"],
        target_masks=left_masks,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        iou_weight=iou_weight,
        object_score_weight=object_score_weight,
    )

    right_loss, right_details = one_hand_loss(
        hand_outputs=model_output["right"],
        target_masks=right_masks,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        iou_weight=iou_weight,
        object_score_weight=object_score_weight,
    )

    total_loss = (left_loss + right_loss) / 2.0
    loss_details = {"left":left_details, "right": right_details}

    return total_loss, loss_details
