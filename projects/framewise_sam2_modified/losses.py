"""计算 SAM2Modified 左右手分支的 mask、IoU 和object_score_logits。"""
import torch
import torch.nn.functional as F



def dice_loss_from_logits(
    logits: torch.Tensor, 
    targets: torch.Tensor, 
) -> torch.Tensor:
    """
    计算dice_loss

    input:
        logits: [B, 1, H(768), W(768)] 
        targets:[B, 1, H(768), W(768)] 每个点为： 0.0， 1.0

    output: 
    [] 大小的 tensor
    """

    probs = torch.sigmoid(logits) # probs: [B, 1, H(768), W(768)] 每个点在 0-1 之间
    targets = targets.float() # [B, 1, H(768), W(768)] 每个点为： 0.0， 1.0

    intersection = (probs * targets).sum(dim=(1, 2, 3)) # [B]
    total_area = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) 
    
    dice = (2.0 * intersection + 1e-6) / (total_area + 1e-6)
    dice_loss = 1.0 - dice
    return dice_loss.mean()

def iou_target_from_logits(
    logits: torch.Tensor, 
    targets: torch.Tensor,
) -> torch.Tensor:
    """只计算iou的值，暂时与loss无关"""

    predicted_masks = (logits > 0).float()
    targets = targets.float()

    intersection = (predicted_masks * targets).sum(dim=(1, 2, 3))
    union = predicted_masks.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    
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

    object_exists = (
        target_masks
        .flatten(start_dim=1)
        .amax(dim=1)
        .gt(0.5)
        .to(dtype=object_score_logits.dtype)
    )

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
                "high_res_multimasks": left_outputs[1],
                "ious": left_outputs[2],                    [B]
                "low_res_masks": left_outputs[3],
                "high_res_masks": left_outputs[4],          [B, 1, 768, 768]
                "obj_ptr": left_outputs[5],
                "object_score_logits": left_outputs[6],
            }
    
    target_masks: [B, 1, 768, 768]

    Outputs:
    total_loss: tensor []

    loss_details: dict，包含 BCE、Dice、IoU 和 object score loss。
    """

    logits = hand_outputs["high_res_masks"]
    predicted_ious = hand_outputs["ious"]
    target_masks = target_masks.float()

    # 计算dice_loss和 bce_loss
    bce_loss = F.binary_cross_entropy_with_logits(logits, target_masks)
    dice_loss = dice_loss_from_logits(logits, target_masks)

    # 计算iou_loss: 模型对iou的预测
    target_ious = iou_target_from_logits(logits.detach(), target_masks)
    iou_loss = F.mse_loss(predicted_ious.view(-1), target_ious.view(-1))

    # 计算 object_score_loss
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
