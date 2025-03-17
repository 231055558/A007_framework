from torch import nn
import torch
import torch.nn.functional as F


class DiseaseClassificationLoss(nn.Module):
    def __init__(self, pos_weights, device='cuda'):
        super(DiseaseClassificationLoss, self).__init__()
        self.pos_weights = pos_weights.to(device)

    def forward(self, predictions, targets):
        # 基础BCE损失
        bce_loss = F.binary_cross_entropy_with_logits(
            predictions, targets,
            pos_weight=self.pos_weights,
            reduction='none'
        )

        # Focal Loss 项
        probs = torch.sigmoid(predictions)
        focal_weight = (1 - probs) * targets + probs * (1 - targets)
        focal_weight = focal_weight ** 2

        # 组合损失
        loss = bce_loss * focal_weight

        return loss.mean()