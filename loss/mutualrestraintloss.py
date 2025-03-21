import torch
from torch import nn


class MutualRestraintLoss(nn.Module):
    def __init__(self, alpha=0.5):
        super(MutualRestraintLoss, self).__init__()
        self.alpha = alpha
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, outputs, targets):
        # 分离正常和疾病的预测
        normal_pred = outputs[:, 0]
        disease_pred = outputs[:, 1:]
        
        # 分离正常和疾病的标签
        normal_target = targets[:, 0]
        disease_target = targets[:, 1:]
        
        # 基础分类损失
        normal_loss = self.bce(normal_pred, normal_target)
        disease_loss = self.bce(disease_pred, disease_target)
        
        # 使用概率空间计算互斥损失
        normal_prob = torch.sigmoid(normal_pred)
        disease_prob = torch.sigmoid(disease_pred)
        mutual_loss = torch.mean(normal_prob * torch.mean(disease_prob, dim=1))
        
        # 确保所有损失值为非负
        normal_loss = torch.clamp(normal_loss, min=0.0)
        disease_loss = torch.clamp(disease_loss, min=0.0)
        mutual_loss = torch.clamp(mutual_loss, min=0.0)
        
        # 组合损失
        total_loss = normal_loss + disease_loss + self.alpha * mutual_loss
        
        # 安全检查，确保损失值有效
        if not torch.isfinite(total_loss):
            print("警告: 损失值无效！使用替代损失")
            return normal_loss + disease_loss  # 回退到基础损失
            
        return total_loss