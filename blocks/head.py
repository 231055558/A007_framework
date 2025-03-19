import torch.nn.functional as F
import torch
import torch.nn as nn
from typing import Optional

from blocks.activation import build_activation


class AttentionFC(nn.Module):
    def __init__(self, in_features, num_classes):
        super(AttentionFC, self).__init__()
        self.fc_normal = nn.Linear(in_features, 1)
        self.fc_disease = nn.Linear(in_features, num_classes - 1)
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )
        self.num_classes = num_classes

    def forward(self, x):
        p_normal = self.fc_normal(x)

        attention_weights = self.attention(x)
        weighted_features = x * attention_weights

        p_disease = self.fc_disease(weighted_features)

        output = torch.cat([p_normal, p_disease], dim=1)
        return output



class VisionTransformerClsHead(nn.Module):
    """
    Vision Transformer classifier head.

    Args:
        num_classes (int): Number of categories.
        in_channels (int): Number of channels in the input feature map.
        hidden_dim (Optional[int]): Number of dimensions for the hidden layer.
                                    If None, no hidden layer is used.
        act_cfg (Optional[dict]): Activation function configuration. Defaults to None.
    """

    def __init__(
        self,
        num_classes: int,
        in_channels: int,
        hidden_dim: Optional[int] = None,
        act_cfg: Optional[dict] = None,
    ):
        super(VisionTransformerClsHead, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        # Initialize layers
        if self.hidden_dim is None:
            self.head = nn.Linear(self.in_channels, self.num_classes)
        else:
            self.fc_pre_logits = nn.Linear(self.in_channels, self.hidden_dim)
            self.act = build_activation(act_cfg)
            self.head = nn.Linear(self.hidden_dim, self.num_classes)


    def pre_logits(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Process features before the final classification head.

        Args:
            feats (torch.Tensor): Input features of shape (B, N, D).

        Returns:
            torch.Tensor: Processed features of shape (B, D).
        """
        # Extract the [CLS] token (first token)
        cls_token = feats[:, 0]

        # Apply hidden layer and activation if exists
        if hasattr(self, 'fc_pre_logits'):
            cls_token = self.fc_pre_logits(cls_token)
            cls_token = self.act(cls_token)

        return cls_token

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the classification head.

        Args:
            feats (torch.Tensor): Input features of shape (B, N, D).

        Returns:
            torch.Tensor: Classification scores of shape (B, num_classes).
        """
        pre_logits = self.pre_logits(feats)
        cls_score = self.head(pre_logits)
        return cls_score


class MultiHeadDiseaseClassifier(nn.Module):
    def __init__(self, in_features, num_classes=8):
        super(MultiHeadDiseaseClassifier, self).__init__()
        
        # 特征增强层
        self.feature_enhancement = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 正常/异常二分类分支
        self.normal_branch = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.LayerNorm(in_features // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(in_features // 2, 1)
        )
        
        # 疾病分类分支（考虑疾病间的关联性）
        self.disease_shared = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 疾病特定分支
        self.disease_specific = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_features, in_features // 2),
                nn.LayerNorm(in_features // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(in_features // 2, 1)
            ) for _ in range(num_classes - 1)
        ])
        
        # 疾病关联注意力
        self.disease_correlation = nn.MultiheadAttention(
            embed_dim=in_features,
            num_heads=4,
            dropout=0.1
        )
        
        # 类别权重（用于处理类别不平衡）
        self.class_weights = nn.Parameter(torch.ones(num_classes))
        
    def forward(self, x):
        # 特征增强
        enhanced_features = self.feature_enhancement(x)
        
        # 正常/异常分类
        normal_logit = self.normal_branch(enhanced_features)
        
        # 疾病共享特征
        disease_features = self.disease_shared(enhanced_features)
        
        # 疾病关联建模
        disease_features = disease_features.unsqueeze(0)  # 添加序列维度
        disease_features, _ = self.disease_correlation(
            disease_features, disease_features, disease_features
        )
        disease_features = disease_features.squeeze(0)  # 移除序列维度
        
        # 各疾病特定分类
        disease_logits = []
        for disease_classifier in self.disease_specific:
            logit = disease_classifier(disease_features)
            disease_logits.append(logit)
        
        # 合并所有预测结果
        disease_logits = torch.cat(disease_logits, dim=1)
        all_logits = torch.cat([normal_logit, disease_logits], dim=1)
        
        # 应用类别权重
        weighted_logits = all_logits * self.class_weights
        
        return weighted_logits
    

    def __init__(self, pos_weights):
        super(DiseaseClassificationLoss, self).__init__()
        self.pos_weights = pos_weights
        
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

class MutualRestraintHead(nn.Module):
    def __init__(self, in_features, num_classes):
        super(MutualRestraintHead, self).__init__()
        self.fc_normal = nn.Linear(in_features, 1)
        self.fc_disease = nn.Linear(in_features, num_classes - 1)
        self.attention = nn.Sequential(
            nn.Linear(in_features, in_features // 2),
            nn.ReLU(),
            nn.Linear(in_features // 2, in_features),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()
        self.num_classes = num_classes

    def forward(self, x):
        # 计算正常和疾病的基础分数
        p_normal = self.fc_normal(x)  # (batch_size, 1)
        
        # 注意力机制处理特征
        attention_weights = self.attention(x)
        weighted_features = x * attention_weights
        p_disease = self.fc_disease(weighted_features)  # (batch_size, num_classes-1)

        # 计算互斥概率
        normal_score = self.sigmoid(p_normal)  # (batch_size, 1)
        disease_scores = self.sigmoid(p_disease)  # (batch_size, num_classes-1)
        
        # 计算互斥权重
        disease_weight = 1 - normal_score.mean()  # 标量
        normal_weight = 1 - disease_scores.mean()  # 标量
        
        # 应用互斥权重
        p_normal_out = p_normal * normal_weight
        p_disease_out = p_disease * disease_weight
        
        # 合并输出
        output = torch.cat([p_normal_out, p_disease_out], dim=1)
        return output

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
        
        # 互斥损失
        mutual_loss = torch.mean(normal_pred * disease_pred.mean(dim=1))
        
        # 组合损失
        total_loss = normal_loss + disease_loss + self.alpha * mutual_loss
        
        return total_loss