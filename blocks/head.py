import torch
import torch.nn as nn
import torch.nn.functional as F


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

import torch
import torch.nn as nn
from typing import Optional

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
            self.pre_logits = nn.Linear(self.in_channels, self.hidden_dim)
            self.act = self._build_activation(act_cfg)
            self.head = nn.Linear(self.hidden_dim, self.num_classes)

        # Initialize weights
        self.init_weights()

    def _build_activation(self, act_cfg: Optional[dict]) -> nn.Module:
        """
        Build activation function based on configuration.

        Args:
            act_cfg (Optional[dict]): Activation function configuration.

        Returns:
            nn.Module: Activation function.
        """
        if act_cfg is None:
            return nn.Identity()
        act_type = act_cfg.get('type', 'Tanh')
        if act_type == 'Tanh':
            return nn.Tanh()
        elif act_type == 'ReLU':
            return nn.ReLU()
        elif act_type == 'GELU':
            return nn.GELU()
        else:
            raise ValueError(f"Unsupported activation type: {act_type}")

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
        if hasattr(self, 'pre_logits'):
            cls_token = self.pre_logits(cls_token)
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
