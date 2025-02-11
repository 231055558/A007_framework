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
        p_normal = torch.sigmoid(self.fc_normal(x))

        attention_weights = self.attention(x)
        weighted_features = x * attention_weights

        p_disease = torch.sigmoid(self.fc_disease(weighted_features))

        output = torch.cat([p_normal, p_disease], dim=1)