import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from blocks.head import AttentionFC


class DeepLabV3PlusClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 不设置 aux_loss，默认使用 True（与预训练权重兼容）
        self.model = deeplabv3_resnet50(aux_loss=False, num_classes=256)
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # 手动移除辅助分类器
        del self.model.aux_classifier
        # 移除主解码器
        # self.model.classifier

        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)  # ASPP 输出通道数为 256

    def forward(self, x):
        # Backbone 特征提取
        x = self.model.backbone(x)['out']  # 输出形状: (bs, 2048, H/16, W/16)
        # ASPP 多尺度融合
        # x = self.model.aspp(x)  # 输出形状: (bs, 256, H/16, W/16)
        x = self.model.classifier(x)
        # 全局池化 + 分类

        x = self.global_pool(x)  # 输出形状: (bs, 256, 1, 1)
        x = x.view(x.size(0), -1)  # 展平: (bs, 256)
        return self.fc(x)  # 输出: (bs, num_classes)

class DeepLabV3PlusClassifierAttentionHeadColorMerge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 不设置 aux_loss，默认使用 True（与预训练权重兼容）
        self.model = deeplabv3_resnet50(aux_loss=False, num_classes=256)
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=6,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # 手动移除辅助分类器
        del self.model.aux_classifier
        # 移除主解码器
        # self.model.classifier

        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(256, num_classes)  # ASPP 输出通道数为 256
        self.fc = AttentionFC(in_features=256, num_classes=num_classes)

    def forward(self, x):
        # Backbone 特征提取
        x = self.model.backbone(x)['out']  # 输出形状: (bs, 2048, H/16, W/16)
        # ASPP 多尺度融合
        # x = self.model.aspp(x)  # 输出形状: (bs, 256, H/16, W/16)
        x = self.model.classifier(x)
        # 全局池化 + 分类

        x = self.global_pool(x)  # 输出形状: (bs, 256, 1, 1)
        x = x.view(x.size(0), -1)  # 展平: (bs, 256)
        return self.fc(x)  # 输出: (bs, num_classes)

class DeepLabV3PlusClassifierAttentionHeadOutputMerge(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 不设置 aux_loss，默认使用 True（与预训练权重兼容）
        self.model = deeplabv3_resnet50(aux_loss=False, num_classes=256)
        self.model.backbone.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # 手动移除辅助分类器
        del self.model.aux_classifier
        # 移除主解码器
        # self.model.classifier

        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(256, num_classes)  # ASPP 输出通道数为 256
        self.fc = AttentionFC(in_features=256, num_classes=num_classes)

    def forward(self, x):
        # Backbone 特征提取
        x = self.model.backbone(x)['out']  # 输出形状: (bs, 2048, H/16, W/16)
        # ASPP 多尺度融合
        # x = self.model.aspp(x)  # 输出形状: (bs, 256, H/16, W/16)
        x = self.model.classifier(x)
        # 全局池化 + 分类

        x = self.global_pool(x)  # 输出形状: (bs, 256, 1, 1)
        x = x.view(x.size(0), -1)  # 展平: (bs, 256)
        return self.fc(x)  # 输出: (bs, num_classes)

if __name__ == '__main__':
    # 使用示例
    model = DeepLabV3PlusClassifier(num_classes=5)
    input_tensor = torch.randn(2, 6, 512, 512)
    output = model(input_tensor)  # 形状: (2, 5)
    print(output.size())
