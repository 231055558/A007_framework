import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights

from blocks.head import AttentionFC, MultiHeadDiseaseClassifier


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

class DeepLabV3PlusClassifierAMultiHeadDiseaseHeadOutputMerge(nn.Module):
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
        self.fc = MultiHeadDiseaseClassifier(
            in_features=256,  # DeepLabV3+的特征维度
            num_classes=8
        )

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

class DeepLabV3PlusClassifierImproved(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 基本结构同之前
        self.model = deeplabv3_resnet50(aux_loss=False, num_classes=256)
        self.model.backbone.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        del self.model.aux_classifier
        
        # 改进1: 使用金字塔池化而非单一全局池化
        self.pyramid_pool = nn.ModuleList([
            nn.AdaptiveAvgPool2d(output_size) for output_size in [(1, 1), (2, 2), (4, 4)]
        ])
        
        # 改进2: 调整分类头以处理多尺度特征
        total_features = 256 * (1*1 + 2*2 + 4*4)  # 所有池化层输出特征的总和
        # self.fc = nn.Sequential(
        #     nn.Linear(total_features, 512),
        #     nn.ReLU(),
        #     nn.Dropout(0.2),
        #     nn.Linear(512, num_classes)
        # )
        self.fc = AttentionFC(in_features=total_features, num_classes=num_classes)

    def forward(self, x):
        # 特征提取
        x = self.model.backbone(x)['out']
        x = self.model.classifier(x)  # ASPP处理
        
        # 金字塔池化以保留不同尺度的信息
        pyramid_features = []
        for pool in self.pyramid_pool:
            features = pool(x)
            pyramid_features.append(features.view(features.size(0), -1))
        
        # 拼接多尺度特征
        concat_features = torch.cat(pyramid_features, dim=1)
        
        # 分类
        return self.fc(concat_features)

class DeepLabV3PlusClassifierWithAttention(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # 基本结构
        self.model = deeplabv3_resnet50(aux_loss=False, num_classes=256)
        self.model.backbone.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        del self.model.aux_classifier
        
        # 空间注意力模块
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 通道注意力模块
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(256, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 256, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 全局池化和分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        # self.fc = nn.Linear(256, num_classes)
        self.fc = AttentionFC(in_features=256, num_classes=num_classes)

    def forward(self, x):
        # 特征提取
        x = self.model.backbone(x)['out']
        x = self.model.classifier(x)  # ASPP处理
        
        # 应用注意力机制
        spatial_weights = self.spatial_attention(x)
        channel_weights = self.channel_attention(x)
        
        # 加权特征
        x = x * spatial_weights * channel_weights
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == '__main__':
    # 使用示例
    model = DeepLabV3PlusClassifier(num_classes=5)
    input_tensor = torch.randn(2, 6, 512, 512)
    output = model(input_tensor)  # 形状: (2, 5)
    print(output.size())
