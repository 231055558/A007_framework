import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn.functional as F
from dataset.A007_txt_merge_model import A007Dataset
from dataset.transform import *
from models.load import load_model_weights


class AttentionFC(nn.Module):
    def _init__(self, in_features, num_classes):
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

    def forward(self, x, return_attention=False):
        p_normal = self.fc_normal(x)

        attention_weights = self.attention(x)
        weighted_features = x * attention_weights

        p_disease = self.fc_disease(weighted_features)

        output = torch.cat([p_normal, p_disease], dim=1)
        if return_attention:
            return output, attention_weights
        return output


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

        # 分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
    def forward(self, x):
        # Backbone 特征提取
        x = self.model.backbone(x)['out']  # 输出形状: (bs, 2048, H/16, W/16)

        return x   # 输出: (bs, num_classes)


# 叠加热力图到原图
def overlay_heatmap(image, heatmap, alpha=0.3):
    """
    将热力图叠加到原始图片上。

    Args:
        image (np.ndarray): 原始图片，形状 (H, W, 3)
        heatmap (np.ndarray): 热力图，形状 (H, W)
        alpha (float): 热力图的透明度，默认为 0.5

    Returns:
        superimposed_img (np.ndarray): 叠加后的图片，形状 (H, W, 3)
    """
    # 将热力图转换为颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    # 将热力图叠加到原始图片上
    superimposed_img = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return superimposed_img


# 加载数据集
data_root = '../../data/data_merge'
transform_val = Compose([LoadImageFromFile(),
                         CenterCrop((512, 512)),
                         ToTensor(),
                         Resize((512, 512)),
                         Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])
val_loader = DataLoader(A007Dataset(txt_file="val.txt",
                                    root_dir=data_root,
                                    transform=transform_val,
                                    seed=42,
                                    preload=False),
                        batch_size=1,
                        shuffle=True,  # 随机打乱数据
                        num_workers=4,
                        pin_memory=True)

# 加载模型
pretrain_ckp = "./best_model.pth"
model = DeepLabV3PlusClassifierAttentionHeadOutputMerge(num_classes=8)
load_model_weights(model, pretrain_ckp)
model.eval()

# 准备绘图
plt.figure(figsize=(20, 20))  # 8x8 网格，每张图片大小为 2x2 英寸

# 处理 64 张图片
for i, batch in enumerate(val_loader):
    if i >= 64:
        break  # 只处理 64 张图片

    image = batch[1]  # 输入图片，形状 (1, 3, 512, 512)
    # 获取模型输出和注意力权重
    output = model(image)
    # 1. 对 output 在 2048 方向上求和
    output_sum = torch.sum(output, dim=1, keepdim=True)  # 形状 (1, 1, 64, 64)

    # 2. 将 image 转换为灰度图
    gray_image = torch.mean(image, dim=1, keepdim=True)  # 形状 (1, 1, 512, 512)
    gray_image = gray_image.squeeze().cpu().numpy()  # 转换为 NumPy 数组，形状 (512, 512)

    # 3. 将 output 转换为热力图并调整大小
    heatmap = (output_sum - output_sum.min()) / (output_sum.max() - output_sum.min())  # 归一化
    heatmap = F.interpolate(heatmap, size=(512, 512), mode='bilinear', align_corners=False)  # 调整为 512x512
    heatmap = heatmap.squeeze().cpu().detach().numpy()  # 转换为 NumPy 数组，形状 (512, 512)

    # 4. 将热力图转换为颜色映射
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)

    # 5. 将灰度图转换为 3 通道
    gray_image_color = cv2.cvtColor(np.uint8(255 * gray_image), cv2.COLOR_GRAY2BGR)

    # 6. 将热力图和灰度图叠加
    alpha = 0.5
    superimposed_img = cv2.addWeighted(gray_image_color, 1 - alpha, heatmap_color, alpha, 0)
    # 绘制到子图中
    plt.subplot(8, 8, i + 1)
    plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
    plt.axis('off')

# 显示图表
plt.tight_layout()
plt.show()
