from torch import nn
from blocks.conv import Conv2dModule
from blocks.resnet import Bottleneck

import matplotlib.pyplot as plt
def gray(feature_map):
    feature_map = feature_map.cpu().detach().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(feature_map, cmap='gray', interpolation='nearest')
    plt.title('Grayscale of the selected feature map')
    plt.colorbar()
    plt.show()

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_heatmap(feature_map: torch.Tensor, image: np.ndarray = None, alpha: float = 0.5, colormap: str = 'jet'):
    """
    生成热力图并可视化。

    Args:
        feature_map (torch.Tensor): 输入的 C×H×W 特征图。
        image (np.ndarray): 原始图像 (H, W, 3)，用于叠加热力图。如果为 None，则只显示热力图。
        alpha (float): 热力图透明度，默认为 0.5。
        colormap (str): 热力图颜色映射，默认为 'jet'。

    Returns:
        heatmap (np.ndarray): 生成的热力图 (H, W, 3)。
    """
    feature_map = feature_map.cpu().detach()
    # 确保输入是 Tensor
    if not isinstance(feature_map, torch.Tensor):
        feature_map = torch.tensor(feature_map)

    # 在通道方向（C）上叠加
    heatmap = torch.sum(feature_map, dim=0)  # 形状变为 (H, W)

    # 归一化到 [0, 1] 范围
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

    # 转换为 NumPy 数组
    heatmap = heatmap.numpy()

    # 应用颜色映射
    heatmap = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET if colormap == 'jet' else cv2.COLORMAP_VIRIDIS)

    # 如果提供了原始图像，将热力图叠加到图像上
    if image is not None:
        # 确保图像和热力图大小一致
        heatmap = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
        heatmap = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    # 显示结果
    plt.imshow(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

    return heatmap


class ResNet_Color_Merge(nn.Module):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth,
                 in_channels=6,
                 stem_channels=64,
                 base_channels=64,
                 num_classes=1000,
                 norm='batch_norm',
                 activation='relu',
                 dilation=1):
        super(ResNet_Color_Merge, self).__init__()

        # 检查 depth 是否支持
        if depth not in self.arch_settings:
            raise ValueError(f"Unsupported depth: {depth}")
        block, num_blocks = self.arch_settings[depth]

        # 初始卷积层（stem）
        self.stem = nn.Sequential(
            Conv2dModule(
                in_channels=in_channels,
                out_channels=stem_channels,
                kernel_size=7,
                stride=2,
                padding=3,
                norm=norm,
                activation=activation
            ),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # 残差层
        self.layer1 = self._make_layer(
            block=block,
            num_blocks=num_blocks[0],
            in_channels=stem_channels,  # 输入通道数
            out_channels=base_channels * block.expansion,  # 输出通道数 = base_channels * 4
            stride=1,
            dilation=dilation,
            norm=norm,
            activation=activation
        )
        self.layer2 = self._make_layer(
            block=block,
            num_blocks=num_blocks[1],
            in_channels=base_channels * block.expansion,  # 输入通道数
            out_channels=base_channels * 2 * block.expansion,  # 输出通道数 = base_channels*2*4
            stride=2,
            dilation=dilation,
            norm=norm,
            activation=activation
        )
        self.layer3 = self._make_layer(
            block=block,
            num_blocks=num_blocks[2],
            in_channels=base_channels * 2 * block.expansion,
            out_channels=base_channels * 4 * block.expansion,
            stride=2,
            dilation=dilation,
            norm=norm,
            activation=activation
        )
        self.layer4 = self._make_layer(
            block=block,
            num_blocks=num_blocks[3],
            in_channels=base_channels * 4 * block.expansion,
            out_channels=base_channels * 8 * block.expansion,
            stride=2,
            dilation=dilation,
            norm=norm,
            activation=activation
        )
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(base_channels * 8 * block.expansion, base_channels * 8 * block.expansion, kernel_size=1,
                      bias=False),
            nn.BatchNorm2d(base_channels * 8 * block.expansion),
            nn.ReLU(inplace=True)
        )

        # 全局平均池化和分类器
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

    def _make_layer(self,
                    block,
                    num_blocks,
                    in_channels,
                    out_channels,  # 这里的 out_channels 是最终的输出通道数
                    stride=1,
                    dilation=1,
                    norm='batch_norm',
                    activation='relu'):
        # 是否需要下采样
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = Conv2dModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=stride,
                norm=norm,
                activation=None
            )

        # 构建残差层
        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,  # 主分支的最终输出通道数
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                norm=norm,
                activation=activation
            )
        )

        # 后续 Blocks
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,  # 输入输出通道数一致
                    stride=1,
                    dilation=dilation,
                    downsample=None,
                    norm=norm,
                    activation=activation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.feature_fusion(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import torch
    resnet50 = ResNet_Color_Merge(depth=50, num_classes=1000)

    x = torch.randn(1, 6, 224, 224)

    output = resnet50(x)
    print(output.shape)