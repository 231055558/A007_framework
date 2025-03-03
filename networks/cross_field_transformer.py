import torch
import numpy as np
from torch import nn
from torch.nn import functional as F

from blocks.conv import Conv2dModule
from blocks.cross_field_transformer import Transformer_head
from blocks.resnet import Bottleneck

class CrossFieldTransformer(nn.Module):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_classes=1000,
                 xfmer_hidden_size=1024,
                 xfmer_layer=3,
                 p_threshold=0.5,
                 norm='batch_norm',
                 activation='relu',
                 dilation=1):
        super(CrossFieldTransformer, self).__init__()

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

        # # 全局平均池化和分类器
        # self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        self.xfmer_hidden_size = xfmer_hidden_size
        self.xfmer_layer=xfmer_layer
        self.p_threshold=p_threshold
        self.reduce = nn.Conv2d(2048, self.xfmer_hidden_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.pool = 'avg'
        self.xfmer_dropout = nn.Dropout(0.1)
        self.xfmer = Transformer_head(hidden_size=self.xfmer_hidden_size, layers=self.xfmer_layer)
        self.xfmer_fc = nn.Sequential(
            nn.LayerNorm(self.xfmer_hidden_size),
            nn.Linear(self.xfmer_hidden_size, num_classes)
        )

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

    def forward(self, x1, x2):
        x1 = self.stem(x1)
        x1 = self.layer1(x1)
        x1 = self.layer2(x1)
        x1 = self.layer3(x1)
        x1 = self.layer4(x1)

        x2 = self.stem(x2)
        x2 = self.layer1(x2)
        x2 = self.layer2(x2)
        x2 = self.layer3(x2)
        x2 = self.layer4(x2)

        x1_layer4 = x1
        x2_layer4 = x2

        x1_patch_layer4 = torch.flatten(x1_layer4, start_dim=2).permute(0, 2, 1) # bs, 256, 2048
        x2_patch_layer4 = torch.flatten(x2_layer4, start_dim=2).permute(0, 2, 1)

        x1 = self.reduce(x1)
        x2 = self.reduce(x2)

        x1_patch = torch.flatten(x1, start_dim=2).permute(0, 2, 1) # bs, 256, 1024
        x2_patch = torch.flatten(x2, start_dim=2).permute(0, 2, 1)

        x1_avg = torch.mean(x1_patch_layer4, dim=-1) #bs, 256
        x2_avg = torch.mean(x2_patch_layer4, dim=-1)

        x1_avg_max, _ = torch.max(x1_avg, dim=1) #bs
        x1_avg_min, _ = torch.min(x1_avg, dim=1)
        x1_avg = (x1_avg - x1_avg_min.unsqueeze(-1)) / (x1_avg_max.unsqueeze(-1) - x1_avg_min.unsqueeze(-1))

        x2_avg_max, _ = torch.max(x2_avg, dim=1)
        x2_avg_min, _ = torch.min(x2_avg, dim=1)
        x2_avg = (x2_avg - x2_avg_min.unsqueeze(-1)) / (x2_avg_max.unsqueeze(-1) - x2_avg_min.unsqueeze(-1))

        pred1 = (x1_avg < self.p_threshold).float()
        pred2 = (x2_avg < self.p_threshold).float()  # bs, 256
        pred = torch.cat([pred1, pred2], dim=-1) # bs, 512

        bs, feat_dim, g_size, _ = x1.size()

        out_patch = torch.cat([x1_patch, x2_patch], dim=1)  # bs, 256, 1024
        out_patch = self.xfmer_dropout(out_patch)
        out_patch = self.xfmer(out_patch, pred)

        if self.pool == 'avg':
            out = torch.mean(out_patch, dim=1)

        out = self.xfmer_fc(out)

        return out

if __name__ == '__main__':

    # 设置随机种子以确保结果可重复
    torch.manual_seed(42)

    # 定义模型参数
    depth = 50  # 支持 50, 101, 152
    in_channels = 3
    num_classes = 10
    batch_size = 2
    image_size = 224  # 假设输入图像大小为 224x224

    # 创建模型实例
    model = CrossFieldTransformer(depth=depth, in_channels=in_channels, num_classes=num_classes)

    # 打印模型结构
    print(model)

    # 生成随机输入数据
    x1 = torch.randn(batch_size, in_channels, image_size, image_size)  # 输入图像 1
    x2 = torch.randn(batch_size, in_channels, image_size, image_size)  # 输入图像 2

    # 打印输入数据的形状
    print(f"Input 1 shape: {x1.shape}")
    print(f"Input 2 shape: {x2.shape}")

    # 前向传播
    output = model(x1, x2)

    # 打印输出数据的形状
    print(f"Output shape: {output.shape}")

    # 检查输出是否符合预期
    assert output.shape == (batch_size, num_classes), "Output shape is incorrect!"

    print("Test passed!")