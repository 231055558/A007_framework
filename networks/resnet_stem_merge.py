from torch import nn
from blocks.conv import Conv2dModule
from blocks.resnet import Bottleneck


class ResNet_Stem_Merge(nn.Module):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth,
                 in_channels=3,
                 stem_channels=64,
                 base_channels=64,
                 num_classes=1000,
                 norm='batch_norm',
                 activation='relu',
                 dilation=1):
        super(ResNet_Stem_Merge, self).__init__()

        # 检查 depth 是否支持
        if depth not in self.arch_settings:
            raise ValueError(f"Unsupported depth: {depth}")
        block, num_blocks = self.arch_settings[depth]

        # 初始卷积层（stem）
        self.stem = nn.Sequential(
            Conv2dModule(
                in_channels=in_channels,
                out_channels=stem_channels // 2,
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

    def forward(self, x1, x2):
        x1 = self.stem(x1)
        x2 = self.stem(x2)
        x = torch.cat((x1, x2), dim=1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == '__main__':
    import torch
    resnet50 = ResNet_Stem_Merge(depth=50, num_classes=1000)

    x1 = torch.randn(1, 3, 224, 224)
    x2 = torch.randn(1, 3, 224, 224)

    output = resnet50(x1, x2)
    print(output.shape)