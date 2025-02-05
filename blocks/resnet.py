import torch.nn as nn

from blocks.conv import Conv2dModule


class ResLayer(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 in_channels,
                 out_channels,
                 expansion=None,
                 stride=1,
                 norm='batch_norm',
                 ):
        super(ResLayer, self).__init__()

        if expansion is None:
            if hasattr(block, 'expansion'):
                expansion = block.expansion
            else:
                raise ValueError(f"expansion must be specified for block {block.__name__}")

        layers = list()

        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expansion=expansion,
                norm=norm
            )
        )
        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=stride,
                    expansion=expansion,
                    norm=norm
                )
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Bottleneck(nn.Module):
    expansion = 4  # 定义通道扩展系数

    def __init__(self,
                 in_channels,
                 out_channels,  # 注意：这里的 out_channels 是主分支的最终输出通道数
                 stride=1,  # 不是中间层的通道数！
                 dilation=1,
                 downsample=None,
                 norm='batch_norm',
                 activation='relu'):
        super(Bottleneck, self).__init__()

        # 计算中间层的通道数
        mid_channels = out_channels // self.expansion

        # 1x1 卷积层（降维）
        self.conv1 = Conv2dModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
            activation=activation
        )

        # 3x3 卷积层（特征提取）
        self.conv2 = Conv2dModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            norm=norm,
            activation=activation
        )

        # 1x1 卷积层（升维）
        self.conv3 = Conv2dModule(
            in_channels=mid_channels,
            out_channels=out_channels,  # 主分支最终输出通道数 = out_channels
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
            activation=None  # 最后一层不激活
        )

        # 残差连接分支
        self.downsample = downsample

        # 激活函数
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        else:
            raise ValueError(f"Unsupported activation: {activation}")

    def forward(self, x):
        identity = x

        # 主分支
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        # 残差连接分支
        if self.downsample is not None:
            identity = self.downsample(identity)

        # 残差相加
        x += identity
        x = self.activation(x)

        return x
