from torch import nn
import torch
import torch.nn.functional as F
from blocks.head import AttentionFC
from blocks.conv import Conv2dModule
from blocks.resnet import Bottleneck

class ResNetFPNAttentionHead(nn.Module):
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
        super(ResNetFPNAttentionHead, self).__init__()

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

        # FPN层
        self.fpn_channels = 256  # FPN的通道数
        
        # 横向连接层
        self.lateral_layer1 = Conv2dModule(base_channels * 2 * block.expansion, self.fpn_channels, 
                                         kernel_size=1, norm=norm, activation=None)
        self.lateral_layer2 = Conv2dModule(base_channels * 4 * block.expansion, self.fpn_channels, 
                                         kernel_size=1, norm=norm, activation=None)
        self.lateral_layer3 = Conv2dModule(base_channels * 8 * block.expansion, self.fpn_channels, 
                                         kernel_size=1, norm=norm, activation=None)
        
        # 自顶向下的路径
        self.fpn_layer1 = Conv2dModule(self.fpn_channels, self.fpn_channels, 
                                     kernel_size=3, padding=1, norm=norm, activation=activation)
        self.fpn_layer2 = Conv2dModule(self.fpn_channels, self.fpn_channels, 
                                     kernel_size=3, padding=1, norm=norm, activation=activation)
        self.fpn_layer3 = Conv2dModule(self.fpn_channels, self.fpn_channels, 
                                     kernel_size=3, padding=1, norm=norm, activation=activation)
        
        # 修改全连接层的输入维度
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = AttentionFC(in_features=self.fpn_channels * 3, num_classes=num_classes)
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
        # 主干网络前向传播
        x = self.stem(x)
        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        
        # FPN自顶向下的路径
        p5 = self.lateral_layer3(c5)
        p4 = self._upsample_add(p5, self.lateral_layer2(c4))
        p3 = self._upsample_add(p4, self.lateral_layer1(c3))
        
        # FPN特征增强
        p3 = self.fpn_layer1(p3)
        p4 = self.fpn_layer2(p4)
        p5 = self.fpn_layer3(p5)
        
        # 池化并拼接特征
        f3 = self.avgpool(p3)
        f4 = self.avgpool(p4)
        f5 = self.avgpool(p5)
        
        # 拼接多尺度特征
        out = torch.cat([f3, f4, f5], dim=1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
        
    def _upsample_add(self, x, y):
        """上采样并相加"""
        return F.interpolate(x, size=y.shape[2:], mode='bilinear', align_corners=False) + y
