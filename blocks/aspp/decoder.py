import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.conv import Conv2dModule

class DeepLabV3PlusDecoder(nn.Module):
    def __init__(self, low_level_channels, aspp_out_channels, decoder_channels=256, norm='batch_norm', activation='relu'):
        super(DeepLabV3PlusDecoder, self).__init__()
        
        # 低层特征的处理
        self.low_level_conv = Conv2dModule(
            in_channels=low_level_channels,
            out_channels=48,  # 论文中使用的通道数
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
            activation=activation
        )

        # 解码器的卷积层
        self.decoder_conv = nn.Sequential(
            Conv2dModule(
                in_channels=aspp_out_channels + 48,  # ASPP输出 + 低层特征
                out_channels=decoder_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                activation=activation
            ),
            Conv2dModule(
                in_channels=decoder_channels,
                out_channels=decoder_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                norm=norm,
                activation=activation
            )
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, low_level_feat):
        # 处理低层特征
        low_level_feat = self.low_level_conv(low_level_feat)
        
        # 上采样ASPP的输出
        x = F.interpolate(x, size=low_level_feat.shape[2:], mode='bilinear', align_corners=True)
        
        # 拼接特征
        x = torch.cat([x, low_level_feat], dim=1)
        
        # 解码器处理
        x = self.decoder_conv(x)
        
        return x 