import torch
import torch.nn as nn
import torch.nn.functional as F
from blocks.conv import Conv2dModule

class ASPPModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, norm='batch_norm', activation='relu'):
        super(ASPPModule, self).__init__()
        self.atrous_conv = Conv2dModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=dilation,
            dilation=dilation,
            norm=norm,
            activation=activation
        )

    def forward(self, x):
        return self.atrous_conv(x)

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm='batch_norm', activation='relu'):
        super(ASPP, self).__init__()
        
        # 1x1 卷积分支
        self.conv1 = Conv2dModule(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
            activation=activation
        )

        # 多个空洞卷积分支
        self.aspp_modules = nn.ModuleList()
        for dilation in atrous_rates:
            self.aspp_modules.append(
                ASPPModule(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    dilation=dilation,
                    norm=norm,
                    activation=activation
                )
            )

        # 全局池化分支
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            Conv2dModule(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                norm=norm,
                activation=activation
            )
        )

        # 最终的1x1卷积，用于融合所有分支
        self.bottleneck = Conv2dModule(
            in_channels=out_channels * (len(atrous_rates) + 2),  # +2是因为有1x1卷积和全局池化分支
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            norm=norm,
            activation=activation
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 获取输入特征图的尺寸
        size = x.size()

        # 1x1 卷积分支
        conv1_out = self.conv1(x)

        # 空洞卷积分支
        aspp_outs = [conv1_out]
        for aspp_module in self.aspp_modules:
            aspp_outs.append(aspp_module(x))

        # 全局池化分支
        global_out = self.global_avg_pool(x)
        global_out = F.interpolate(global_out, size=size[2:], mode='bilinear', align_corners=True)
        aspp_outs.append(global_out)

        # 拼接所有分支的输出
        concat_out = torch.cat(aspp_outs, dim=1)

        # 通过1x1卷积融合特征
        output = self.bottleneck(concat_out)

        return output 