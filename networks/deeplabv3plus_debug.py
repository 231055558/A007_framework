import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.conv import Conv2dModule
from blocks.resnet import Bottleneck
from blocks.aspp.aspp import ASPP
from blocks.aspp.decoder import DeepLabV3PlusDecoder
from blocks.head import AttentionFC, MultiHeadDiseaseClassifier

class DeepLabV3Plus(nn.Module):
    arch_settings = {
        50: (Bottleneck, (3, 4, 6, 3)),
        101: (Bottleneck, (3, 4, 23, 3)),
        152: (Bottleneck, (3, 8, 36, 3)),
    }

    def __init__(self,
                 depth=50,
                 in_channels=3,
                 num_classes=1000,
                 aspp_out_channels=256,
                 decoder_channels=256,
                 output_stride=16,
                 norm='batch_norm',
                 activation='relu'):
        super(DeepLabV3Plus, self).__init__()
        
        if depth not in self.arch_settings:
            raise ValueError(f"Unsupported depth: {depth}")
        block, num_blocks = self.arch_settings[depth]

        if output_stride not in [8, 16]:
            raise ValueError("Output stride must be 8 or 16!")

        # 计算每层的步长和空洞率
        if output_stride == 16:
            strides = [1, 2, 2, 1]
            dilations = [1, 1, 1, 2]
            aspp_rates = [6, 12, 18]
        else:  # output_stride == 8
            strides = [1, 2, 1, 1]
            dilations = [1, 1, 2, 4]
            aspp_rates = [12, 24, 36]

        # Backbone
        self.stem = nn.Sequential(
            Conv2dModule(
                in_channels=in_channels,
                out_channels=64,
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
            in_channels=64,
            out_channels=64 * block.expansion,
            stride=strides[0],
            dilation=dilations[0],
            norm=norm,
            activation=activation
        )
        self.layer2 = self._make_layer(
            block=block,
            num_blocks=num_blocks[1],
            in_channels=64 * block.expansion,
            out_channels=128 * block.expansion,
            stride=strides[1],
            dilation=dilations[1],
            norm=norm,
            activation=activation
        )
        self.layer3 = self._make_layer(
            block=block,
            num_blocks=num_blocks[2],
            in_channels=128 * block.expansion,
            out_channels=256 * block.expansion,
            stride=strides[2],
            dilation=dilations[2],
            norm=norm,
            activation=activation
        )
        self.layer4 = self._make_layer(
            block=block,
            num_blocks=num_blocks[3],
            in_channels=256 * block.expansion,
            out_channels=512 * block.expansion,
            stride=strides[3],
            dilation=dilations[3],
            norm=norm,
            activation=activation
        )

        # ASPP模块
        self.aspp = ASPP(
            in_channels=512 * block.expansion,
            out_channels=aspp_out_channels,
            atrous_rates=aspp_rates,
            norm=norm,
            activation=activation
        )

        # 解码器
        self.decoder = DeepLabV3PlusDecoder(
            low_level_channels=64 * block.expansion,  # layer1的输出通道数
            aspp_out_channels=aspp_out_channels,
            decoder_channels=decoder_channels,
            norm=norm,
            activation=activation
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self,
                   block,
                   num_blocks,
                   in_channels,
                   out_channels,
                   stride=1,
                   dilation=1,
                   norm='batch_norm',
                   activation='relu'):
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

        layers = []
        layers.append(
            block(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                norm=norm,
                activation=activation
            )
        )

        for _ in range(1, num_blocks):
            layers.append(
                block(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    stride=1,
                    dilation=dilation,
                    downsample=None,
                    norm=norm,
                    activation=activation
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):
        input_shape = x.shape[-2:]

        # Backbone
        x = self.stem(x)
        low_level_feat = self.layer1(x)  # 用于跳跃连接
        x = self.layer2(low_level_feat)
        x = self.layer3(x)
        x = self.layer4(x)

        # ASPP
        x = self.aspp(x)

        # Decoder
        x = self.decoder(x, low_level_feat)

        # 上采样到原始大小
        x = F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

        # 分类头
        x = self.classifier(x)

        return x

class DeepLabV3PlusClassifier(nn.Module):
    def __init__(self, num_classes, in_channels=3, backbone_depth=50):
        super(DeepLabV3PlusClassifier, self).__init__()
        
        # DeepLabV3Plus主干网络
        self.model = DeepLabV3Plus(
            depth=backbone_depth,
            in_channels=in_channels,
            num_classes=256,  # 中间特征维度
            output_stride=16
        )
        
        # 移除最后的分类层
        del self.model.classifier
        
        # 全局池化和分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # 获取特征
        x = self.model.stem(x)
        low_level_feat = self.model.layer1(x)
        x = self.model.layer2(low_level_feat)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.aspp(x)
        x = self.model.decoder(x, low_level_feat)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.fc(x)
        return x

class DeepLabV3PlusClassifierAttentionHead(nn.Module):
    def __init__(self, num_classes, in_channels=3, backbone_depth=50):
        super(DeepLabV3PlusClassifierAttentionHead, self).__init__()
        
        # DeepLabV3Plus主干网络
        self.model = DeepLabV3Plus(
            depth=backbone_depth,
            in_channels=in_channels,
            num_classes=256,  # 中间特征维度
            output_stride=16
        )
        
        # 移除最后的分类层
        del self.model.classifier
        
        # 全局池化和注意力分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = AttentionFC(in_features=256, num_classes=num_classes)

    def forward(self, x):
        # 获取特征
        x = self.model.stem(x)
        low_level_feat = self.model.layer1(x)
        x = self.model.layer2(low_level_feat)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.aspp(x)
        x = self.model.decoder(x, low_level_feat)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.fc(x)
        return x

class DeepLabV3PlusClassifierMultiHeadDisease(nn.Module):
    def __init__(self, in_channels=3, backbone_depth=50):
        super(DeepLabV3PlusClassifierMultiHeadDisease, self).__init__()
        
        # DeepLabV3Plus主干网络
        self.model = DeepLabV3Plus(
            depth=backbone_depth,
            in_channels=in_channels,
            num_classes=256,  # 中间特征维度
            output_stride=16
        )
        
        # 移除最后的分类层
        del self.model.classifier
        
        # 全局池化和多头疾病分类器
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = MultiHeadDiseaseClassifier(in_features=256, num_classes=8)

    def forward(self, x):
        # 获取特征
        x = self.model.stem(x)
        low_level_feat = self.model.layer1(x)
        x = self.model.layer2(low_level_feat)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.aspp(x)
        x = self.model.decoder(x, low_level_feat)
        
        # 全局池化
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        # 分类
        x = self.fc(x)
        return x

class DeepLabV3PlusClassifierAttentionHeadOutputMerge(nn.Module):
    def __init__(self, num_classes, backbone_depth=50):
        super(DeepLabV3PlusClassifierAttentionHeadOutputMerge, self).__init__()
        
        # DeepLabV3Plus主干网络 - 注意这里使用3通道输入
        self.model = DeepLabV3Plus(
            depth=backbone_depth,
            in_channels=3,  # RGB输入
            num_classes=256,  # 中间特征维度
            output_stride=16
        )
        
        # 移除最后的分类层
        del self.model.classifier
        
        # 全局池化和注意力分类头
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = AttentionFC(in_features=256, num_classes=num_classes)

    def forward(self, x):
        # 获取特征
        x = self.model.stem(x)
        low_level_feat = self.model.layer1(x)
        x = self.model.layer2(low_level_feat)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.aspp(x)
        x = self.model.decoder(x, low_level_feat)
        
        # 全局池化
        x = self.global_pool(x)  # 输出形状: (bs, 256, 1, 1)
        x = x.view(x.size(0), -1)  # 展平: (bs, 256)
        
        # 分类
        x = self.fc(x)  # 输出: (bs, num_classes)
        return x

if __name__ == '__main__':
    # 测试代码
    model = DeepLabV3Plus(depth=50, num_classes=21)
    x = torch.randn(2, 3, 513, 513)
    output = model(x)
    print(f"Segmentation output shape: {output.shape}")

    # 测试分类器
    classifier = DeepLabV3PlusClassifier(num_classes=5)
    output = classifier(x)
    print(f"Classification output shape: {output.shape}")

    # 测试注意力头分类器
    attention_classifier = DeepLabV3PlusClassifierAttentionHead(num_classes=5)
    output = attention_classifier(x)
    print(f"Attention classification output shape: {output.shape}")

    # 测试多头疾病分类器
    disease_classifier = DeepLabV3PlusClassifierMultiHeadDisease()
    output = disease_classifier(x)
    print(f"Multi-head disease classification output shape: {output.shape}")

    # 测试输出融合的注意力头分类器
    output_merge_classifier = DeepLabV3PlusClassifierAttentionHeadOutputMerge(num_classes=5)
    output = output_merge_classifier(x)
    print(f"Output merge attention classification shape: {output.shape}") 