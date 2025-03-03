import torch
import torch.nn as nn

from blocks.head import VisionTransformerClsHead
from blocks.transformer import PatchEmbed, TransformerEncoderLayer, resize_pos_embed


class VisionTransformer(nn.Module):
    """
    Vision Transformer (ViT) implementation.

    Args:
        arch (str): Architecture type, e.g., 'small', 'base', 'large', etc.
        img_size (int or tuple): Input image size (height, width).
        patch_size (int or tuple): Patch size (height, width).
        in_channels (int): Number of input channels (default: 3 for RGB).
        out_indices (list): Indices of layers to output features.
        out_type (str): Output type, e.g., 'raw', 'cls_token', 'featmap', 'avg_featmap'.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Drop path rate for stochastic depth.
        qkv_bias (bool): Whether to add bias to the qkv projection.
        norm_cfg (dict): Configuration for normalization layer.
        layer_scale_init_value (float): Initial value for layer scale.
    """

    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small', 'dinov2-s', 'dinov2-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
        **dict.fromkeys(
            ['dinov2-g', 'dinov2-giant'], {
                'embed_dims': 1536,
                'num_layers': 40,
                'num_heads': 24,
                'feedforward_channels': 6144
            }),
    }

    num_extra_tokens = 1  # Class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(
        self,
        arch='base',
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000,
        out_indices=None,
        out_type='cls_token',
        drop_rate=0.,
        drop_path_rate=0.,
        qkv_bias=True,
        norm_cfg={'type': 'LN'},
        layer_scale_init_value=0.,
    ):
        super(VisionTransformer, self).__init__()
        self.arch_settings = self.arch_zoo[arch]
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.out_indices = out_indices if out_indices is not None else []
        self.out_type = out_type
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.norm_cfg = norm_cfg
        self.layer_scale_init_value = layer_scale_init_value

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dims=self.arch_settings['embed_dims'],
        )
        self.patch_resolution = self.patch_embed.grid_size

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.arch_settings['embed_dims']))

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_extra_tokens, self.arch_settings['embed_dims']))

        # Dropout after Position Embedding
        self.drop_after_pos = nn.Dropout(p=self.drop_rate)

        # Pre-normalization
        self.pre_norm = nn.LayerNorm(self.arch_settings['embed_dims'])

        # Transformer Layers
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.arch_settings['num_layers'])]
        self.layers = nn.ModuleList()
        for i in range(self.arch_settings['num_layers']):
            self.layers.append(TransformerEncoderLayer(
                embed_dims=self.arch_settings['embed_dims'],
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                layer_scale_init_value=self.layer_scale_init_value,
                drop_rate=self.drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=self.qkv_bias,
                norm_cfg=self.norm_cfg,
            ))

        # Final Normalization
        self.final_norm = nn.LayerNorm(self.arch_settings['embed_dims'])

        self.cls_head = VisionTransformerClsHead(
            num_classes=num_classes,
            in_channels=self.arch_settings['embed_dims'],
            hidden_dim=None,
            act_cfg={'type': "GELU"}
        )


    def _format_output(self, x, patch_resolution):
        """
        Format the output based on `out_type`.

        Args:
            x (Tensor): Input tensor of shape (B, N, D).
            patch_resolution (tuple): Patch resolution (H_patches, W_patches).

        Returns:
            Tensor: Formatted output.
        """
        if self.out_type == 'raw':
            return x
        elif self.out_type == 'cls_token':
            return x[:, 0]
        elif self.out_type == 'featmap':
            B, N, D = x.shape
            H, W = patch_resolution
            return x[:, 1:].reshape(B, H, W, D).permute(0, 3, 1, 2)
        elif self.out_type == 'avg_featmap':
            return x[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Unsupported out_type: {self.out_type}")

    def forward(self, x):
        """
        Forward pass for VisionTransformer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple: Contains outputs from specified layers.
        """
        # Input and Patch Embedding
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # Add Class Token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add Position Embedding
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode='bicubic',
            num_extra_tokens=self.num_extra_tokens,
        )

        # Dropout after Position Embedding
        x = self.drop_after_pos(x)

        # Pre-normalization
        x = self.pre_norm(x)

        # Transformer Layers
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.final_norm(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        cls_score = self.cls_head(x)
        return cls_score

        # return tuple(outs)


class VisionTransformer_Color_Merge(nn.Module):
    """
    Vision Transformer (ViT) implementation.

    Args:
        arch (str): Architecture type, e.g., 'small', 'base', 'large', etc.
        img_size (int or tuple): Input image size (height, width).
        patch_size (int or tuple): Patch size (height, width).
        in_channels (int): Number of input channels (default: 3 for RGB).
        out_indices (list): Indices of layers to output features.
        out_type (str): Output type, e.g., 'raw', 'cls_token', 'featmap', 'avg_featmap'.
        drop_rate (float): Dropout rate.
        drop_path_rate (float): Drop path rate for stochastic depth.
        qkv_bias (bool): Whether to add bias to the qkv projection.
        norm_cfg (dict): Configuration for normalization layer.
        layer_scale_init_value (float): Initial value for layer scale.
    """

    arch_zoo = {
        **dict.fromkeys(
            ['s', 'small'], {
                'embed_dims': 768,
                'num_layers': 8,
                'num_heads': 8,
                'feedforward_channels': 768 * 3,
            }),
        **dict.fromkeys(
            ['b', 'base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 3072
            }),
        **dict.fromkeys(
            ['l', 'large'], {
                'embed_dims': 1024,
                'num_layers': 24,
                'num_heads': 16,
                'feedforward_channels': 4096
            }),
        **dict.fromkeys(
            ['h', 'huge'],
            {
                'embed_dims': 1280,
                'num_layers': 32,
                'num_heads': 16,
                'feedforward_channels': 5120
            }),
        **dict.fromkeys(
            ['eva-g', 'eva-giant'],
            {
                'embed_dims': 1408,
                'num_layers': 40,
                'num_heads': 16,
                'feedforward_channels': 6144
            }),
        **dict.fromkeys(
            ['deit-t', 'deit-tiny'], {
                'embed_dims': 192,
                'num_layers': 12,
                'num_heads': 3,
                'feedforward_channels': 192 * 4
            }),
        **dict.fromkeys(
            ['deit-s', 'deit-small', 'dinov2-s', 'dinov2-small'], {
                'embed_dims': 384,
                'num_layers': 12,
                'num_heads': 6,
                'feedforward_channels': 384 * 4
            }),
        **dict.fromkeys(
            ['deit-b', 'deit-base'], {
                'embed_dims': 768,
                'num_layers': 12,
                'num_heads': 12,
                'feedforward_channels': 768 * 4
            }),
        **dict.fromkeys(
            ['dinov2-g', 'dinov2-giant'], {
                'embed_dims': 1536,
                'num_layers': 40,
                'num_heads': 24,
                'feedforward_channels': 6144
            }),
    }

    num_extra_tokens = 1  # Class token
    OUT_TYPES = {'raw', 'cls_token', 'featmap', 'avg_featmap'}

    def __init__(
        self,
        arch='base',
        img_size=224,
        patch_size=16,
        in_channels=6,
        num_classes=1000,
        out_indices=None,
        out_type='cls_token',
        drop_rate=0.,
        drop_path_rate=0.,
        qkv_bias=True,
        norm_cfg={'type': 'LN'},
        layer_scale_init_value=0.,
    ):
        super(VisionTransformer_Color_Merge, self).__init__()
        self.arch_settings = self.arch_zoo[arch]
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.out_indices = out_indices if out_indices is not None else []
        self.out_type = out_type
        self.drop_rate = drop_rate
        self.drop_path_rate = drop_path_rate
        self.qkv_bias = qkv_bias
        self.norm_cfg = norm_cfg
        self.layer_scale_init_value = layer_scale_init_value

        # Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=self.img_size,
            patch_size=self.patch_size,
            in_channels=self.in_channels,
            embed_dims=self.arch_settings['embed_dims'],
        )
        self.patch_resolution = self.patch_embed.grid_size

        # Class Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.arch_settings['embed_dims']))

        # Position Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches + self.num_extra_tokens, self.arch_settings['embed_dims']))

        # Dropout after Position Embedding
        self.drop_after_pos = nn.Dropout(p=self.drop_rate)

        # Pre-normalization
        self.pre_norm = nn.LayerNorm(self.arch_settings['embed_dims'])

        # Transformer Layers
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.arch_settings['num_layers'])]
        self.layers = nn.ModuleList()
        for i in range(self.arch_settings['num_layers']):
            self.layers.append(TransformerEncoderLayer(
                embed_dims=self.arch_settings['embed_dims'],
                num_heads=self.arch_settings['num_heads'],
                feedforward_channels=self.arch_settings['feedforward_channels'],
                layer_scale_init_value=self.layer_scale_init_value,
                drop_rate=self.drop_rate,
                drop_path_rate=dpr[i],
                qkv_bias=self.qkv_bias,
                norm_cfg=self.norm_cfg,
            ))

        # Final Normalization
        self.final_norm = nn.LayerNorm(self.arch_settings['embed_dims'])

        self.cls_head = VisionTransformerClsHead(
            num_classes=num_classes,
            in_channels=self.arch_settings['embed_dims'],
            hidden_dim=None,
            act_cfg={'type': "GELU"}
        )


    def _format_output(self, x, patch_resolution):
        """
        Format the output based on `out_type`.

        Args:
            x (Tensor): Input tensor of shape (B, N, D).
            patch_resolution (tuple): Patch resolution (H_patches, W_patches).

        Returns:
            Tensor: Formatted output.
        """
        if self.out_type == 'raw':
            return x
        elif self.out_type == 'cls_token':
            return x[:, 0]
        elif self.out_type == 'featmap':
            B, N, D = x.shape
            H, W = patch_resolution
            return x[:, 1:].reshape(B, H, W, D).permute(0, 3, 1, 2)
        elif self.out_type == 'avg_featmap':
            return x[:, 1:].mean(dim=1)
        else:
            raise ValueError(f"Unsupported out_type: {self.out_type}")

    def forward(self, x):
        """
        Forward pass for VisionTransformer.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            tuple: Contains outputs from specified layers.
        """
        # Input and Patch Embedding
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        # Add Class Token
        cls_token = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)

        # Add Position Embedding
        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode='bicubic',
            num_extra_tokens=self.num_extra_tokens,
        )

        # Dropout after Position Embedding
        x = self.drop_after_pos(x)

        # Pre-normalization
        x = self.pre_norm(x)

        # Transformer Layers
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.final_norm(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        cls_score = self.cls_head(x)
        return cls_score

        # return tuple(outs)

if __name__ == '__main__':
    def test_vision_transformer():
        # 初始化模型
        model = VisionTransformer_Color_Merge(
            arch='base',  # 使用 'base' 架构
            img_size=224,  # 输入图像大小为 224x224
            patch_size=16,  # Patch 大小为 16x16
            in_channels=6,  # 输入通道数为 3 (RGB)
            out_indices=[2, 5, 8, 11],  # 输出第 3、6、9、12 层的特征
            out_type='featmap',  # 输出类型为特征图
            drop_rate=0.1,  # Dropout 率为 0.1
            drop_path_rate=0.1,  # Drop Path 率为 0.1
            qkv_bias=True,  # 启用 qkv 偏置
            norm_cfg={'type': 'LN'},  # 使用 LayerNorm
            layer_scale_init_value=0.,  # 层尺度初始值为 0
        )

        # 打印模型结构
        print(model)

        # 准备输入数据
        batch_size = 2  # 批大小为 2
        in_channels = 6  # 输入通道数为 3 (RGB)
        img_size = 224  # 输入图像大小为 224x224
        x = torch.randn(batch_size, in_channels, img_size, img_size)  # 随机输入张量

        # 前向传播
        outputs = model(x)

        # 检查输出
        print("\nOutputs:")
        for i, out in enumerate(outputs):
            print(f"Layer {model.out_indices[i]} output shape: {out.shape}")

        # 验证输出形状
        expected_shapes = [
            (batch_size, model.arch_settings['embed_dims'], 14, 14),  # 第 3 层特征图
            (batch_size, model.arch_settings['embed_dims'], 14, 14),  # 第 6 层特征图
            (batch_size, model.arch_settings['embed_dims'], 14, 14),  # 第 9 层特征图
            (batch_size, model.arch_settings['embed_dims'], 14, 14),  # 第 12 层特征图
        ]
        for out, expected_shape in zip(outputs, expected_shapes):
            assert out.shape == expected_shape, f"Output shape mismatch: {out.shape} vs {expected_shape}"

        print("\nTest passed! All output shapes are correct.")


    if __name__ == "__main__":
        test_vision_transformer()
