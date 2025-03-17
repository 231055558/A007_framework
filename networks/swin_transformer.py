import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple
from blocks.swin_transformer import SwinTransformerBlock
from blocks.head import AttentionFC


class SwinTransformer(nn.Module):
    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_channels=3,
                 num_classes=1000,
                 embed_dim=96,
                 depths=[2, 2, 6, 2],
                 num_heads=[3, 6, 12, 24],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        
        # patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_channels=in_channels, embed_dim=embed_dim)
        
        # position embedding (we use learnable position embedding)
        self.pos_drop = nn.Dropout(p=drop_rate)
        
        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=int(embed_dim * 2 ** i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None
            )
            self.layers.append(layer)
            
        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = AttentionFC(in_features=self.num_features, num_classes=num_classes)
        
    def forward(self, x):
        # x: [B, C, H, W]
        x = self.patch_embed(x)  # [B, N, C]
        x = self.pos_drop(x)
        
        # 重塑维度为 [B, H', W', C]
        B, N, C = x.shape
        H = W = int(N ** 0.5)  # N = H * W
        x = x.view(B, H, W, C)
        
        # 通过Transformer层
        for layer in self.layers:
            x = layer(x)
        
        # 最后的处理
        x = self.norm(x)  # [B, H, W, C]
        x = x.mean(dim=(1, 2))  # Global Average Pooling [B, C]
        x = self.head(x)  # 分类头 [B, num_classes]
        
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=4, in_channels=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]
        
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        
        # 使用卷积进行patch embedding
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]
        
        # 重塑并转置
        x = x.flatten(2).transpose(1, 2)  # [B, H*W/patch_size^2, embed_dim]
        x = self.norm(x)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        
        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                norm_layer=norm_layer
            )
            for i in range(depth)])
            
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None
            
    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class PatchMerging(nn.Module):
    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)
        
    def forward(self, x):
        B, H, W, C = x.shape
        
        # padding
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))
            
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        
        x = self.norm(x)
        x = self.reduction(x)
        
        return x 

# def window_partition(x, window_size):
#     """将特征图划分为多个窗口"""
#     B, H, W, C = x.shape
#     x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
#     windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
#     return windows

# def window_reverse(windows, window_size, H, W):
#     """将窗口还原为特征图"""
#     B = int(windows.shape[0] / (H * W / window_size / window_size))
#     x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
#     x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
#     return x

if __name__ == "__main__":
    # 创建一个简单的测试配置
    config = {
        'img_size': 224,
        'patch_size': 4,
        'in_channels': 3,
        'num_classes': 8,  # 假设是8分类任务
        'embed_dim': 96,
        'depths': [2, 2, 6, 2],
        'num_heads': [3, 6, 12, 24],
        'window_size': 7,
        'mlp_ratio': 4.,
        'qkv_bias': True,
        'drop_rate': 0.1,
        'attn_drop_rate': 0.1,
        'drop_path_rate': 0.1
    }
    
    # 创建模型
    model = SwinTransformer(**config)
    model = model.cuda()
    print("Model created successfully!")
    
    # 创建随机输入数据进行测试
    batch_size = 2
    input_tensor = torch.randn(batch_size, config['in_channels'], 
                             config['img_size'], config['img_size']).cuda()
    print(f"\nInput shape: {input_tensor.shape}")
    
    # 前向传播测试
    try:
        with torch.no_grad():
            output = model(input_tensor)
            print(f"Output shape: {output.shape}")
            print("\nModel forward pass test successful!")
            
            # 打印模型参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\nTotal parameters: {total_params:,}")
            print(f"Trainable parameters: {trainable_params:,}")
            
    except Exception as e:
        print(f"Error during forward pass: {str(e)}")
    
    # 测试模型在不同输入大小下的表现
    test_sizes = [160, 192, 224, 256]
    print("\nTesting different input sizes:")
    
    for size in test_sizes:
        try:
            test_input = torch.randn(1, config['in_channels'], size, size).cuda()
            with torch.no_grad():
                test_output = model(test_input)
                print(f"Input size: {size}x{size}, Output shape: {test_output.shape}")
        except Exception as e:
            print(f"Failed for size {size}x{size}: {str(e)}")
    
    print("\nTest completed!") 