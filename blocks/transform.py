import torch
import torch.nn as nn
import torch.nn.functional as F

from blocks.activation import build_activation
from blocks.dropout import build_dropout


import torch
import torch.nn.functional as F

def resize_pos_embed(pos_embed, src_shape, dst_shape, mode='bicubic', num_extra_tokens=1):
    """
    Resize position embedding to match the target resolution.

    Args:
        pos_embed (Tensor): Position embedding tensor of shape (1, N, D).
        src_shape (tuple): Source patch resolution (H_src, W_src).
        dst_shape (tuple): Target patch resolution (H_dst, W_dst).
        mode (str): Interpolation mode ('nearest', 'bilinear', 'bicubic').
        num_extra_tokens (int): Number of extra tokens (e.g., CLS token).

    Returns:
        Tensor: Resized position embedding of shape (1, M, D), where M = H_dst * W_dst + num_extra_tokens.
    """
    # Extract the position embedding for patches (excluding extra tokens)
    pos_embed_patches = pos_embed[:, num_extra_tokens:]  # (1, N_patches, D)

    # Reshape to 2D grid
    pos_embed_patches = pos_embed_patches.reshape(1, src_shape[0], src_shape[1], -1).permute(0, 3, 1, 2)  # (1, D, H_src, W_src)

    # Resize to target resolution
    pos_embed_patches = F.interpolate(
        pos_embed_patches,
        size=dst_shape,
        mode=mode,
        align_corners=False if mode != 'nearest' else None,
    )  # (1, D, H_dst, W_dst)

    # Reshape back to 1D sequence
    pos_embed_patches = pos_embed_patches.permute(0, 2, 3, 1).reshape(1, -1, pos_embed.shape[-1])  # (1, N_dst, D)

    # Combine with extra tokens (e.g., CLS token)
    if num_extra_tokens > 0:
        pos_embed_extra = pos_embed[:, :num_extra_tokens]  # (1, num_extra_tokens, D)
        pos_embed = torch.cat([pos_embed_extra, pos_embed_patches], dim=1)  # (1, N_dst + num_extra_tokens, D)
    else:
        pos_embed = pos_embed_patches

    return pos_embed


class PatchEmbed(nn.Module):
    """
    Patch Embedding module for Vision Transformer (ViT).

    Args:
        img_size (int or tuple): Input image size (height, width).
        patch_size (int or tuple): Patch size (height, width).
        in_channels (int): Number of input channels (default: 3 for RGB).
        embed_dim (int): Embedding dimension for each patch.
        norm_layer (nn.Module, optional): Normalization layer (default: None).
        flatten (bool): Whether to flatten the patch embeddings (default: True).
    """

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        norm_layer=None,
        flatten=True,
    ):
        super().__init__()
        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.patch_size = (patch_size, patch_size) if isinstance(patch_size, int) else patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.flatten = flatten

        # Calculate number of patches
        self.grid_size = (
            self.img_size[0] // self.patch_size[0],
            self.img_size[1] // self.patch_size[1],
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        # Projection: Convolution to convert patches into embeddings
        self.proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        # Optional normalization layer
        self.norm = norm_layer(embed_dim) if norm_layer is not None else None

    def forward(self, x):
        """
        Forward pass for PatchEmbed.

        Args:
            x (Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            Tensor: Patch embeddings of shape (B, N, D), where N is the number of patches
                   and D is the embedding dimension.
            Tuple[int, int]: Grid size (H_patches, W_patches).
        """
        # Check input size
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model's expected size ({self.img_size[0]}*{self.img_size[1]})."

        # Project patches into embedding space
        x = self.proj(x)  # (B, embed_dim, H_patches, W_patches)

        # Flatten and transpose if needed
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # (B, N, embed_dim)

        # Apply normalization if defined
        if self.norm is not None:
            x = self.norm(x)

        return x, self.grid_size


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer for Vision Transformer (ViT).

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        feedforward_channels (int): Dimension of the feedforward network.
        layer_scale_init_value (float, optional): Initial value for layer scale. Default: 0.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Drop path rate for stochastic depth. Default: 0.
        num_fcs (int, optional): Number of fully connected layers in FFN. Default: 2.
        qkv_bias (bool, optional): Whether to add bias to the qkv projection. Default: True.
        ffn_type (str, optional): Type of feedforward network ('origin' or 'swiglu_fused'). Default: 'origin'.
        act_cfg (dict, optional): Configuration for activation function. Default: {'type': 'GELU'}.
        norm_cfg (dict, optional): Configuration for normalization layer. Default: {'type': 'LN'}.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        layer_scale_init_value=0.,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.,
        num_fcs=2,
        qkv_bias=True,
        ffn_type='origin',
        act_cfg={'type': 'GELU'},
        norm_cfg={'type': 'LN'},
    ):
        super(TransformerEncoderLayer, self).__init__()
        self.embed_dims = embed_dims

        # Layer normalization 1
        self.norm1 = nn.LayerNorm(embed_dims)

        # Multi-head self-attention
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            dropout_layer={'type': 'DropPath', 'drop_prob': drop_path_rate},
            qkv_bias=qkv_bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        # Layer normalization 2
        self.norm2 = nn.LayerNorm(embed_dims)

        # Feedforward network (FFN)
        if ffn_type == 'origin':
            self.ffn = FFN(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                num_fcs=num_fcs,
                ffn_drop=drop_rate,
                dropout_layer={'type': 'DropPath', 'drop_prob': drop_path_rate},
                act_cfg=act_cfg,
                layer_scale_init_value=layer_scale_init_value,
            )
        elif ffn_type == 'swiglu_fused':
            self.ffn = SwiGLUFFNFused(
                embed_dims=embed_dims,
                feedforward_channels=feedforward_channels,
                layer_scale_init_value=layer_scale_init_value,
            )
        else:
            raise NotImplementedError(f"Unsupported FFN type: {ffn_type}")

    def forward(self, x):
        """
        Forward pass for TransformerEncoderLayer.

        Args:
            x (Tensor): Input tensor of shape (B, N, D), where B is batch size,
                       N is sequence length, and D is embedding dimension.

        Returns:
            Tensor: Output tensor of shape (B, N, D).
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x))

        # Feedforward network with residual connection
        x = x + self.ffn(self.norm2(x))

        return x

class MultiheadAttention(nn.Module):
    """
    Multi-head Self-attention Layer.

    Args:
        embed_dims (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        attn_drop (float): Attention dropout rate.
        proj_drop (float): Projection dropout rate.
        dropout_layer (dict): Dropout layer configuration.
        qkv_bias (bool): Whether to add bias to the qkv projection.
        layer_scale_init_value (float): Initial value for layer scale.
    """

    def __init__(
        self,
        embed_dims,
        num_heads,
        attn_drop=0.,
        proj_drop=0.,
        dropout_layer={'type': 'DropPath', 'drop_prob': 0.},
        qkv_bias=True,
        layer_scale_init_value=0.,
    ):
        super(MultiheadAttention, self).__init__()
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.head_dims = embed_dims // num_heads

        # Linear projections for q, k, v
        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop)

        # Dropout layer (e.g., DropPath)
        self.dropout_layer = build_dropout(dropout_layer)

        # Layer scale (optional)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(embed_dims)) \
            if layer_scale_init_value > 0 else None

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dims).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        attn = (q @ k.transpose(-2, -1)) / (self.head_dims ** 0.5)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)

        # Projection
        x = self.proj(x)
        x = self.proj_drop(x)

        # Dropout layer
        x = self.dropout_layer(x)

        # Layer scale (optional)
        if self.layer_scale is not None:
            x = x * self.layer_scale

        return x


class FFN(nn.Module):
    """
    Feedforward Network (FFN) for Transformer.

    Args:
        embed_dims (int): Embedding dimension.
        feedforward_channels (int): Dimension of the feedforward network.
        num_fcs (int): Number of fully connected layers.
        ffn_drop (float): Dropout rate for FFN.
        dropout_layer (dict): Dropout layer configuration.
        act_cfg (dict): Activation function configuration.
        layer_scale_init_value (float): Initial value for layer scale.
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        num_fcs=2,
        ffn_drop=0.,
        dropout_layer={'type': 'DropPath', 'drop_prob': 0.},
        act_cfg={'type': 'GELU'},
        layer_scale_init_value=0.,
    ):
        super(FFN, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        # Fully connected layers
        self.fcs = nn.ModuleList()
        for _ in range(num_fcs):
            self.fcs.append(nn.Linear(embed_dims, feedforward_channels))
            self.fcs.append(nn.Linear(feedforward_channels, embed_dims))

        # Dropout layer
        self.dropout = build_dropout(dropout_layer)

        # Activation function
        self.act = build_activation(act_cfg)

        # Layer scale (optional)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(embed_dims)) \
            if layer_scale_init_value > 0 else None

    def forward(self, x, identity=None):
        """
        Forward pass for FFN.

        Args:
            x (Tensor): Input tensor of shape (B, N, D).
            identity (Tensor, optional): Identity tensor for residual connection.

        Returns:
            Tensor: Output tensor of shape (B, N, D).
        """
        out = x
        for fc in self.fcs:
            out = fc(out)
            out = self.act(out)
            out = self.dropout(out)

        # Residual connection
        if identity is not None:
            out = out + identity

        # Layer scale (optional)
        if self.layer_scale is not None:
            out = out * self.layer_scale

        return out

class SwiGLUFFNFused(nn.Module):
    """
    SwiGLU Fused Feedforward Network for Transformer.

    Args:
        embed_dims (int): Embedding dimension.
        feedforward_channels (int): Dimension of the feedforward network.
        layer_scale_init_value (float): Initial value for layer scale.
    """

    def __init__(
        self,
        embed_dims,
        feedforward_channels,
        layer_scale_init_value=0.,
    ):
        super(SwiGLUFFNFused, self).__init__()
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels

        # SwiGLU layer
        self.fc1 = nn.Linear(embed_dims, feedforward_channels * 2)
        self.fc2 = nn.Linear(feedforward_channels, embed_dims)

        # Layer scale (optional)
        self.layer_scale = nn.Parameter(layer_scale_init_value * torch.ones(embed_dims)) \
            if layer_scale_init_value > 0 else None

    def forward(self, x, identity=None):
        """
        Forward pass for SwiGLUFFNFused.

        Args:
            x (Tensor): Input tensor of shape (B, N, D).
            identity (Tensor, optional): Identity tensor for residual connection.

        Returns:
            Tensor: Output tensor of shape (B, N, D).
        """
        out = self.fc1(x)
        out = F.silu(out[:, :, :self.feedforward_channels]) * out[:, :, self.feedforward_channels:]
        out = self.fc2(out)

        # Residual connection
        if identity is not None:
            out = out + identity

        # Layer scale (optional)
        if self.layer_scale is not None:
            out = out * self.layer_scale

        return out
