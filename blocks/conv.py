import torch.nn as nn
class Conv2dModule(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True,
                 padding_mode='zeros',
                 activation='relu',
                 norm='batch_norm'):
        super(Conv2dModule, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'leaky_relu':
            self.activation = nn.LeakyReLU(inplace=True)
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        if norm == 'batch_norm':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'instance_norm':
            self.norm = nn.InstanceNorm2d(out_channels)
        elif norm == 'layer_norm':
            self.norm = nn.LayerNorm(out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f"Unsupported norm: {norm}")

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x
