import torch
from torch import nn


class DropPath(nn.Module):
    """
    DropPath (Stochastic Depth) implementation.

    Args:
        drop_prob (float): Probability of dropping the path.
    """

    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # (B, 1, 1, ...)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob


def build_dropout(dropout_cfg):
    """
    Build dropout layer based on configuration.

    Args:
        dropout_cfg (dict): Dropout configuration. For example:
            - {'type': 'Dropout', 'drop_prob': 0.1}
            - {'type': 'DropPath', 'drop_prob': 0.1}

    Returns:
        nn.Module: Dropout layer.
    """
    if dropout_cfg is None:
        return nn.Identity()

    dropout_type = dropout_cfg.get('type', 'Dropout')
    drop_prob = dropout_cfg.get('drop_prob', 0.)

    if dropout_type == 'Dropout':
        return nn.Dropout(p=drop_prob)
    elif dropout_type == 'DropPath':
        return DropPath(drop_prob)
    else:
        raise ValueError(f"Unsupported dropout type: {dropout_type}")
