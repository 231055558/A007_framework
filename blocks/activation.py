from torch import nn


def build_activation(act_cfg):
    """
    Build activation function based on configuration.

    Args:
        act_cfg (dict): Activation function configuration. For example:
            - {'type': 'GELU'}

    Returns:
        nn.Module: Activation function.
    """
    act_type = act_cfg.get('type', 'GELU')
    if act_type == 'GELU':
        return nn.GELU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        raise ValueError(f"Unsupported activation type: {act_type}")
