import numpy as np
from torchvision.transforms import Normalize
from dataset.transform.basetransform import BaseTransform
class Preprocess(BaseTransform):
    # 必须放在ToTensor后面
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        assert len(mean) == 3 and len(std) == 3
        self.normalize = Normalize(mean = [ x / 255.0 for x in mean], std = [ y / 255.0 for y in std ])

    def transform(self, results):
        img = results['img']
        img = self.normalize(img)
        results['img'] = img
        return results


from torchvision.transforms import Normalize
import torch


class AdaptiveNormalize(BaseTransform):
    def __init__(self, eps=1e-7):
        """
        Args:
            eps (float): 防除零小量，默认为 1e-7
        """
        super().__init__()
        self.eps = eps

    def transform(self, results):
        img = results['img']  # 输入应为 Tensor，形状 (C, H, W)

        # 计算每个通道的均值和标准差
        channel_means = torch.mean(img, dim=(1, 2))  # shape (C,)
        channel_stds = torch.std(img, dim=(1, 2))  # shape (C,)

        # 防除零处理：标准差为0时设为1
        channel_stds = torch.where(
            channel_stds < self.eps,
            torch.ones_like(channel_stds),
            channel_stds
        )

        # 自适应归一化
        normalized_img = (img - channel_means[:, None, None]) / channel_stds[:, None, None]

        results['img'] = normalized_img
        return results


