import torch
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

class AdaptiveNormalizeN(BaseTransform):
    def __init__(self, eps=1e-7):
        """
        Args:
            eps (float): 防除零小量，默认为 1e-7
            clip_value (float): 归一化后数值的截断阈值, 默认为3.0
            gamma (float): gamma矫正系数, 默认为0.5
        """
        super().__init__()
        self.eps = eps
        self.clip_value = 3.0#测试
        self.gamma = 0.5#测试

    def _enhance_details(self, img):#测试
        """增强局部细节"""
        # 使用高斯模糊获取低频分量
        blurred = torch.nn.functional.avg_pool2d(img, kernel_size=15, stride=1, padding=7)
        # 高频分量 = 原图 - 低频分量
        details = img - blurred
        # 增强高频分量
        enhanced = img + 0.8 * details
        return enhanced#测试

    def transform(self, results):
        img = results['img']  # 输入应为 Tensor，形状 (C, H, W)

        
        # 自适应gamma校正
        channel_means = torch.mean(img, dim=(1,2), keepdim=True)
        gamma_corrected = torch.pow(img / (channel_means + self.eps), self.gamma)#测试
        # 增强局部细节
        img = self._enhance_details(gamma_corrected)#测试

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
        #normalized_img = (img - channel_means[:, None, None]) / channel_stds[:, None, None]
        normalized_img =  (img - channel_means[:,None,None]) / channel_stds[:,None,None]#测试
        # 添加阈值限制防止过曝（可选）
        normalized_img = torch.clamp(normalized_img, -self.clip_value, self.clip_value)

        results['img'] = normalized_img
        return results
        
class NormalNormalize(BaseTransform):
    def __init__(self):
        super().__init__()

    def transform(self, results):
        img = results['img']  # 输入应为 Tensor，形状 (C, H, W)
        
        # 简单归一化到[0,1]范围
        normalized_img = img.float() / 255.0
        
        results['img'] = normalized_img
        return results