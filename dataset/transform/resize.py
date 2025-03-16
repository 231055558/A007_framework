import numpy as np
import cv2
from dataset.transform.basetransform import BaseTransform
from torchvision.transforms import Resize as torch_resize

class Resize(BaseTransform):
    def __init__(self, target_size=None, scale_factor=None):
        assert target_size or scale_factor
        assert not (target_size and scale_factor)

        self.target_size = target_size
        self.scale_factor = scale_factor
        if target_size is not None:
            self.resize_function = torch_resize(self.target_size, antialias=None)

    def transform(self, results):
        img = results['img']
        h, w, c = img.shape

        if self.target_size:
            resized_img = self.resize_function(img)
        else:
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            resized_img = np.zeros((new_h, new_w, c), dtype=img.dtype)
            for i in range(new_h):
                for j in range(new_w):
                    orig_x = int(j / new_w * w)
                    orig_y = int(i / new_h * h)
                    resized_img[i, j] = img[orig_y, orig_x]

        results['img'] = resized_img
        return results


class Resize_Numpy(BaseTransform):
    def __init__(self, target_size=None, scale_factor=None, interpolation=cv2.INTER_LINEAR):
        """
        初始化 Resize 类。

        Args:
            target_size (tuple): 目标尺寸，格式为 (height, width)。
            scale_factor (float): 缩放因子。
            interpolation (int): 插值方法，默认为 cv2.INTER_LINEAR。
        """
        assert target_size or scale_factor, "必须指定 target_size 或 scale_factor"
        assert not (target_size and scale_factor), "不能同时指定 target_size 和 scale_factor"

        self.target_size = target_size
        self.scale_factor = scale_factor
        self.interpolation = interpolation

    def transform(self, results):
        """
        对图像进行缩放。

        Args:
            results (dict): 包含 'img' 键的字典，'img' 是 NumPy 数组，形状为 (H, W, C)。

        Returns:
            results (dict): 缩放后的图像存储在 'img' 键中。
        """
        img = results['img']
        h, w, c = img.shape

        if self.target_size:
            # 根据 target_size 缩放图像
            resized_img = cv2.resize(img, (self.target_size[1], self.target_size[0]), interpolation=self.interpolation)
        else:
            # 根据 scale_factor 缩放图像
            new_h, new_w = int(h * self.scale_factor), int(w * self.scale_factor)
            resized_img = cv2.resize(img, (new_w, new_h), interpolation=self.interpolation)

        results['img'] = resized_img
        return results
