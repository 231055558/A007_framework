import cv2
import numpy as np
from dataset.transform.basetransform import BaseTransform


class CenterCrop(BaseTransform):
    def __init__(self, output_size=None):
        """
        Args:
            output_size (tuple, optional): (height, width) 指定输出的大小，默认为 None（保持裁剪后大小）
        """
        self.output_size = output_size

    def transform(self, results):
        img = results['img']
        h, w, c = img.shape

        # 裁剪部分的代码保持不变
        crop_size = min(h, w)
        if h > w:
            start_h = (h - crop_size) // 2
            start_w = 0
        else:
            start_h = 0
            start_w = (w - crop_size) // 2

        cropped_img = img[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

        # 新增：如果指定了 output_size，将裁剪后的图像调整到该大小
        if self.output_size is not None:
            cropped_img = cv2.resize(cropped_img, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)

        # 更新结果
        results['img'] = cropped_img
        results['crop_size'] = cropped_img.shape

        return results

class Center_Roi_Crop(BaseTransform):
    def __init__(self, output_size=None):
        """
        Args:
            output_size (tuple, optional): (height, width) 指定输出的大小，默认为 None（保持裁剪后大小）
        """
        self.output_size = output_size

    def get_crop_params(self, img, image_path):
        """
        根据 image_path 的关键词计算裁剪参数。

        Args:
            img (np.ndarray): 输入图像，形状为 (H, W, C)。
            image_path (str): 图像路径，可能包含关键词（如 "left" 或 "right"）。

        Returns:
            start_h, start_w, crop_size: 裁剪的起始位置和裁剪尺寸。
        """
        h, w, _ = img.shape
        crop_size = min(h, w)  # 裁剪尺寸为短边长度

        # 如果 image_path 包含 "left"
        if "left" in image_path:
            start_h = (h - crop_size) // 2  # 垂直居中
            start_w = 16  # 从左侧开始

        # 如果 image_path 包含 "right"
        elif "right" in image_path:
            start_h = (h - crop_size) // 2  # 垂直居中
            start_w = w - crop_size - 16  # 从右侧开始

        # 默认中心裁剪
        else:
            print("warning")
            if h > w:
                start_h = (h - crop_size) // 2
                start_w = 0
            else:
                start_h = 0
                start_w = (w - crop_size) // 2

        return start_h, start_w, crop_size

    def transform(self, results):
        """
        对图像进行裁剪。

        Args:
            results (dict): 包含 'img' 和 'image_path' 键的字典，'img' 是 NumPy 数组，形状为 (H, W, C)。

        Returns:
            results (dict): 裁剪后的图像存储在 'img' 键中。
        """
        img = results['img']
        image_path = results['image_path']

        # 获取裁剪参数
        start_h, start_w, crop_size = self.get_crop_params(img, image_path)

        # 裁剪图像
        cropped_img = img[start_h:start_h + crop_size, start_w:start_w + crop_size, :]

        # 如果指定了 output_size，将裁剪后的图像调整到该大小
        if self.output_size is not None:
            cropped_img = cv2.resize(cropped_img, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)

        # 更新结果
        results['img'] = cropped_img
        results['crop_size'] = cropped_img.shape

        return results
