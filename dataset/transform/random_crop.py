import cv2
import numpy as np
from dataset.transform.basetransform import BaseTransform


import numpy as np

class RandomCrop(BaseTransform):
    def __init__(self, output_size=None, pad_if_needed=False, padding=None, pad_val=0):
        """
        Args:
            output_size (tuple, optional): (height, width) 指定输出的大小，默认为 None（保持裁剪后大小）
            pad_if_needed (bool): 如果图像尺寸小于裁剪尺寸，是否进行填充
            padding (tuple, optional): (top, bottom, left, right) 指定额外的填充大小
            pad_val (int): 填充值，默认为 0（黑色填充）
        """
        self.output_size = output_size
        self.pad_if_needed = pad_if_needed
        self.padding = padding
        self.pad_val = pad_val

    def pad_image(self, img, padding):
        """对图像进行填充"""
        h, w, c = img.shape
        top, bottom, left, right = padding
        new_h, new_w = h + top + bottom, w + left + right

        padded_img = np.full((new_h, new_w, c), self.pad_val, dtype=img.dtype)
        padded_img[top:top+h, left:left+w] = img
        return padded_img

    def pad_if_small(self, img):
        """如果图像尺寸小于裁剪尺寸，进行填充"""
        h, w, c = img.shape
        # 裁剪尺寸为短边长度
        crop_size = min(h, w)
        pad_top = max(0, (crop_size - h) // 2)
        pad_bottom = max(0, crop_size - h - pad_top)
        pad_left = max(0, (crop_size - w) // 2)
        pad_right = max(0, crop_size - w - pad_left)

        return self.pad_image(img, (pad_top, pad_bottom, pad_left, pad_right))

    def rand_crop_params(self, img):
        """计算随机裁剪的起始点"""
        h, w, _ = img.shape
        crop_size = min(h, w)  # 裁剪尺寸为短边长度

        if h > w:
            # 长边是高度，随机裁剪高度
            offset_h = np.random.randint(0, h - crop_size + 1)
            offset_w = 0
        else:
            # 长边是宽度，随机裁剪宽度
            offset_h = 0
            offset_w = np.random.randint(0, w - crop_size + 1)

        return offset_h, offset_w, crop_size

    def transform(self, results):
        img = results['img']

        # 裁剪部分的代码保持不变
        if self.padding:
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        if self.pad_if_needed:
            img_h, img_w = img.shape[:2]
            crop_size = min(img_h, img_w)
            if img_h < crop_size or img_w < crop_size:
                img = self.pad_if_small(img)

        offset_h, offset_w, crop_size = self.rand_crop_params(img)
        cropped_img = img[offset_h:offset_h + crop_size, offset_w:offset_w + crop_size].copy()

        # 新增：如果指定了 output_size，将裁剪后的图像调整到该大小
        if self.output_size is not None:
            cropped_img = cv2.resize(cropped_img, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)

        # 更新结果
        results['img'] = cropped_img
        results['crop_size'] = cropped_img.shape

        return results

