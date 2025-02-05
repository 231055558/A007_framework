import numpy as np
from dataset.transform.basetransform import BaseTransform


class RandomCrop(BaseTransform):
    def __init__(self, crop_size, pad_if_needed=False, padding=None, pad_val=0):
        """
        Args:
            crop_size (tuple): (height, width) 指定裁剪的大小
            pad_if_needed (bool): 如果图像尺寸小于裁剪尺寸，是否进行填充
            padding (tuple, optional): (top, bottom, left, right) 指定额外的填充大小
            pad_val (int): 填充值，默认为 0（黑色填充）
        """
        self.crop_size = crop_size
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
        h, w, c = img.shape
        crop_h, crop_w = self.crop_size
        pad_top = max(0, (crop_h - h) // 2)
        pad_bottom = max(0, crop_h - h - pad_top)
        pad_left = max(0, (crop_w - w) // 2)
        pad_right = max(0, crop_w - w - pad_left)

        return self.pad_image(img, (pad_top, pad_bottom, pad_left, pad_right))

    def rand_crop_params(self, img):
        h, w, _ = img.shape
        crop_h, crop_w = self.crop_size

        max_h = max(0, h - crop_h)
        max_w = max(0, w - crop_w)

        offset_h = np.random.randint(0, max_h + 1) if max_h > 0 else 0
        offset_w = np.random.randint(0, max_w + 1) if max_w > 0 else 0

        return offset_h, offset_w

    def transform(self, results):
        img = results['img']
        if self.padding:
            img = self.pad_image(img, self.padding)

        if self.pad_if_needed:
            img = self.pad_if_small(img)

        offset_h, offset_w = self.rand_crop_params(img)
        crop_h, crop_w = self.crop_size

        cropped_img = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]

        results['img'] = cropped_img
        results['crop_size'] = cropped_img.shape
        return results
