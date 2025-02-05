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

    # def transform(self, results):
    #     img = results['img']
    #     if self.padding:
    #         img = self.pad_image(img, self.padding)
    #
    #     if self.pad_if_needed:
    #         img = self.pad_if_small(img)
    #
    #     offset_h, offset_w = self.rand_crop_params(img)
    #     crop_h, crop_w = self.crop_size
    #
    #     cropped_img = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w]
    #
    #     results['img'] = cropped_img
    #     results['crop_size'] = cropped_img.shape
    #     return results

    def transform(self, results):
        img = results['img']

        # **优化1：减少内存拷贝，使用 NumPy 视图**
        if self.padding:
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        # **优化2：尽量减少 pad_if_small 的计算**
        if self.pad_if_needed:
            img_h, img_w = img.shape[:2]
            crop_h, crop_w = self.crop_size
            if img_h < crop_h or img_w < crop_w:
                pad_h = max(0, crop_h - img_h)
                pad_w = max(0, crop_w - img_w)
                img = np.pad(img, ((pad_h // 2, pad_h - pad_h // 2),
                                   (pad_w // 2, pad_w - pad_w // 2),
                                   (0, 0)), mode='constant')

        # **优化3：随机裁剪点计算**
        img_h, img_w = img.shape[:2]
        crop_h, crop_w = self.crop_size

        if img_h > crop_h:
            offset_h = np.random.randint(0, img_h - crop_h + 1)
        else:
            offset_h = 0

        if img_w > crop_w:
            offset_w = np.random.randint(0, img_w - crop_w + 1)
        else:
            offset_w = 0

        # **优化4：确保 NumPy 视图减少不必要的数据复制**
        cropped_img = img[offset_h:offset_h + crop_h, offset_w:offset_w + crop_w].copy()

        results['img'] = cropped_img
        results['crop_size'] = cropped_img.shape

        return results
