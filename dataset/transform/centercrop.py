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
    def __init__(self, output_size=None, pad_if_needed=False, padding=None, pad_val=0):
        super().__init__()
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

    def get_crop_params(self, img, image_path):
        """根据 image_path 的关键词和输出尺寸计算裁剪参数"""
        h, w, _ = img.shape
        output_h, output_w = self.output_size

        # 如果 image_path 包含 "left" 并且输出尺寸为 (512, 512)
        if "left" in image_path:
            # 从左侧中心点开始裁剪
            center_h = h // 2
            start_h = center_h - output_h // 2  # 向上扩展
            end_h = start_h + output_h         # 向下扩展
            start_w = 0                        # 从左侧开始
            end_w = output_w                   # 向右扩展
            random_range = self.output_size[0] // 16
            # 随机微调裁剪区域（32x32 范围）
            # random_offset_h = random.randint(-random_range // 2, random_range // 2)  # 随机偏移高度
            # random_offset_w = random.randint(0, random_range)    # 随机偏移宽度

            start_w += random_range // 2
            end_w += random_range // 2

            # 确保裁剪区域不超出图像边界
            start_h = max(0, start_h)
            end_h = min(h, end_h)
            start_w = max(0, start_w)
            end_w = min(w, end_w)

            return start_h, start_w, end_h, end_w

        # 如果 image_path 包含 "right" 并且输出尺寸为 (512, 512)
        elif "right" in image_path:
            # 从右侧中心点开始裁剪
            center_h = h // 2
            start_h = center_h - output_h // 2  # 向上扩展
            end_h = start_h + output_h         # 向下扩展
            start_w = w - output_w             # 从右侧开始
            end_w = w                          # 向左扩展

            random_range = self.output_size[0] // 16

            start_w -= random_range // 2
            end_w -= random_range // 2

            # 确保裁剪区域不超出图像边界
            start_h = max(0, start_h)
            end_h = min(h, end_h)
            start_w = max(0, start_w)
            end_w = min(w, end_w)

            return start_h, start_w, end_h, end_w

        else:
            assert False


    def transform(self, results):
        img = results['img']
        image_path = results['image_path']

        # 填充处理
        if self.padding:
            img = np.pad(img, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')

        if self.pad_if_needed:
            img_h, img_w = img.shape[:2]
            crop_size = min(img_h, img_w)
            if img_h < crop_size or img_w < crop_size:
                img = self.pad_if_small(img)

        # 获取裁剪参数
        start_h, start_w, end_h, end_w = self.get_crop_params(img, image_path)

        # 裁剪图像
        cropped_img = img[start_h:end_h, start_w:end_w].copy()

        # 调整到输出尺寸
        if self.output_size is not None:
            cropped_img = cv2.resize(cropped_img, self.output_size[::-1], interpolation=cv2.INTER_LINEAR)

        results['img'] = cropped_img
        results['crop_size'] = cropped_img.shape
        return results
