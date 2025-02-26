import cv2

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
