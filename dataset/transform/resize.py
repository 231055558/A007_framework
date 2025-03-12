import numpy as np
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
        results['img_shape'] = resized_img.shape
        return results
