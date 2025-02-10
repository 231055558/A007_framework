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

