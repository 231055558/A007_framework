import numpy as np

from dataset.transform.basetransform import BaseTransform
class Preprocess(BaseTransform):
    def __init__(self, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
        assert len(mean) == 3 and len(std) == 3
        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def transform(self, results):
        img = results['img']
        img = img[:, :, ::-1].astype(np.float32)
        img = (img - self.mean) / self.std
        results['img'] = img
        return results

