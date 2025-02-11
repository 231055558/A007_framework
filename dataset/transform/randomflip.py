import numpy as np

from dataset.transform.basetransform import BaseTransform
class RandomFlip(BaseTransform):
    def __init__(self, flip_horizontal=True, flip_vertaical=True, flip_prob=0.5):
        assert 0.0 <= flip_prob <= 1.0
        self.flip_horizontal = flip_horizontal
        self.flip_vertaical = flip_vertaical
        self.flip_prob = flip_prob

    def transform(self, results):
        img = results['img']
        if self.flip_horizontal and np.random.rand() < self.flip_prob:
            img = img[:, ::-1, :]

        if self.flip_vertaical and np.random.rand() < self.flip_prob:
            img = img[::-1, :, :]

        results['img'] = img
        return results
