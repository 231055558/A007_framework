from dataset.transform.basetransform import BaseTransform
import numpy as np
import torch

class ToTensor(BaseTransform):
    def transform(self, results):
        img = results['img']
        label = results['label']
        img = img.astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.tensor(label, dtype=torch.float32)

        results['img'] = img
        results['label'] = label
        return results
