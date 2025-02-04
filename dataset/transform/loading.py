import numpy as np
import cv2
from PIL import Image
from dataset.transform.basetransform import BaseTransform

class LoadImageFromFile(BaseTransform):
    def __init__(self,
                 to_float32: bool = False,
                 color_type: str = 'color',
                 imdecode_backend: str = 'cv2',
                 ignore_empty: bool = False):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.imdecode_backend = imdecode_backend
        self.ignore_empty = ignore_empty

    def transform(self, results):
        img_info = results['img_info']
        img_path = img_info['filename']

        if self.imdecode_backend == 'cv2':
            img = self._load_image_cv2(img_path=img_path)
        elif self.imdecode_backend == 'PIL':
            img = self._load_image_pil(img_path=img_path)
        else:
            raise ValueError(f'Unsupported image backend: {self.imdecode_backend}')

        if img is None and self.ignore_empty:
            return None

        if self.to_float32:
            img = img.astype(np.float32)
        results['img'] = img
        return results

    def _load_image_cv2(self, img_path):
        if self.color_type == 'color':
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        elif self.color_type == 'grayscale':
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        elif self.color_type == 'unchanged':
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        else:
            raise ValueError(f'Unsupported color type: {self.color_type}')

        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")

        return img

    def _load_image_pil(self, img_path):
        img = Image.open(img_path)

        if self.color_type == 'color':
            img = img.convert('RGB')
        elif self.color_type == 'grayscale':
            img = img.convert('L')
        elif self.color_type == 'unchanged':
            pass
        else:
            raise ValueError(f'Unsupported color type: {self.color_type}')

        img = np.array(img)
        return img
