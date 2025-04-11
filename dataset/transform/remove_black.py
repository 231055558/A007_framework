import numpy as np
import cv2
from PIL import Image
from dataset.transform.basetransform import BaseTransform

class RemoveBlackBorder(BaseTransform):
    def __init__(self, threshold=5):
        """
        移除图像黑边的独立transform操作
        Args:
            threshold (int): 黑色像素阈值，默认5
        """
        super().__init__()
        self.threshold = threshold

    def transform(self, results):
        img = results['img']
        
        if len(img.shape) == 2:  # 灰度图
            gray = img
        else:  # 彩色图
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.shape[2] == 3 else img[:,:,0]
        
        # 找到非黑色区域边界
        rows = np.any(gray > self.threshold, axis=1)
        cols = np.any(gray > self.threshold, axis=0)
        
        if not np.any(rows) or not np.any(cols):
            return results
            
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        
        # 裁剪图像并更新结果
        results['img'] = img[rmin:rmax+1, cmin:cmax+1]
        return results