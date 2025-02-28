import os
import cv2
import random
import numpy as np
from functools import lru_cache
from dataset.transform import BaseTransform


class RandomColorTransfer(BaseTransform):
    def __init__(
            self,
            source_image_dir: str,
            prob: float = 0.5,
            max_samples: int = 1000,
            cache_size: int = 500
    ):
        """
        Args:
            source_image_dir (str): 存放颜色源图片的文件夹路径
            prob (float): 应用颜色迁移的概率，默认为 0.5
            max_samples (int): 预处理的源图像最大数量（避免内存溢出），默认为 1000
            cache_size (int): 缓存最近加载的源图像数量（加快处理速度），默认为 500
        """
        super().__init__()
        self.prob = prob
        self.source_image_dir = source_image_dir
        self.max_samples = max_samples
        self.source_stats = self._precompute_source_stats()

        self._get_random_source_path = lru_cache(maxsize=cache_size)(self._get_random_source_path)

    def _precompute_source_stats(self):
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp'}
        all_files = [
            f for f in os.listdir(self.source_image_dir)
            if os.path.splitext(f)[1].lower() in valid_exts
        ]

        selected_files = random.sample(
            all_files,
            k=min(self.max_samples, len(all_files))
        )

        stats = []
        for filename in selected_files:
            img_path = os.path.join(self.source_image_dir, filename)
            img = cv2.imread(img_path)
            if img is not None:
                lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
                mean, std = cv2.meanStdDev(lab)
                stats.append((
                    mean.reshape(3).astype(np.float32),  # shape (3,)
                    std.reshape(3).astype(np.float32)     # shape (3,)
                ))
        return stats

    def _get_random_source_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """随机选择一个预存的源图像统计量"""
        return random.choice(self.source_stats)

    def color_transfer(self, target_img: np.ndarray) -> np.ndarray:
        """基于预存的源统计量进行颜色迁移"""
        # 随机选择一个源统计量
        source_mean, source_std = self._get_random_source_path()

        # 转换目标图像到 LAB 空间
        target_lab = cv2.cvtColor(target_img, cv2.COLOR_BGR2LAB)

        # 计算目标的均值和标准差
        target_mean, target_std = cv2.meanStdDev(target_lab)
        target_mean = target_mean.reshape(3).astype(np.float32)
        target_std = target_std.reshape(3).astype(np.float32)

        # 标准化目标图像
        target_normalized = (target_lab.astype(np.float32) - target_mean) / (target_std + 1e-10)

        # 应用源图像的统计量
        result_lab = (target_normalized * source_std) + source_mean
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

        # 转换回 BGR
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def transform(self, results: dict) -> dict:
        if random.random() < self.prob:
            target_img = results['img']
            transferred_img = self.color_transfer(target_img)
            results['img'] = transferred_img
        return results
