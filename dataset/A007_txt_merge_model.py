'''
可以通过txt文本读取A007单图像数据集的基本信息
要求把数据整理成如下格式
root_path
train.txt
val.txt
train(文件夹，存放图片)
val(文件夹，存放图片)

其中train.txt/val.txt格式如下:
0_left.jpg 0_right.jpg 00010000
1_left.jpg 1_right.jpg 10000000
2_left.jpg 2_right.jpg 01000001
3_left.jpg 3_right.jpg 00000001

'''

import os
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader

from dataset.A007_txt import ImageInfo




class A007Dataset:
    def __init__(
            self,
            txt_file: str,
            root_dir: str,
            transform: Optional[Callable] = None,
            seed: Optional[int] = 42,
            preload: bool = True  # 新增参数：是否预加载数据到内存
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_infos = []
        self.preloaded_data = []  # 新增：存储预处理后的数据（图像和标签）

        if seed is not None:
            random.seed(seed)

        self._load_data(txt_file)

        # 预加载数据到内存
        if preload:
            self._preload_all_data()

    def _load_data(self, txt_file: str):
        txt_path = os.path.join(self.root_dir, txt_file)
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"File {txt_path} not exists!")

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 3:
                    continue
                image_path_l = os.path.join(self.root_dir, 'images', parts[0])
                image_path_r = os.path.join(self.root_dir, 'images', parts[1])
                label = [int(x) for x in parts[2]]
                self.image_infos.append([ImageInfo(image_path_l, label), ImageInfo(image_path_r, label)])

    def _preload_all_data(self):
        """
        预加载所有数据到内存，并应用静态预处理（如 Resize、Normalize）。
        """

        for img_info in self.image_infos:
            # image = cv2.imread(img_info.data["image_path"])
            # if image is None:
            #     raise ValueError(f"Failed to load image {img_info.data['image_path']}")

            # 应用静态预处理
            results = self.transform(img_info)  # 转换为 Tensor 并归一化

            self.preloaded_data.append(results)

    def __getitem__(self, idx):
        # if self.preloaded_data:
        #     # 直接返回预加载的数据
        #     data = self.preloaded_data[idx]
        # 动态加载数据（仅在未预加载时使用）
        img_info_l = self.image_infos[idx][0]
        img_info_r = self.image_infos[idx][1]
        data_l = self.transform(img_info_l)
        data_r = self.transform(img_info_r)

        # # 动态应用随机增强（如 RandomHorizontalFlip）
        # if self.transform:
        #     data = self.transform(data)

        return data_l['img'], data_r['img'], data_l['label']

    def __len__(self):
        return len(self.image_infos)


if __name__ == '__main__':
    from dataset.transform import *
    import time

    transform = Compose([LoadImageFromFile(),
                         RandomFlip(),
                         RandomCrop((224, 224)),
                         ToTensor(),
                         Resize((224, 224)),
                         Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])
    dataset = A007Dataset(txt_file="train.txt",
                          root_dir="/mnt/mydisk/medical_seg/fwwb_a007/data/dataset",
                          transform=transform,
                          seed=42,
                          preload=False)
    dataloader = DataLoader(dataset,
                            batch_size=32,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True
                            )
    time_start = time.time()
    for images, labels in dataloader:
        time_cost = time.time() - time_start
        print(time_cost)
        time_start = time.time()
