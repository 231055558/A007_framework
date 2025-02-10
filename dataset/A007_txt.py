'''
可以通过txt文本读取A007单图像数据集的基本信息
要求把数据整理成如下格式
root_path
train.txt
val.txt
train(文件夹，存放图片)
val(文件夹，存放图片)

其中train.txt/val.txt格式如下:
0_left.jpg 00010000
0_right.jpg 00010000
1_left.jpg 10000000
1_right.jpg 10000000
2_left.jpg 01000001
2_right.jpg 01000001
3_left.jpg 00000001
3_right.jpg 00000001

'''

import os
import random
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
from typing import Callable, Optional
import torch
from torch.utils.data import DataLoader


class ImageInfo:
    def __init__(self, img_path: str, label: list):
        self.data = {'image_path': img_path, 'label': label}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"ImageInfo\n {self.data}"


# class A007Dataset:
#     def __init__(self, txt_file: str, root_dir: str, transform: Optional[Callable] = None, seed: Optional[int] = 42):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.image_infos = list()
#
#         if seed is not None:
#             random.seed(seed)
#
#         self._load_data(txt_file)
#
#     def _load_data(self, txt_file: str):
#         txt_path = os.path.join(self.root_dir, txt_file)
#         if not os.path.exists(txt_path):
#             raise FileNotFoundError(f"File {txt_path} not exists!")
#
#         with open(txt_path, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 if len(parts) != 2:
#                     continue
#                 image_path = os.path.join(self.root_dir, 'train' if 'train' in txt_file else 'val', parts[0])
#                 label = [int(x) for x in parts[1]]
#                 self.image_infos.append(ImageInfo(image_path, label))
#
#     def __getitem__(self, idx):
#         results = self.image_infos[idx]
#         # image = cv2.imread(img_info['image_path'])
#         # if image is None:
#         #     raise ValueError(f"Failed to load image {img_info['image_path']}")
#
#         if self.transform:
#             results = self.transform(results)
#
#         # return image, img_info['label']
#         return results['img'], results['label']
#
#     def __len__(self):
#         return len(self.image_infos)
#
#
# class A007DataLoader:
#     def __init__(self, dataset: A007Dataset, batch_size: int, shuffle: bool = True, num_workers: int = 4):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = list(range(len(dataset)))
#         self.num_workers = num_workers
#         self.current_idx = 0
#
#     def __iter__(self):
#         self.current_idx = 0
#         if self.shuffle:
#             random.shuffle(self.indices)
#         return self
#
#     def fetch_data(self, idx):
#         return self.dataset[idx]
#
#     def __next__(self):
#         if self.current_idx >= len(self.indices):
#             raise StopIteration
#
#         batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
#
#         # 使用多线程加载数据
#         with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
#             batch = list(executor.map(self.fetch_data, batch_indices))
#
#         self.current_idx += self.batch_size
#
#         # 将 batch 拆分成 images 和 labels
#         images, labels = zip(*batch)
#         return torch.stack(images), torch.stack(labels)
#
#     def __len__(self):
#         return (len(self.dataset) + self.batch_size - 1) // self.batch_size


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
                if len(parts) != 2:
                    continue
                image_path = os.path.join(self.root_dir, 'train' if 'train' in txt_file else 'val', parts[0])
                label = [int(x) for x in parts[1]]
                self.image_infos.append(ImageInfo(image_path, label))

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
        if self.preloaded_data:
            # 直接返回预加载的数据
            data = self.preloaded_data[idx]
        else:
            # 动态加载数据（仅在未预加载时使用）
            img_info = self.image_infos[idx]
            data = self.transform(img_info)

        # # 动态应用随机增强（如 RandomHorizontalFlip）
        # if self.transform:
        #     data = self.transform(data)

        return data['img'], data['label']

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
