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
import numpy as np
from typing import Callable, Optional
import torch



class ImageInfo:
    def __init__(self, img_path: str, label: list):
        self.data = {'image_path': img_path, 'label': label}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"ImageInfo\n {self.data}"

class A007Dataset:
    def __init__(self, txt_file:str, root_dir:str, transform: Optional[Callable] = None, seed: Optional[int] = 42 ):
        self.root_dir = root_dir
        self.transform = transform
        self.image_infos = list()

        if seed is not None:
            random.seed(seed)

        self._load_data(txt_file)

    def _load_data(self, txt_file:str):
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
    def __getitem__(self, idx):
        results = self.image_infos[idx]
        # image = cv2.imread(img_info['image_path'])
        # if image is None:
        #     raise ValueError(f"Failed to load image {img_info['image_path']}")

        if self.transform:
            results = self.transform(results)

        # return image, img_info['label']
        return results['img'], results['label']

    def __len__(self):
        return len(self.image_infos)

class A007DataLoader:
    def __init__(self, dataset:A007Dataset, batch_size:int, shuffle: bool=True):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))

    def __iter__(self):
        self.current_idx = 0
        if self.shuffle:
            random.shuffle(self.indices)
        return self

    def __next__(self):
        if self.current_idx >= len(self.indices):
            raise StopIteration

        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        batch = [self.dataset[idx] for idx in batch_indices]
        self.current_idx += self.batch_size

        images, labels = zip(*batch)
        images, labels = torch.stack(images), torch.stack(labels)
        return images, labels

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size


if __name__ == '__main__':
    from dataset.transform import *

    transform = Compose([LoadImageFromFile(),
                         Preprocess(),
                         RandomCrop((224, 224)),
                         Resize((224, 224)),
                         ToTensor()])
    dataset = A007Dataset(txt_file="train.txt",
                          root_dir="/mnt/mydisk/medical_seg/fwwb_a007/data/training_data",
                          transform=transform,
                          seed=42)
    dataloader = A007DataLoader(dataset, batch_size=4)
    for images, labels in dataloader:
        print(images.shape, labels)