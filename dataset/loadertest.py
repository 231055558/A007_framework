import os
import random
from typing import Callable, Optional
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class ImageInfo:
    def __init__(self, img_path: str, label: list):
        self.data = {'image_path': img_path, 'label': label}

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        self.data[key] = value

    def __repr__(self):
        return f"ImageInfo\n {self.data}"

class A007Dataset(Dataset):
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
                image_path = os.path.join(self.root_dir, 'Training', parts[0])
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

def collate_fn(batch):
    images, labels = zip(*batch)
    return torch.stack(images), torch.stack(labels)

if __name__ == '__main__':
    from dataset.transform import *
    import time

    transform = Compose([LoadImageFromFile(),
                         Preprocess(),
                         RandomCrop((224, 224)),
                         Resize((224, 224)),
                         ToTensor()])

    dataset = A007Dataset(txt_file="train.txt",
                          root_dir="D:\\code\\A07\\dataset",
                          transform=transform,
                          seed=42)

    # 使用 DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4, collate_fn=collate_fn)

    # 测试加载时间
    time_start = time.time()
    for images, labels in dataloader:
        time_cost = time.time() - time_start
        print(time_cost)
        print(time_cost)
        time_start = time.time()