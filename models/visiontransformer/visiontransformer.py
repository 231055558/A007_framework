from torch.utils.data import DataLoader

from dataset.A007_txt import A007Dataset
from dataset.transform import *


class VisionTransformer:
    def __init__(self):
        self.data_root = '../../../data/dataset'
        self.model_name = 'VisionTransformer'
        self.transform_train = Compose([LoadImageFromFile(),
                                        RandomFlip(),
                                        RandomCrop((1080, 1080)),
                                        ToTensor(),
                                        Resize((256, 256)),
                                        Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])
        self.transform_val = Compose([LoadImageFromFile(),
                                      CenterCrop((1080, 1080)),
                                      ToTensor(),
                                      Resize((256, 256)),
                                      Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])
        self.model =
        self.train_loader = DataLoader(A007Dataset(txt_file="train.txt",
                                                   root_dir=self.data_root,
                                                   transform=self.transform_train,
                                                   seed=42,
                                                   preload=False),
                                       batch_size=32,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True
                                       )
        self.val_loader = DataLoader(A007Dataset(txt_file="train.txt",
                                                 root_dir=self.data_root,
                                                 transform=self.transform_val,
                                                 seed=42,
                                                 preload=False),
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True
                                     )
