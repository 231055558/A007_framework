from torch.utils.data import DataLoader
from dataset.A007_txt_merge_model import A007Dataset
from dataset.transform import *
from dataset.transform.color_exchange import RandomColorTransfer
from loss.cross_entropy import CrossEntropyLoss
from metrics.a007_metric import A007_Metrics_Label
from models.load import load_model_weights
from networks.visiontransformer import VisionTransformer_Color_Merge
from optims.optimizer import Optimizer
from tools.predict import predict_model
from tools.train import train_color_merge_model
from tools.val import val_color_merge_model
from visualization.visualizer import Visualizer


class VisionTransformer_S_224_Bce_Adam_Lr1e_3_Bs32:
    def __init__(self):
        img_size = 224
        self.pretrain_ckp = "../../../checkpoints/vit-base.pth"

        self.model = VisionTransformer_Color_Merge(arch='base',
                                                   img_size=img_size,
                                                   patch_size=32,
                                                   num_classes=8,
                                                   drop_rate=0.1)
        self.data_root = '../../../data/data_merge'
        self.model_name = 'VisionTransformer'
        self.transform_train = Compose([LoadImageFromFile(),
                                        RandomColorTransfer(source_image_dir='../../../data/data_merge/images'),
                                        RandomFlip(),
                                        RandomCrop((224, 224)),
                                        ToTensor(),
                                        Resize((224, 224)),
                                        Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])
        self.transform_val = Compose([LoadImageFromFile(),
                                      CenterCrop((224, 224)),
                                      ToTensor(),
                                      Resize((224, 224)),
                                      Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375))])
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
        self.val_loader = DataLoader(A007Dataset(txt_file="val.txt",
                                                 root_dir=self.data_root,
                                                 transform=self.transform_val,
                                                 seed=42,
                                                 preload=False),
                                     batch_size=32,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True
                                     )
        self.loss_fn = CrossEntropyLoss(use_sigmoid=True)
        self.metric = A007_Metrics_Label(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
        self.optimizer = Optimizer(model_params=self.model.parameters(),
                                   optimizer='adam',
                                   lr=1e-3,
                                   weight_decay=1e-4
                                   )
        self.visualizer = Visualizer(experiment_name=self.model_name, metrics=self.metric)

    def train(self, epoch=100, val=True):
        load_model_weights(self.model, self.pretrain_ckp)
        train_color_merge_model(
            model=self.model,
            model_name=self.model_name,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            metric=self.metric,
            optimizer=self.optimizer,
            device='cuda',
            num_epochs=epoch,
            save_path='best_model.pth',
            val=val,
            visualizer=self.visualizer
        )

    def val(self):
        # trained_ckp = "./best_model.pth"
        # load_model_weights(self.model, trained_ckp)
        val_color_merge_model(
            model=self.model,
            model_name=self.model_name,
            val_loader=self.val_loader,
            metric=self.metric,
            device='cuda'
        )

    def predict_model(self):
        # trained_ckp = "./best_model.pth"
        # load_model_weights(self.model, trained_ckp)
        predict_model(
            model=self.model,
            test_loader=self.val_loader,
            metric=self.metric,
            model_name=self.model_name,
            device='cuda',
            output_folder="./output"
        )


if __name__ == '__main__':
    model = VisionTransformer_S_224_Bce_Adam_Lr1e_3_Bs32()
    model.train()
