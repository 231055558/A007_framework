from models.load import load_model_weights
from networks.resnet_color_merge import ResNet_Color_Merge
from tools.predict import predict_model
from tools.train import train_color_merge_model
from tools.val import val_model
from loss.cross_entropy import CrossEntropyLoss
from metrics.a007_metric import A007_Metrics
from optims.optimizer import Optimizer
from dataset.A007_txt_color_merge import A007Dataset
from dataset.transform import *
from torch.utils.data import DataLoader
from visualization.visualizer import Visualizer


class ResNet50_Color_Merge_224_Bce_Adam_Lr1e_3_Bs32:
    def __init__(self):
        self.data_root = '../../../data/dataset'
        self.model_name = 'ResNet50_224_Bce_Adam_Lr1e_3_Bs32'
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
        self.model = ResNet_Color_Merge(depth=50,
                                        num_classes=8)

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
        self.loss_fn = CrossEntropyLoss(use_sigmoid=True)
        self.metric = A007_Metrics(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
        self.optimizer = Optimizer(model_params=self.model.parameters(),
                                   optimizer='adam',
                                   lr=1e-3,
                                   weight_decay=1e-4
                                   )
        self.visualizer = Visualizer(experiment_name=self.model_name, metrics=self.metric)
        # self.pretrain_ckp = "../../../checkpoints/resnet50.pth"
        self.pretrain_ckp = "./best_model.pth"

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
        trained_ckp = "./best_model.pth"
        load_model_weights(self.model, trained_ckp)
        val_model(
            model=self.model,
            model_name=self.model_name,
            val_loader=self.val_loader,
            metric=self.metric,
            device='cuda'
        )

    def predict_model(self):
        trained_ckp = "./best_model.pth"
        load_model_weights(self.model, trained_ckp)
        predict_model(
            model=self.model,
            test_loader=self.val_loader,
            metric=self.metric,
            model_name=self.model_name,
            device='cuda',
            output_folder="./output"
        )


if __name__ == '__main__':
    model = ResNet50_Color_Merge_224_Bce_Adam_Lr1e_3_Bs32()
    model.val()
