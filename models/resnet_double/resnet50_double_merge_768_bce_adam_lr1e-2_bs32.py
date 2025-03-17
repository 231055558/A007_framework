from models.load import load_model_weights
from networks.resnet_to_linear import ResNet_To_Linear
from blocks.head import AttentionFC
from tools.predict import predict_model
from tools.train import train_double_merge_model
from tools.val import val_double_merge_model, val_output_merge_model
from loss.cross_entropy import CrossEntropyLoss
from metrics.a007_metric import A007_Metrics_Label
from optims.optimizer import Optimizer
from dataset.A007_txt_merge_model import A007Dataset
from dataset.transform import *
from torch.utils.data import DataLoader
from visualization.visualizer import Visualizer


class ResNet50_Double_Merge_768_Bce_Adam_Lr1e_2_Bs32:
    def __init__(self):
        self.data_root = '../../../data/data_merge'
        self.pretrain_ckp_model_1 = "../../../checkpoints/resnet50.pth"
        self.pretrain_ckp_model_2 = "../../../checkpoints/resnet50.pth"
        self.pretrain_ckp_head = None
        self.model_name = 'ResNet50_224_Bce_Adam_Lr1e_3_Bs32'
        self.transform_train = Compose([LoadImageFromFile(),
                                        Resize_Numpy((1080, 1080)),
                                        Random_Roi_Crop((768, 768)),
                                        # RandomColorTransfer(source_image_dir='../../../data/data_merge/images'),
                                        RandomFlip(),
                                        # RandomCrop((512, 512)),
                                        ToTensor(),
                                        # Resize((512, 512)),
                                        # Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
                                        AdaptiveNormalize()
                                        ])

        self.transform_val = Compose([LoadImageFromFile(),
                                      Resize_Numpy((1080, 1080)),
                                      Center_Roi_Crop((768, 768)),
                                      # CenterCrop((512, 512)),
                                      ToTensor(),
                                      # Resize((512, 512)),
                                      # Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
                                      AdaptiveNormalize()
                                      ])
        self.model_1 = ResNet_To_Linear(depth=50,
                            num_classes=8)
        self.model_2 = ResNet_To_Linear(depth=50,
                            num_classes=8)
        self.head = AttentionFC(in_features=4096, num_classes=8)

        self.train_loader = DataLoader(A007Dataset(txt_file="train.txt",
                                                   root_dir=self.data_root,
                                                   transform=self.transform_train,
                                                   seed=42,
                                                   preload=False),
                                       batch_size=16,
                                       shuffle=True,
                                       num_workers=4,
                                       pin_memory=True
                                       )
        self.val_loader = DataLoader(A007Dataset(txt_file="val.txt",
                                                 root_dir=self.data_root,
                                                 transform=self.transform_val,
                                                 seed=42,
                                                 preload=False),
                                     batch_size=16,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True
                                     )
        self.loss_fn = CrossEntropyLoss(use_sigmoid=True)
        self.metric = A007_Metrics_Label(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
        self.optimizer_1 = Optimizer(model_params=self.model_1.parameters(),
                                   optimizer='adam',
                                   lr=1e-3,
                                   weight_decay=1e-4
                                   )
        self.optimizer_2 = Optimizer(model_params=self.model_2.parameters(),
                                      optimizer='adam',
                                      lr=1e-3,
                                      weight_decay=1e-4
                                      )
        self.visualizer = Visualizer(experiment_name=self.model_name, metrics=self.metric)
        #self.pretrain_ckp = "./best_model.pth"

    def train(self, epoch=100, val=True):
        load_model_weights(self.model_1, self.pretrain_ckp_model_1)
        load_model_weights(self.model_2, self.pretrain_ckp_model_2)
        if self.pretrain_ckp_head is not None:
            load_model_weights(self.head, self.pretrain_ckp_head)

        train_double_merge_model(
            model_1=self.model_1,
            model_2=self.model_2,
            head=self.head,
            model_name=self.model_name,
            train_loader=self.train_loader,
            val_loader=self.val_loader,
            loss_fn=self.loss_fn,
            metric=self.metric,
            optimizer=self.optimizer_1,
            optimizer_2=self.optimizer_2,
            device='cuda',
            num_epochs=epoch,
            save_path='best_model.pth',
            val=val,
            visualizer=self.visualizer
        )

    def val(self):
        trained_ckp = "./best_model.pth"
        load_model_weights(self.model_1, trained_ckp)
        load_model_weights(self.model_2, trained_ckp)
        load_model_weights(self.head, trained_ckp)
        val_double_merge_model(
            model_1=self.model_1,
            model_2=self.model_2,
            head=self.head,
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
    model = ResNet50_Double_Merge_768_Bce_Adam_Lr1e_2_Bs32()
    model.val()
