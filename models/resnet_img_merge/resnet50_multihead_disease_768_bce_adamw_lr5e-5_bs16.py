from models.load import load_model_weights
from networks.resnet import ResNet
from tools.predict import predict_model
from tools.train import train_output_merge_model
from tools.val import val_output_merge_model
from loss.diseaseclassificationloss import DiseaseClassificationLoss
from metrics.a007_metric import A007_Metrics_Label
from optims.optimizer import Optimizer
from dataset.A007_txt_merge_model import A007Dataset
from dataset.transform import *
from torch.utils.data import DataLoader
from visualization.visualizer import Visualizer
from blocks.head import MultiHeadDiseaseClassifier
import torch


class ResNet50_MultiHead_Disease_768_Bce_AdamW:
    def __init__(self):
        self.data_root = '../../../data/data_merge'
        self.pretrain_ckp = "../../../checkpoints/resnet50.pth"

        self.model_name = 'ResNet50_MultiHead_Disease_768_Bce_AdamW_Lr5e-5_Bs16'
        
        # 定义训练数据转换
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

        # 初始化模型
        backbone = ResNet(
            depth=50,
            num_classes=8
        )
        # 替换原有分类头为新的多头疾病分类器
        backbone.head = MultiHeadDiseaseClassifier(
            in_features=2048,  # ResNet50的特征维度
            num_classes=8
        )
        self.model = backbone

        # 数据加载器
        self.train_loader = DataLoader(
            A007Dataset(
                txt_file="train.txt",
                root_dir=self.data_root,
                transform=self.transform_train,
                seed=42,
                preload=False
            ),
            batch_size=16,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

        self.val_loader = DataLoader(
            A007Dataset(
                txt_file="val.txt",
                root_dir=self.data_root,
                transform=self.transform_val,
                seed=42,
                preload=False
            ),
            batch_size=16,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 计算类别权重
        pos_counts = torch.tensor([1136, 161, 211, 525, 215, 77, 174, 964])
        neg_counts = torch.tensor([1749, 2724, 2674, 2360, 2670, 2808, 2711, 1921])
        pos_weights = torch.sqrt(neg_counts / pos_counts)

        # 损失函数、评估指标和优化器
        self.loss_fn = DiseaseClassificationLoss(pos_weights=pos_weights)
        self.metric = A007_Metrics_Label(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
        self.optimizer = Optimizer(
            model_params=self.model.parameters(),
            optimizer='adamw',
            lr=5e-5,
            weight_decay=0.05
        )

        # 可视化工具
        self.visualizer = Visualizer(
            experiment_name=self.model_name,
            metrics=self.metric
        )

    def train(self, epoch=100, val=True):
        """训练模型"""
        load_model_weights(self.model, self.pretrain_ckp)
        train_output_merge_model(
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
        """验证模型"""
        trained_ckp = "./best_model.pth"
        load_model_weights(self.model, trained_ckp)
        val_output_merge_model(
            model=self.model,
            model_name=self.model_name,
            val_loader=self.val_loader,
            metric=self.metric,
            device='cuda'
        )

    def predict_model(self):
        """预测"""
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
    model = ResNet50_MultiHead_Disease_768_Bce_AdamW()
    model.train(epoch=100)  # 训练模型
    # model.val()  # 验证模型
    # model.predict_model()  # 预测 