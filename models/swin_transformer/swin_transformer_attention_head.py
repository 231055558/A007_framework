from dataset.transform import *
from torch.utils.data import DataLoader
from visualization.visualizer import Visualizer
from networks.swin_transformer import SwinTransformer
from blocks.head import MultiHeadDiseaseClassifier
from dataset.A007_txt_merge_model import A007Dataset
from metrics.a007_metric import A007_Metrics_Label
from loss.cross_entropy import CrossEntropyLoss
from optims.optimizer import Optimizer
from tools.train import train_output_merge_model
from tools.val import val_output_merge_model
from tools.predict import predict_model
from models.load import load_model_weights


class SwinTransformer_Attention_Head:
    def __init__(self):
        self.data_root = '../../../data/data_merge'
        self.pretrain_ckp = "./best_model.pth"
        
        self.model_name = 'SwinTransformer_Attention_Head_768_BCE_AdamW_Lr1e-4_Bs16'
        
        # 定义训练数据转换
        self.transform_train = Compose([
            LoadImageFromFile(),
            Resize_Numpy((1080, 1080)),
            Random_Roi_Crop((768, 768)),
            RandomFlip(),
            ToTensor(),
            AdaptiveNormalize()
            # 移除了Resize，直接使用768x768的尺寸
        ])

        # 定义验证数据转换
        self.transform_val = Compose([
            LoadImageFromFile(),
            Resize_Numpy((1080, 1080)),
            Center_Roi_Crop((768, 768)),
            ToTensor(),
            AdaptiveNormalize()
            # 移除了Resize，直接使用768x768的尺寸
        ])

        # 初始化模型 - 调整为更大的模型配置
        self.model = SwinTransformer(
            img_size=768,  # 增大输入尺寸
            patch_size=4,
            in_channels=3,
            num_classes=8,
            embed_dim=128,  # 增大基础特征维度
            depths=[2, 2, 18, 2],  # 加深网络
            num_heads=[4, 8, 16, 32],  # 增加注意力头数
            window_size=12,  # 增大窗口大小
            mlp_ratio=4.,
            qkv_bias=True,
            drop_rate=0.1,
            attn_drop_rate=0.1,
            drop_path_rate=0.2  # 略微增加drop path rate以防止过拟合
        )

        # 数据加载器 - 由于模型变大，减小batch size
        self.train_loader = DataLoader(
            A007Dataset(
                txt_file="train.txt",
                root_dir=self.data_root,
                transform=self.transform_train,
                seed=42,
                preload=False
            ),
            batch_size=16,  # 减小batch size以适应更大的模型
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
            batch_size=16,  # 减小batch size
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # 损失函数、评估指标和优化器
        self.loss_fn = CrossEntropyLoss(use_sigmoid=True)
        self.metric = A007_Metrics_Label(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
        self.optimizer = Optimizer(
            model_params=self.model.parameters(),
            optimizer='adamw',
            lr=5e-5,  # 由于模型变大，稍微降低学习率
            weight_decay=0.05
        )

        # 可视化工具
        self.visualizer = Visualizer(
            experiment_name=self.model_name,
            metrics=self.metric
        )

    def train(self, epoch=100, val=True):
        """训练模型"""
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
        load_model_weights(self.model, self.pretrain_ckp)
        val_output_merge_model(
            model=self.model,
            model_name=self.model_name,
            val_loader=self.val_loader,
            metric=self.metric,
            device='cuda'
        )

    def predict_model(self):
        """预测"""
        load_model_weights(self.model, self.pretrain_ckp)
        predict_model(
            model=self.model,
            test_loader=self.val_loader,
            metric=self.metric,
            model_name=self.model_name,
            device='cuda',
            output_folder="./output"
        )


if __name__ == '__main__':
    model = SwinTransformer_Attention_Head()
    model.train(epoch=100)  # 训练模型
    # model.val()  # 验证模型
    # model.predict_model()  # 预测 