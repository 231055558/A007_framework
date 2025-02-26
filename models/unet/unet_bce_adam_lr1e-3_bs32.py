from models.load import load_model_weights
from blocks.unet import UNetEncoder
from tools.train import train_model
from tools.val import val_model
from loss.cross_entropy import CrossEntropyLoss
from metrics.a007_metric import A007_Metrics
from optims.optimizer import Optimizer
from dataset.loadertest import A007Dataset, DataLoader as A007DataLoader
#from dataset.A007_txt import A007DataLoader, A007Dataset
from dataset.transform import *

if __name__ == "__main__":
    # 数据集路径
    data_root = 'D:\\code\\A07\\dataset'

    # 训练时的数据预处理
    transform_train = Compose([LoadImageFromFile(),
                               Preprocess(),
                               RandomCrop((224, 224)),
                               Resize((224, 224)),
                               ToTensor()])

    # 加载UNet模型 (去掉解码器部分，保留编码器并添加分类头)
    model = UNetEncoder(in_channels=3,  # 假设输入是RGB图像
                        num_classes=8,  # 输出8个类别
                        num_filters=64)  # 默认的卷积核数目

    # 你可以选择加载预训练的权重
    ckp = "D:\\code\\A07\\model\\unet_encoder.pth"  # 假设UNet的权重在这里
    #load_model_weights(model, ckp)  # 加载权重

    # 数据加载器
    train_loader = A007DataLoader(dataset=A007Dataset(txt_file='train.txt',
                                                      root_dir=data_root,
                                                      transform=transform_train,
                                                      seed=42),
                                  batch_size=32,
                                  num_workers=4)
    val_loader = A007DataLoader(dataset=A007Dataset(txt_file='val.txt',
                                                    root_dir=data_root,
                                                    transform=transform_train,
                                                    seed=42),
                                batch_size=32,
                                num_workers=4)

    # 损失函数
    loss_fn = CrossEntropyLoss(use_sigmoid=True)  # 使用sigmoid，因为是像素级分类任务

    # 评价指标
    metric = A007_Metrics(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])

    # 优化器
    optimizer = Optimizer(model_params=model.parameters(),
                          optimizer='adam',
                          lr=1e-3,
                          weight_decay=1e-4)

    # 训练模型
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        metric=metric,
        optimizer=optimizer,
        device='cuda',
        num_epochs=5,
        save_path='best_model.pth'
    )

    # 验证模型
    val_model(
        model=model,
        val_loader=val_loader,
        metric=metric,
        device='cuda'
    )
