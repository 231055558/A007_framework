from models.load import load_model_weights
from networks.resnet import ResNet
from tools.train import train_model
from tools.val import val_model
from loss.cross_entropy import CrossEntropyLoss
from metrics.a007_metric import A007_Metrics
from optims.optimizer import Optimizer
from dataset.A007_txt import A007Dataset
from dataset.transform import *
from torch.utils.data import DataLoader


def main(mode):
    data_root = '../../../data/dataset'
    transform_train = Compose([LoadImageFromFile(),
                               RandomFlip(),
                               RandomCrop((1080, 1080)),
                               ToTensor(),
                               Resize((256, 256)),
                               Preprocess(mean=(26.79446452, 48.51940625, 76.53684116),
                                          std=(27.8611716, 47.70409773, 72.05617777))])

    transform_val = Compose([LoadImageFromFile(),
                             CenterCrop((1080, 1080)),
                             ToTensor(),
                             Resize((256, 256)),
                             Preprocess(mean=(26.79446452, 48.51940625, 76.53684116),
                                        std=(27.8611716, 47.70409773, 72.05617777))])

    model = ResNet(depth=50,
                   num_classes=8)

    train_loader = DataLoader(A007Dataset(txt_file="train.txt",
                                          root_dir=data_root,
                                          transform=transform_train,
                                          seed=42,
                                          preload=False),
                              batch_size=32,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True
                              )
    val_loader = DataLoader(A007Dataset(txt_file="val.txt",
                                        root_dir=data_root,
                                        transform=transform_val,
                                        seed=42,
                                        preload=False),
                            batch_size=32,
                            shuffle=True,
                            num_workers=4,
                            pin_memory=True
                            )
    loss_fn = CrossEntropyLoss(use_sigmoid=True)
    metric = A007_Metrics(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
    optimizer = Optimizer(model_params=model.parameters(),
                          optimizer='adam',
                          lr=1e-3,
                          weight_decay=1e-4
                          )
    if mode == "train":
        pretrain_ckp = "../../../checkpoints/resnet50.pth"
        load_model_weights(model, pretrain_ckp)
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=loss_fn,
            metric=metric,
            optimizer=optimizer,
            device='cuda',
            num_epochs=100,
            save_path='best_model.pth',
            val=True
        )
    if mode == "val":
        trained_ckp = "../../../checkpoints/resnet50_224_bce_adam_lr1e-3_bs32_checkpoint/best_model.pth"
        load_model_weights(model, trained_ckp)
        val_model(
            model=model,
            val_loader=val_loader,
            metric=metric,
            device='cuda'
        )


if __name__ == '__main__':
    mode = "train"
    main(mode)
