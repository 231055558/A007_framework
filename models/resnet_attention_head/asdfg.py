from models.load import load_model_weights
from networks.resnet import ResNet
from networks.resnet_attention_head import ResNetAttentionHead
from tools.train import train_model
from tools.val import val_model
from loss.cross_entropy import CrossEntropyLoss
from metrics.a007_metric import A007_Metrics
from optims.optimizer import Optimizer
from dataset.A007_txt import A007Dataset
from dataset.transform import *
from torch.utils.data import DataLoader


def main(mode):


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
