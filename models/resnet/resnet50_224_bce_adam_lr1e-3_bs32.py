from models.load import load_model_weights
from networks.resnet import ResNet
from tools.train import train_model
from tools.val import val_model
from loss.cross_entropy import CrossEntropyLoss
from metrics.a007_metric import A007_Metrics
from optims.optimizer import Optimizer
from dataset.A007_txt import A007DataLoader, A007Dataset
from dataset.transform import *

data_root = '/mnt/mydisk/medical_seg/fwwb_a007/data/dataset'
transform_train = Compose([LoadImageFromFile(),
                         Preprocess(),
                         RandomCrop((224, 224)),
                         Resize((224, 224)),
                         ToTensor()])

model = ResNet(depth=50,
               num_classes=8)
ckp = "../../../checkpoints/resnet50.pth"

load_model_weights(model, ckp)



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
loss_fn = CrossEntropyLoss(use_sigmoid=True)
metric = A007_Metrics(thresholds=[0.1, 0.3, 0.5, 0.7, 0.9])
optimizer = Optimizer(model_params=model.parameters(),
                      optimizer='adam',
                      lr=1e-3,
                      weight_decay=1e-4
                      )


train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    loss_fn=loss_fn,
    metric=metric,
    optimizer=optimizer,
    device='cuda',
    num_epochs=100,
    save_path='best_model.pth'
)

# val_model(
#     model=model,
#     val_loader=val_loader,
#     metric=metric,
#     device='cuda'
# )
