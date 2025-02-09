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


data_root = '/mnt/mydisk/medical_seg/fwwb_a007/data/dataset'
transform_train = Compose([LoadImageFromFile(),
                           Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
                           RandomCrop((224, 224)),
                           Resize((224, 224)),
                           ToTensor()])

transform_test = Compose([LoadImageFromFile(),
                          Preprocess(mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)),
                          CenterCrop((224, 224)),
                          Resize((224, 224)),
                          ToTensor()])

model = ResNet(depth=50,
               num_classes=8)
ckp = "../../../checkpoints/resnet50.pth"

load_model_weights(model, ckp)

train_loader = DataLoader(A007Dataset(txt_file="train.txt",
                                      root_dir="/mnt/mydisk/medical_seg/fwwb_a007/data/dataset",
                                      transform=transform_train,
                                      seed=42,
                                      preload=False),
                          batch_size=32,
                          shuffle=True,
                          num_workers=16,
                          pin_memory=True
                          )
val_loader = DataLoader(A007Dataset(txt_file="train.txt",
                                      root_dir="/mnt/mydisk/medical_seg/fwwb_a007/data/dataset",
                                      transform=transform_test,
                                      seed=42,
                                      preload=False),
                          batch_size=32,
                          shuffle=True,
                          num_workers=16,
                          pin_memory=True
                          )
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
    save_path='best_model.pth',
    val=True
)

# val_model(
#     model=model,
#     val_loader=val_loader,
#     metric=metric,
#     device='cuda'
# )
