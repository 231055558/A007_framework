from networks.resnet import ResNet
from tools.train import train_model
from loss.cross_entropy import CrossEntropyLoss
from optims.optimizer import Optimizer
from dataset.A007_txt import A007DataLoader, A007Dataset
from dataset.transform import *

data_root = '/mnt/mydisk/medical_seg/fwwb_a007/data/training_data'
transform_train = Compose([LoadImageFromFile(),
                         Preprocess(),
                         RandomCrop((224, 224)),
                         Resize((224, 224)),
                         ToTensor()])

model = ResNet(depth=50,
               num_classes=8)

train_loader = A007DataLoader(dataset=A007Dataset(txt_file='train.txt',
                                                  root_dir=data_root,
                                                  transform=transform_train,
                                                  seed=42),
                              batch_size=32)
loss_fn = CrossEntropyLoss(use_sigmoid=True)
optimizer = Optimizer(model_params=model.parameters(),
                      optimizer='adam',
                      lr=1e-3,
                      weight_decay=1e-4
                      )


train_model(
    model=model,
    train_loader=train_loader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    device='cuda',
    num_epochs=100,
    save_path='best_model.pth'
)
