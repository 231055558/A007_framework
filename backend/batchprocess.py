from models.load import load_model_weights
from networks.deeplabv3plus import DeepLabV3PlusClassifierAttentionHeadLinearMerge
from tools.predict import detect_model
from dataset.A007_detect import A007Dataset
from dataset.transform import *
from torch.utils.data import DataLoader


class DeepLabV3Plus_Batch_Detect:
    def __init__(self,
                 data_root='static/uploads/114514',
                 txt_file="selected.txt",
                 threshold=0.5,
                 pretrain_ckp=None):
        self.data_root = data_root
        if pretrain_ckp is None:
            import os
            self.pretrain_ckp = os.path.join(os.path.dirname(__file__), "best_model.pth")
        else:
            self.pretrain_ckp = pretrain_ckp
        self.txt_file = txt_file

        self.transform_val = Compose([LoadImageFromFile(),
                                      #RemoveBlackBorder(),
                                      Resize_Numpy((1080, 1080)),
                                      Center_Roi_Crop((768, 768)),
                                      DetectToTensor(),
                                      AdaptiveNormalize()
                                      ])
        self.model = DeepLabV3PlusClassifierAttentionHeadLinearMerge(num_classes=8)

        self.pre_loader = DataLoader(A007Dataset(txt_file=self.txt_file,
                                                 root_dir=self.data_root,
                                                 transform=self.transform_val),
                                     batch_size=3,
                                     shuffle=False,
                                     num_workers=4,
                                     pin_memory=True
                                     )
        self.threshold = threshold


    async def predict_model(self,progress_callback=None):
        trained_ckp = self.pretrain_ckp
        load_model_weights(self.model, trained_ckp)
        result = await detect_model(
            model=self.model,
            test_loader=self.pre_loader,
            threshold=self.threshold,
            progress_callback=progress_callback,
            device='cuda',
            output_folder=self.data_root + "/output"
        )
        return result


if __name__ == '__main__':
    #测试读取static/uplodas/114514/selected.txt
    import os
    import asyncio

    async def main():
        base_dir = "D://Users//drlou//Desktop//"
        data_dir = os.path.join(base_dir, 'selected')
        detector = DeepLabV3Plus_Batch_Detect(data_root=data_dir,
                                            txt_file="selected.txt",
                                            threshold=0.5)
        result = await detector.predict_model()
        print(result)
    asyncio.run(main())
