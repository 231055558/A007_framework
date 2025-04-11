from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from torchvision.transforms import Compose
from dataset.transform import LoadImageFromFile, Resize_Numpy, Center_Roi_Crop, ToTensor, AdaptiveNormalize
from networks.deeplabv3plus import DeepLabV3PlusClassifierAttentionHeadLinearMerge
import torch.nn as nn

class DeepLabCAMWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model
        self.gradients = []
        self.activations = []
        self.categories = ['N', 'A', 'C', 'D', 'G', 'H', 'M', 'O']
        
        # 注册钩子到classifier的最后一个卷积层（示例层，需根据实际结构调整）
        target_layer = self.model.model.classifier[-4]  # 假设最后一个卷积是classifier[4]
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients.append(grad_output[0])

    def forward(self, x1, x2):
        self.activations = []
        self.gradients = []
        
        # 处理第一个图像
        features1 = self.model.model.backbone(x1)['out']
        out1 = self.model.model.classifier(features1)
        pooled1 = self.model.global_pool(out1)
        pooled1 = pooled1.view(pooled1.size(0), -1)
        
        # 处理第二个图像
        features2 = self.model.model.backbone(x2)['out']
        out2 = self.model.model.classifier(features2)
        pooled2 = self.model.global_pool(out2)
        pooled2 = pooled2.view(pooled2.size(0), -1)
        
        # 合并特征（假设原始模型使用concat合并）
        combined = torch.cat([pooled1, pooled2], dim=1)
        output = self.model.fc(combined)
        
        return output, out1, out2


def generate_cam_with_auto_target(model_wrapper, x1, x2):
    """自动获取预测类别并生成热力图"""
    # 前向传播获取预测类别
    with torch.no_grad():
        output, _, _ = model_wrapper(x1.unsqueeze(0), x2.unsqueeze(0))
        threshold = 0.5
        binary_outputs = (output >= threshold).int()
        binary_outputs = binary_outputs.cpu().numpy().squeeze(0)
        #binary_outputs每一位代表一个类别，分别是N A C D G H M O，我希望返回一个str，内容是是1的类别
        # 获取激活的类别索引
        active_indices = np.where(binary_outputs == 1)[0]
        # 获取类别名称
        detected_categories = [model_wrapper.categories[i] for i in active_indices]
        pred_class = ''.join(detected_categories)
    cam1_list = []
    cam2_list = []
    for i in active_indices:
        cam1, cam2 = generate_cam(model_wrapper, x1.unsqueeze(0), x2.unsqueeze(0), i)
        cam1_list.append(cam1)
        cam2_list.append(cam2)

    return cam1_list, cam2_list, pred_class

def generate_cam(model_wrapper, x1, x2, target_class):
    # 前向传播
    model_wrapper.zero_grad()
    output, act1, act2 = model_wrapper(x1, x2) 
    # 反向传播
    one_hot = torch.zeros_like(output)
    one_hot[0, target_class] = 1
    output.backward(gradient=one_hot, retain_graph=True)
    
    # 获取梯度
    gradients = model_wrapper.gradients
    
    # 处理第一个图像
    pooled_gradients1 = torch.mean(gradients[0], dim=[2, 3], keepdim=True)
    cam1 = torch.sum(act1 * pooled_gradients1, dim=1, keepdim=True)
    cam1 = F.relu(cam1)
    cam1 = F.interpolate(cam1, x1.shape[2:], mode='bilinear', align_corners=False)
    cam1 = (cam1 - cam1.min()) / (cam1.max() - cam1.min())
    
    # 处理第二个图像
    pooled_gradients2 = torch.mean(gradients[1], dim=[2, 3], keepdim=True)
    cam2 = torch.sum(act2 * pooled_gradients2, dim=1, keepdim=True)
    cam2 = F.relu(cam2)
    cam2 = F.interpolate(cam2, x2.shape[2:], mode='bilinear', align_corners=False)
    cam2 = (cam2 - cam2.min()) / (cam2.max() - cam2.min())
    
    return cam1.detach().squeeze().cpu().numpy(), cam2.detach().squeeze().cpu().numpy()

# 初始化包装器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DeepLabV3PlusClassifierAttentionHeadLinearMerge(num_classes=8)
#print(model.model.classifier)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
cam_model = DeepLabCAMWrapper(model).to(device)
cam_model.eval()


# 生成热力图
root = 'D:\\Users\\drlou\\Desktop\\data\\images\\'
'''
left_image = Image.open('D:\\Users\\drlou\\Desktop\\data\\images\\N\\1_left.jpg')
right_image = Image.open('D:\\Users\\drlou\\Desktop\\data\\images\\N\\1_right.jpg')
'''
left_image = Image.open('D:\\Users\\drlou\\Desktop\\data\\images\\D\\7_left.jpg')
right_image = Image.open('D:\\Users\\drlou\\Desktop\\data\\images\\D\\7_right.jpg')

transform = Compose([Resize_Numpy((1080, 1080)),
                                      Center_Roi_Crop((768, 768)),
                                      ToTensor(),
                                      AdaptiveNormalize()
                                      ])
img_l = transform({'img': np.array(left_image), 'image_path': 'left', 'label': 9})
img_r = transform({'img': np.array(right_image), 'image_path': 'right', 'label': 9})
img_l = img_l['img'].to(device)
img_r = img_r['img'].to(device)

cam1_list, cam2_list, pred_class= generate_cam_with_auto_target(cam_model, img_l, img_r)
print(pred_class)
# 可视化
def show_cam(image, cam, row, col):
    plt.subplot(len(cam1_list), 2, row*2 + col)
    plt.imshow(image.permute(1,2,0).cpu().numpy())
    plt.imshow(cam, alpha=0.5, cmap='jet')
    plt.axis('off')
    #plt.show()
plt.figure(figsize = (16,8*len(cam1_list)), dpi=96)
for idx, (cam1, cam2) in enumerate(zip(cam1_list, cam2_list)):
    show_cam(img_l, cam1, idx, 1)
    show_cam(img_r, cam2, idx, 2)
plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
plt.savefig('static/test.jpg')