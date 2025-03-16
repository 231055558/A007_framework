import os
import random
import torch
import numpy as np
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 假设 AdaptiveNormalize 类已经定义
class AdaptiveNormalize:
    def __init__(self, eps=1e-7):
        self.eps = eps

    def transform(self, results):
        img = results['img']  # 输入应为 Tensor，形状 (C, H, W)

        # 计算每个通道的均值和标准差
        channel_means = torch.mean(img, dim=(1, 2))  # shape (C,)
        channel_stds = torch.std(img, dim=(1, 2))  # shape (C,)

        # 防除零处理：标准差为0时设为1
        channel_stds = torch.where(
            channel_stds < self.eps,
            torch.ones_like(channel_stds),
            channel_stds
        )

        # 自适应归一化
        normalized_img = (img - channel_means[:, None, None]) / channel_stds[:, None, None]

        results['img'] = normalized_img
        return results


# 定义函数：加载图片并将其转换为 Tensor
def load_image_to_tensor(image_path):
    image = Image.open(image_path).convert('RGB')  # 确保图片是 RGB 格式
    transform = transforms.ToTensor()  # 将 PIL 图片转换为 Tensor
    return transform(image)


# 定义函数：从文件夹中随机选取 16 张图片
def get_random_images(folder_path, num_images=16):
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    return [os.path.join(folder_path, f) for f in selected_files]


# 定义函数：绘制对比图表
def plot_comparison(original_images, normalized_images):
    plt.figure(figsize=(20, 10))
    for i in range(len(original_images)):
        # 显示原始图片
        plt.subplot(4, 8, 2 * i + 1)
        plt.imshow(original_images[i].permute(1, 2, 0))  # Tensor 转换为 (H, W, C) 格式
        plt.title('Original')
        plt.axis('off')

        # 显示归一化后的图片
        plt.subplot(4, 8, 2 * i + 2)
        plt.imshow(normalized_images[i].permute(1, 2, 0))  # Tensor 转换为 (H, W, C) 格式
        plt.title('Normalized')
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# 主函数
def main(folder_path):
    # 随机选取 16 张图片
    image_paths = get_random_images(folder_path)

    # 加载图片并转换为 Tensor
    original_images = [load_image_to_tensor(path) for path in image_paths]

    # 初始化 AdaptiveNormalize
    normalizer = AdaptiveNormalize()

    # 对每张图片进行归一化处理
    normalized_images = []
    for img in original_images:
        results = {'img': img}
        normalized_results = normalizer.transform(results)
        normalized_images.append(normalized_results['img'])

    # 绘制对比图表
    plot_comparison(original_images, normalized_images)


# 指定文件夹路径并运行
folder_path = '../../../data/data_merge/images'  # 替换为你的图片文件夹路径
main(folder_path)
