import os
import random
import numpy as np
from PIL import Image
from tqdm import tqdm

def read_data_label(file_path):
    """
    读取 data_label.txt 文件，返回图片路径和标签的列表。
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()
    data = [line.strip().split() for line in lines]
    return data

def calculate_mean_std(image_paths):
    """
    计算给定图片列表的均值和标准差。
    """
    # 初始化累加器
    mean = np.zeros(3)
    std = np.zeros(3)
    total_pixels = 0

    # 遍历图片
    for img_path in tqdm(image_paths, desc="Calculating mean and std"):
        img = Image.open(img_path).convert('RGB')
        img = np.array(img)
        mean += img.mean(axis=(0, 1))  # 计算每通道的均值
        std += img.std(axis=(0, 1))    # 计算每通道的标准差
        total_pixels += img.shape[0] * img.shape[1]

    # 计算平均值
    mean /= len(image_paths)
    std /= len(image_paths)
    return mean, std

def check_deviation(image_path, mean, std, threshold=2):
    """
    检查单张图片的均值和标准差是否严重偏离给定值。
    threshold: 偏离的阈值，默认为 2 倍标准差。
    """
    img = Image.open(image_path).convert('RGB')
    img = np.array(img) / 255.0  # 归一化到 [0, 1]
    img_mean = img.mean(axis=(0, 1))
    img_std = img.std(axis=(0, 1))

    # 判断是否偏离
    mean_deviation = np.abs(img_mean - mean) > threshold * std
    std_deviation = np.abs(img_std - std) > threshold * std
    return np.any(mean_deviation) or np.any(std_deviation)

def main(data_label_path, image_dir):
    """
    主函数，执行数据预处理分析。
    """
    # 读取数据
    data = read_data_label(data_label_path)
    image_paths = [os.path.join(image_dir, item[0]) for item in data]

    # 随机选择1500张图片
    random.seed(42)  # 固定随机种子以确保可重复性
    selected_images = random.sample(image_paths, 1500)

    # 计算随机选择图片的均值和标准差
    mean, std = calculate_mean_std(selected_images)
    print(f"Mean of selected images: {mean}")
    print(f"Std of selected images: {std}")

    # 统计所有图片中严重偏离的数量
    severe_deviation_count = 0
    for img_path in tqdm(image_paths, desc="Checking deviations"):
        if check_deviation(img_path, mean, std):
            severe_deviation_count += 1

    # 输出结果
    total_images = len(image_paths)
    print(f"Total images: {total_images}")
    print(f"Images with severe deviation: {severe_deviation_count}")
    print(f"Percentage of severe deviation: {severe_deviation_count / total_images * 100:.2f}%")

if __name__ == "__main__":
    # 文件路径
    data_label_path = "/mnt/mydisk/medical_seg/fwwb_a007/data/data_label.txt"  # 替换为你的文件路径
    image_dir = "/mnt/mydisk/medical_seg/fwwb_a007/data/images"  # 图片所在的目录

    # 运行主函数
    main(data_label_path, image_dir)
