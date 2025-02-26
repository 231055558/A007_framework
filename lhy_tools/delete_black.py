import cv2
import numpy as np
import os
from tqdm import tqdm  # 用于显示进度条

def crop_black_edges(image):
    """
    裁剪掉图像中所有纯黑色的部分
    :param image: 输入图像 (BGR 格式)
    :return: 裁剪后的图像
    """
    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 找到非黑色区域的边界
    rows = np.any(gray != 0, axis=1)  # 检查每一行是否有非黑色像素
    cols = np.any(gray != 0, axis=0)  # 检查每一列是否有非黑色像素
    # 获取非黑色区域的最小和最大边界
    rmin, rmax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, gray.shape[0])
    cmin, cmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, gray.shape[1])
    # 裁剪图像
    cropped_image = image[rmin:rmax + 1, cmin:cmax + 1]
    return cropped_image

def process_folder(input_folder, output_folder):
    """
    处理文件夹中的所有图像
    :param input_folder: 输入文件夹路径
    :param output_folder: 输出文件夹路径
    """
    # 获取文件夹中的所有图像文件
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.jpeg', '.bmp'))]
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 处理每张图像
    for file in tqdm(image_files, desc="Processing images"):
        # 加载图像
        image_path = os.path.join(input_folder, file)
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error loading image: {file}")
            continue

        # 裁剪黑色边缘
        cropped_image = crop_black_edges(image)
        # 保存裁剪后的图像
        output_path = os.path.join(output_folder, file)
        cv2.imwrite(output_path, cropped_image)

# 示例使用
input_folder = "/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/images_"  # 输入文件夹路径
output_folder = "/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/images"     # 输出文件夹路径
process_folder(input_folder, output_folder)
