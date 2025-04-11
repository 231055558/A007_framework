import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm  # 用于显示进度条

def crop_black_edges_pil(pil_image, threshold=5):
    """
    裁剪掉PIL图像中所有纯黑色的部分
    :param pil_image: 输入PIL图像
    :param threshold: 黑色像素阈值，默认5
    :return: 裁剪后的PIL图像
    """
    # 将PIL图像转换为numpy数组
    img_array = np.array(pil_image)
    # 如果是RGBA图像，只取RGB通道
    if img_array.shape[2] == 4:
        img_array = img_array[:, :, :3]
    
    # 转换为灰度图像
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    # 找到非黑色区域的边界
    rows = np.any(gray > threshold, axis=1)
    cols = np.any(gray > threshold, axis=0)
    
    # 获取非黑色区域的最小和最大边界
    rmin, rmax = np.where(rows)[0][[0, -1]] if np.any(rows) else (0, gray.shape[0])
    cmin, cmax = np.where(cols)[0][[0, -1]] if np.any(cols) else (0, gray.shape[1])
    
    # 裁剪图像
    cropped_array = img_array[rmin:rmax + 1, cmin:cmax + 1]
    
    # 转换回PIL图像
    return Image.fromarray(cropped_array)

if __name__ == '__main__':
    from PIL import Image

    # 加载PIL图像
    pil_img = Image.open("D:\\code\\A07\\dataset\\yanzhengji\\final\\1_left.jpg")
    # 去除黑边
    cropped_img = crop_black_edges_pil(pil_img)
    # 保存结果
    cropped_img.save("output.jpg")