import os
import shutil
import random

# 配置参数
dataset_txt = "../../data/data_label.txt"  # 你的数据集文件
image_folder = "../../data/images"   # 存放图片的文件夹
output_folder = "../../data/output"  # 结果保存的文件夹
train_ratio = 0.8         # 训练集比例

# 创建 train 和 val 目录
train_folder = os.path.join(output_folder, "train")
val_folder = os.path.join(output_folder, "val")
os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)

# 读取数据
with open(dataset_txt, "r") as f:
    lines = f.readlines()

# 数据预处理
data = [line.strip().split() for line in lines]  # [['0_left.jpg', '00010000'], ...]
random.shuffle(data)  # 随机打乱数据

# 训练集 & 验证集划分
split_idx = int(len(data) * train_ratio)
train_data = data[:split_idx]
val_data = data[split_idx:]

# 复制图片到对应的文件夹
def move_images(data_list, target_folder):
    for img_name, label in data_list:
        img_path = os.path.join(image_folder, img_name)
        if os.path.exists(img_path):
            shutil.copy(img_path, os.path.join(target_folder, img_name))

move_images(train_data, train_folder)
move_images(val_data, val_folder)

# 生成 train.txt 和 val.txt
def save_txt(data_list, filename):
    with open(filename, "w") as f:
        for img_name, label in data_list:
            f.write(f"{img_name} {label}\n")

save_txt(train_data, os.path.join(output_folder, "train.txt"))
save_txt(val_data, os.path.join(output_folder, "val.txt"))

print("数据集划分完成！")