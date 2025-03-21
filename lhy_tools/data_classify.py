import os
import shutil

# 定义标签映射
label_map = {
    0: 'N',
    1: 'A',
    2: 'C',
    3: 'D',
    4: 'G',
    5: 'H',
    6: 'M',
    7: 'O'
}

# 输入文件路径
input_txt = '/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/data_merge_label.txt'  # 你的 txt 文件路径
source_folder = '/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/images'  # 图片所在的文件夹
target_folder = '/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/label_images'  # 目标文件夹

# 创建目标文件夹和子文件夹
if not os.path.exists(target_folder):
    os.makedirs(target_folder)
for label_name in label_map.values():
    label_folder = os.path.join(target_folder, label_name)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)

# 读取 txt 文件并处理
output_lines = []
with open(input_txt, 'r') as file:
    for line in file:
        # 解析每一行
        left_image, right_image, label_str = line.strip().split()
        labels = [int(bit) for bit in label_str]  # 将标签字符串转换为列表

        # 复制图片到对应的文件夹
        for index, bit in enumerate(labels):
            if bit == 1:
                label_name = label_map[index]  # 使用字母作为文件夹名称
                # 复制左图
                src_left = os.path.join(source_folder, left_image)
                dst_left = os.path.join(target_folder, label_name, left_image)
                shutil.copy(src_left, dst_left)
                # 复制右图
                src_right = os.path.join(source_folder, right_image)
                dst_right = os.path.join(target_folder, label_name, right_image)
                shutil.copy(src_right, dst_right)
                # 添加到输出行
                output_lines.append(f"{label_name}/{left_image} {label_name}/{right_image} {label_str}\n")

# 保存新的 txt 文件
output_txt = os.path.join(target_folder, 'output.txt')
with open(output_txt, 'w') as file:
    file.writelines(output_lines)

print("处理完成！")
