import os
import shutil

# 定义文件夹和标签的映射
folder_label_map = {
    'A1': '01000000',
    'C1': '00100000',
    'D1': '00010000',
    'G1': '00001000',
    'H1': '00000100',
    'M1': '00000010'
}

# 输入路径
new_dataset_folder = '/mnt/mydisk/medical_seg/fwwb_a007/data/new'  # 新数据集的文件夹路径
target_folder = '/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/label_images'  # 目标文件夹路径
input_txt = '/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/label_images/output.txt'  # 原有的 txt 文件路径

# 遍历新数据集中的每个文件夹
for folder_name, label in folder_label_map.items():
    source_folder = os.path.join(new_dataset_folder, folder_name)
    target_subfolder = os.path.join(target_folder, folder_name[0])  # 目标文件夹（A, C, D, G, H, M）

    # 确保目标文件夹存在
    if not os.path.exists(target_subfolder):
        os.makedirs(target_subfolder)

    # 遍历源文件夹中的图片
    for file_name in os.listdir(source_folder):
        if file_name.endswith('left.jpg'):
            # 构建新的文件名
            new_file_name = f"extra_{file_name[:-8]}"
            # 构建源路径和目标路径
            src_path = os.path.join(source_folder, file_name[:-8])
            dst_path = os.path.join(target_subfolder, new_file_name)
            # 移动文件
            shutil.move(src_path + 'left' + '.jpg', dst_path + 'left' + '.jpg')
            shutil.move(src_path + 'right' + '.jpg', dst_path + 'right' + '.jpg')

            # 添加到 txt 文件的内容
            with open(input_txt, 'a') as file:
                file.write(f"{folder_name[0]}/{new_file_name + 'left' + '.jpg'} {folder_name[0]}/{new_file_name + 'right' + '.jpg'} {label}\n")

print("处理完成！")
