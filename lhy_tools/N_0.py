# 定义输入和输出文件路径
input_file_path = '/mnt/mydisk/medical_seg/fwwb_a007/data/dataset/val.txt'
output_file_path = '/mnt/mydisk/medical_seg/fwwb_a007/data/dataset/val.txt'

# 打开输入文件并读取内容
with open(input_file_path, 'r', encoding='utf-8') as infile:
    lines = infile.readlines()

# 用于存储处理后的行
processed_lines = []

# 遍历每一行
for line in lines:
    # 分割文件名和标签
    filename, label = line.strip().split()

    # 如果标签长度大于等于 2，将倒数第二个数字移到第一位
    if len(label) >= 2:
        new_label = label[-2] + label[:-2] + label[-1]
    else:
        # 如果标签长度小于 2，保持不变
        new_label = label

    # 重新组合文件名和新标签
    processed_line = f"{filename} {new_label}\n"
    processed_lines.append(processed_line)

# 打开输出文件并写入处理后的内容
with open(output_file_path, 'w', encoding='utf-8') as outfile:
    outfile.writelines(processed_lines)

print(f"处理完成，结果已保存到 {output_file_path}")