import re
import matplotlib.pyplot as plt

# 配置日志文件路径
log_file = "/mnt/mydisk/medical_seg/fwwb_a007/mmpretrain/work_dirs/resnet50_8xb32_in1k/20250130_164729/20250130_164729.log"  # 你的日志文件

# 用于存储数据
epochs = []
precisions = []

# 读取日志并提取精度
with open(log_file, "r") as f:
    for line in f:
        match = re.search(r"strict-multi-label/precision:\s*([\d\.]+)", line)
        if match:
            precision = float(match.group(1))
            epochs.append(len(epochs) + 1)  # 假设每次出现精度时就是一个新的 epoch
            precisions.append(precision)

# 绘制精度曲线
plt.figure(figsize=(10, 5))
plt.plot(epochs, precisions, marker='o', linestyle='-')
plt.xlabel("Epoch")
plt.ylabel("Precision")
plt.title("Strict Multi-Label Precision over Epochs")
plt.grid(True)
plt.savefig("precision.png")
plt.show()


# 找出最高的五个精度值
top5 = sorted(zip(epochs, precisions), key=lambda x: x[1], reverse=True)[:5]

# 打印最高的5个 epoch 及其精度
print("Top 5 Epochs with Highest Precision:")
for epoch, precision in top5:
    print(f"Epoch {epoch}: Precision {precision:.4f}")