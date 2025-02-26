import cv2
import numpy as np

def color_transfer(source, target):
    """
    将 target 图像的颜色迁移到 source 图像上
    :param source: 源图像 (BGR 格式)
    :param target: 目标图像 (BGR 格式)
    :return: 颜色迁移后的图像 (BGR 格式)
    """
    # 将图像转换为 LAB 颜色空间
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB)

    # 计算 source 和 target 的均值和标准差
    source_mean, source_std = cv2.meanStdDev(source_lab)
    target_mean, target_std = cv2.meanStdDev(target_lab)

    # 将均值和标准差的形状从 (3, 1) 调整为 (1, 1, 3)
    source_mean = source_mean.reshape(1, 1, 3)
    source_std = source_std.reshape(1, 1, 3)
    target_mean = target_mean.reshape(1, 1, 3)
    target_std = target_std.reshape(1, 1, 3)

    # 标准化 source 图像
    source_normalized = (source_lab - source_mean) / source_std
    # 应用 target 的均值和标准差
    result_lab = source_normalized * target_std + target_mean

    # 将结果限制在 LAB 空间的合法范围内 (0 到 255)
    result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)

    # 转换回 BGR 颜色空间
    result = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return result

# 示例使用
# 加载源图像和目标图像
source_image = cv2.imread('/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/images/2_left.jpg')
target_image = cv2.imread('/mnt/mydisk/medical_seg/fwwb_a007/data/data_merge/images/6_left.jpg')

# 确保图像加载成功
if source_image is None or target_image is None:
    print("Error: 图片加载失败，请检查路径是否正确！")
else:
    # 执行颜色交换
    result_image = color_transfer(source_image, target_image)

    # 保存结果
    cv2.imwrite('result.jpg', result_image)
    print("颜色交换完成！结果已保存为 result.jpg")

    # # 显示结果（可选）
    # cv2.imshow('Source Image', source_image)
    # cv2.imshow('Target Image', target_image)
    # cv2.imshow('Result Image', result_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

