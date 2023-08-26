import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

# 示例用法
image_folder = 'F:/rightImg8bit_trainvaltest/rightImg8bit/train/'
mask_folder = 'F:/gtFine_trainvaltest/gtFine/train/'

# 获取文件列表
image_files = glob.glob(image_folder + '*.png')
mask_files = [file for file in glob.glob(mask_folder + '*.png') if 'color' in os.path.basename(file)]

# 循环处理每个文件
for image_path, mask_path in zip(image_files, mask_files):
    # 加载图像和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # 将mask转换为灰度图像
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 创建一个具有相同形状的三通道掩码
    mask_color = np.zeros_like(image)

    # 生成固定的颜色列表
    fixed_colors = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]  # 例如，这里使用红、绿、蓝三种颜色

    # Cityscapes标签颜色映射
    label_colors = {label: color for label, color in zip(np.unique(mask_gray), fixed_colors)}

    def visualize_image_mask(image, mask_gray, label_colors):
        # 为每个label设置颜色
        for label in np.unique(mask_gray):
            if label == 0:
                continue
            if label in label_colors:
                mask_color[mask_gray == label] = label_colors[label]
            else:
                mask_color[mask_gray == label] = image[mask_gray == label]

        # 绘制原始图像和掩码图像
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axs[0].set_title('Original Image')
        axs[0].axis('off')
        axs[1].imshow(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB))
        axs[1].set_title('Mask')
        axs[1].axis('off')

        plt.show()

    visualize_image_mask(image, mask_gray, label_colors)