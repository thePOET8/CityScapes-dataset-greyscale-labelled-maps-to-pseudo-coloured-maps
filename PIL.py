import cv2
import numpy as np
import glob
import os
from PIL import Image
# 示例用法
image_folder = 'F:/rightImg8bit_trainvaltest/rightImg8bit/train/'
mask_folder = 'F:/gtFine_trainvaltest/gtFine/train/'
output_folder = 'C:/Users/Lenovo/Desktop/pythonProject'
file_path = 'C:/Users/Lenovo/Desktop/pythonProject/label_colors.npy'

# 获取文件列表
image_files = glob.glob(image_folder + '*.png')
mask_files = [file for file in glob.glob(mask_folder + '*.png') if 'color' in os.path.basename(file)]

# 加载标签颜色数组文件（如果存在）
def load_label_colors(file_path):
    if os.path.exists(file_path):
        label_colors = np.load(file_path)
    else:
        label_colors = np.zeros((256, 3), dtype=np.uint8)
    return label_colors

label_colors = load_label_colors(file_path)

# 循环处理每个文件
for image_path, mask_path in zip(image_files, mask_files):
    # 加载图像和mask
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path)

    # 将mask转换为灰度图像
    mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

    # 创建一个具有相同形状的三通道掩码
    mask_color = np.zeros_like(image)

    # 获取当前文件的所有标签值
    labels = np.unique(mask_gray)

    # 为每个标签值分配颜色
    for label in labels:
        if label == 0:
            continue
        if np.any(label_colors[label]):
            # 使用标签颜色数组中的颜色
            color = label_colors[label]
        else:
            # 如果标签不在数组中，则生成新的随机颜色
            color = np.random.randint(0, 256, size=3, dtype=np.uint8)
            label_colors[label] = color

        # 将颜色应用于掩码
        mask_color[mask_gray == label] = color

    # 创建一个与画布大小相同的空白图像
    canvas = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    # 将生成的图像放置在空白图像的中心位置
    x_offset = int((canvas.shape[1] - mask_color.shape[1]) / 2)
    y_offset = int((canvas.shape[0] - mask_color.shape[0]) / 2)
    canvas[y_offset:y_offset + mask_color.shape[0], x_offset:x_offset + mask_color.shape[1]] = mask_color


    def visualize_mask(mask_color):
        mask_color = mask_color[1:-1, 1:-1]
        mask_color_resized = cv2.resize(cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB), (2048, 1024))  # 调整图像大小

        # 转换为PIL图像
        pil_image = Image.fromarray(mask_color_resized)

        # 转换为8位
        pil_image = pil_image.quantize()

        output_path = os.path.join(output_folder, os.path.splitext(os.path.basename(image_path))[0] + ".png")

        # 保存图像
        pil_image.save(output_path)

        print(f"Saved image to {output_path}")


    visualize_mask(canvas)

# 保存标签颜色数组为文件
np.save(file_path, label_colors)