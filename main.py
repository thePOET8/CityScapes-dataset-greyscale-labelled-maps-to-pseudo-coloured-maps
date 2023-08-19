import os
import numpy as np
import cv2

ignore_label = 255
ID_TO_TRAINID = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                 3: ignore_label, 4: ignore_label, 5: ignore_label, 6: ignore_label,
                 7: 0, 8: 1, 9: ignore_label, 10: ignore_label, 11: 2, 12: 3, 13: 4,
                 14: ignore_label, 15: ignore_label, 16: ignore_label, 17: 5,
                 18: ignore_label, 19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
                 28: 15, 29: ignore_label, 30: ignore_label, 31: 16, 32: 17, 33: 18}


def label_to_color_image(label):
    # Color map
    color_map = np.array([
        [128, 64, 128],  # road
        [244, 35, 232],  # sidewalk
        [70, 70, 70],  # building
        [102, 102, 156],  # wall
        [190, 153, 153],  # fence
        [153, 153, 153],  # pole
        [250, 170, 30],  # traffic light
        [220, 220, 0],  # traffic sign
        [107, 142, 35],  # vegetation
        [152, 251, 152],  # terrain
        [70, 130, 180],  # sky
        [220, 20, 60],  # person
        [255, 0, 0],  # rider
        [0, 0, 142],  # car
        [0, 0, 70],  # truck
        [0, 60, 100],  # bus
        [0, 80, 100],  # train
        [0, 0, 230],  # motorcycle
        [119, 11, 32]  # bicycle
    ])

    h, w = label.shape
    color_label = np.zeros((h, w, 3), dtype=np.uint8)

    for i in range(19):
        color_label[label == i] = color_map[i]

    return color_label


# 获取图片文件列表
image_folder = "F:/rightImg16bit_trainvaltest/rightImg16bit/val"
output_folder = "C:/Users/Lenovo/Desktop/pythonProject"
image_files = os.listdir(image_folder)
# 创建输出文件夹
os.makedirs(output_folder, exist_ok=True)
# 处理每个图片文件
for i, image_file in enumerate(image_files):
    # 构建图片文件路径
    image_path = os.path.join(image_folder, image_file)

    # 读取图片
    label = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # 原始标注34类转为19类
    for k, v in ID_TO_TRAINID.items():
        label[label == k] = v

    # 生成彩色标签图像
    color_label = label_to_color_image(label)
    color_label = cv2.cvtColor(color_label, cv2.COLOR_BGR2RGB)

    # 构建输出文件路径
    output_path = os.path.join(output_folder, f"output_{i}.png")

    # 保存处理后的图片
    cv2.imwrite(output_path, color_label)
