import cv2
import numpy as np
import os
from PIL import Image

def batch_colorful(input_folder, output_folder):
    file_list = os.listdir(input_folder)
    for file in file_list:
        if file.endswith('.png'):
            file_path = os.path.join(input_folder, file)
            out = cv2.imread(file_path)[:,:,1]
            out = np.where(out==255, 19, out)
            name = os.path.splitext(file)[0] + '_new.png'
            output_path = os.path.join(output_folder, name)
            colorful(out, output_path)

def colorful(out, name):
    arr = out.astype(np.uint8)
    im = Image.fromarray(arr)
    palette = []
    for i in range(256):
        palette.extend((i, i, i))
    palette[:3 * 21] = np.array([[128, 64, 128],
                                 [244, 35, 232],
                                 [70, 70, 70],
                                 [102, 102, 156],
                                 [190, 153, 153],
                                 [128, 0, 128],
                                 [153, 153, 153],
                                 [250, 170, 30],
                                 [220, 220,  0],
                                 [107, 142, 35],
                                 [152, 251, 152],
                                 [70, 130, 180],
                                 [220, 20, 60],
                                 [0, 0, 142],
                                 [0, 0, 70],
                                 [0, 60, 100],
                                 [0, 80, 100],
                                 [0, 0, 230],
                                 [119, 11, 32],
                                 [0, 0, 0],
                                 [0, 64, 128]
                                 ], dtype='uint8').flatten()

    im.putpalette(palette)
    im.save(name)

input_folder = 'C:/Users/Lenovo/Desktop/cityscapesScripts-master/1/gtFine/train/aachen'  # 替换为输入文件夹的路径
output_folder = 'C:/Users/Lenovo/Desktop/cityscapesScripts-master/train'  # 替换为输出文件夹的路径
batch_colorful(input_folder, output_folder)