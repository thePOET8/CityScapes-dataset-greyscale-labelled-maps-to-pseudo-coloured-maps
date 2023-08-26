# CityScapes-dataset-greyscale-labelled-maps-to-pseudo-coloured-maps
CityScapes数据集灰度标签图转为伪彩色图


目前搜集到的代码都或多或少存在个别问题，针对CityScapes数据集在训练网络时可能需要将灰度标签图转为伪彩色图这一需求，

其中的三个程序分别针对不同需求：


**Mask叠加在原图**
![Mask叠加在原图](https://github.com/thePOET8/CityScapes-dataset-greyscale-labelled-maps-to-pseudo-coloured-maps/blob/main/README.assets/Mask%E5%8F%A0%E5%8A%A0%E5%9C%A8%E5%8E%9F%E5%9B%BE.jpg)



**灰度标签图转为伪彩色图(main.py）**



**批量输出Mask图像，同时保存标注label的颜色值**
优点：可以在多次运行时，输出的label颜色不变，且可修改，（第一次运行时自动生成随机且不重复的颜色值）
![批量输出Mask图像](https://github.com/thePOET8/CityScapes-dataset-greyscale-labelled-maps-to-pseudo-coloured-maps/blob/main/README.assets/Mask.jpg)
