# -*- coding:utf-8 -*-
import numpy as np
from PIL import Image


def letterbox_image(image, size):
    """
    图像比例不变的,图像不失真的resize,不足区域用灰色 rgb=(128, 128, 128) 填充
    :param image:
    :param size:
    :return:
    """
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw)//2, (h - nh)//2))

    return new_image


def get_classes(classes_path):
    """
    获得类
    :param classes_path: 路径
    :return: class_names
    """
    with open(classes_path, encoding='utf-8') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]   #str.strip() 用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    return class_names, len(class_names)


def cvtColor(image):
    """
    将图像转换成RGB图像，防止灰度图在预测时报错。
    代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB。
    :param image:
    :return:
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def preprocess_input(x):
    """
    预处理训练图片
    :param x:
    :return:
    """
    x /= 127.5
    x -= 1.
    return x