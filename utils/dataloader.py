# -*- coding:utf-8 -*-
import math
from random import shuffle
import cv2
import keras
import numpy as np
from keras.utils import np_utils, Sequence
from PIL import Image
from .utils import cvtColor, preprocess_input


class ClsDatasets(Sequence):
    def __init__(self, annotation_lines, input_shape, batch_size, num_classes, train):
        """
        __init__()方法是一种特殊的方法，被称为类的初始化方法，当创建这个类的实例时就会调用该方法
        self 代表类的实例，self 在定义类的方法时是必须有的，虽然在调用时不必传入相应的参数
        :param annotation_lines:
        :param input_shape:
        :param batch_size:
        :param num_classes:
        :param train:
        """
        self.annotation_lines = annotation_lines
        self.length = len(annotation_lines)
        self.input_shape = input_shape
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.train = train   ###?不是 True 啥的?

    def __len__(self):
        # __len__  python语言定义的特殊方法，避免自定义成这种方法。
        return math.ceil(len(self.annotation_lines) / float(self.batch_size))

    def __getitem__(self, index):
        X_train = []
        Y_train = []
        for i in range(index * self.batch_size, (index + 1) * self.batch_size):
            i = i % self.length
            annotation_path = self.annotation_lines[i].split(';')[1].split()[0]
            image = Image.open(annotation_path)
            image = self.get_random_data(image, self.input_shape, random=self.train)
            image = preprocess_input(np.array(image).astype(np.float32))

            X_train.append(image)
            Y_train.append(int(self.annotation_lines[i].split(';')[0]))

        X_train = np.array(X_train)
        Y_train = np_utils.to_categorical(np.array(Y_train), num_classes=self.num_classes)
        return X_train, Y_train

    def on_epoch_begin(self):
        """
        将序列的所有元素随机排序。
        :return: annotation_lines
        """
        shuffle(self.annotation_lines)

    def rand(self, a=0, b=1):
        """
        提供一个随机参数，用于扭曲和缩放
        :param a:
        :param b:
        :return:
        """
        return np.random.rand() * (b - a) + a

    def get_random_data(self, image, input_shape, jitter=.3, hue=.1, sat=1.5, val=1.5, random=True):
        """
        读取图像并转换成RGB图像, 获得图像的高宽与目标高宽
        :param image:
        :param input_shape:
        :param jitter:
        :param hue:
        :param sat:
        :param val:
        :param random:
        :return:
        """
        image = cvtColor(image)
        iw, ih = image.size
        h, w = input_shape

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # 将图像多余的部分加上灰条
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            return image_data

        # 对图像进行缩放并且进行长和宽的扭曲
        new_ar = w / h * self.rand(1-jitter, 1+jitter) / self.rand(1-jitter, 1+jitter)
        scale = self.rand(.75, 1.25)
        if new_ar < 1:
            nh = int(scale * h)
            nw = int(nh * new_ar)
        else:
            nw = int(scale * w)
            nh = int(nw / new_ar)
        image = image.resize((nw, nh), Image.BICUBIC)

        # 将图像多余的部分加上灰条
        dx = int(self.rand(0, w - nw))
        dy = int(self.rand(0, h - nh))
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image = new_image

        # 翻转图像
        flip = self.rand() < .5
        if flip:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)

        rotate = self.rand() < .5
        if rotate:
            angle = np.random.randint(-15, 15)
            a, b = w / 2, h / 2
            M = cv2.getRotationMatrix2D((a, b), angle, 1)
            image = cv2.warpAffine(np.array(image), M, (w, h), borderValue=[128, 128, 128])

        # 色域扭曲
        hue = self.rand(-hue, hue)
        sat = self.rand(1, sat) if self.rand() < .5 else 1 / self.rand(1, sat)
        val = self.rand(1, val) if self.rand() < .5 else 1 / self.rand(1, val)
        x = cv2.cvtColor(np.array(image, np.float32) / 255, cv2.COLOR_BGR2HSV)
        x[..., 1] *= sat
        x[..., 2] *= val
        x[x[:, :, 0] > 360, 0] = 360
        x[:, :, 1:][x[:, :, 1:] > 1] = 1
        x[x < 0] = 0
        image_data = cv2.cvtColor(x, cv2.COLOR_HSV2RGB) * 255

        return image_data