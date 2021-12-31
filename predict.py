# -*- coding:utf-8 -*-
from PIL import Image
import numpy as np
from keras.utils import plot_model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
import matplotlib.pyplot as plt
from mobilenet import MobileNet
from utils.utils import cvtColor, get_classes, letterbox_image, preprocess_input


def detect_image(image, input_shape):
    """detect_image"""
    image = cvtColor(image)
    image_data = letterbox_image(image, [input_shape[1], input_shape[0]])
    # 归一化+添加上batch_size维度
    image_data = np.expand_dims(preprocess_input(np.array(image_data, np.float32)), 0)
    # 图片传入网络进行预测
    preds = model.predict(image_data)[0]
    # 获得所属种类
    classes_path = 'model_data/cls_classes.txt'
    class_names, num_classes = get_classes(classes_path)
    class_name = class_names[np.argmax(preds)]
    probability = np.max(preds)

    plt.subplot(1, 1, 1)
    plt.imshow(np.array(image))
    plt.title('Class:%s Probability:%.3f' % (class_name, probability))
    plt.show()
    return class_name


if __name__ == '__main__':
    input_shape = [224, 224]
    model = MobileNet(input_shape=[input_shape[0], input_shape[0], 3], alpha=0.25, depth_multiplier=1, dropout=1e-3, classes=2)
    plot_model(model, "net.svg", show_shapes=True)
    weights_path = 'model_data/mobilenet025_catvsdog.h5' #problem: prams not suitable
    model.load_weights(weights_path)

    img_path = 'img/cat.jpg'
    image = Image.open(img_path)
    class_name = detect_image(image, input_shape=[224, 224])
    print(class_name)