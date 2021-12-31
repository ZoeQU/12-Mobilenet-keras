# -*- coding:utf-8 -*-
# 轻量模型 手机 深度可分离卷积块 替代3x3普通卷积

from __future__ import absolute_import, print_function

import warnings
from keras.utils import plot_model
import numpy as np
from keras import backend as K
from keras import layers
from keras.applications.imagenet_utils import decode_predictions
from keras.engine.topology import get_source_inputs
from keras.layers import (Activation, AveragePooling2D, DepthwiseConv2D, BatchNormalization,
                          Conv2D, Dense, GlobalAveragePooling2D,
                          Input, Dropout, Reshape)
from keras.models import Model
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.utils.layer_utils import convert_all_kernels_in_model


def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    filters = int(filters * alpha)
    x = Conv2D(filters, kernel,
               padding='same',
               use_bias=False,
               strides=strides, name='conv1')(inputs)
    x = BatchNormalization(name='conv1_bn')(x)
    return Activation(relu6, name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha, depth_multiplier=1, strides=(1, 1), block_id=1):
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)
    x = DepthwiseConv2D((3, 3), padding='same',
                        depth_multiplier=depth_multiplier,
                        strides=strides,
                        use_bias=False,
                        name='conv_dw_%d' % block_id)(inputs)
    x = BatchNormalization(name='conv_dw_%d_bn' % block_id)(x)
    x = Activation(relu6, name='conv_dw_%d_relu' % block_id)(x)
    x = Conv2D(pointwise_conv_filters, (1, 1),
               padding='same',
               use_bias=False,
               strides=(1, 1),
               name='conv_pw_%d' % block_id)(x)
    x = BatchNormalization(name='conv_pw_%d_bn' % block_id)(x)
    return Activation(relu6, name='conv_pw_%d_relu' % block_id)(x)


def MobileNet(input_shape=None, alpha=1.0, depth_multiplier=1, dropout=1e-3, classes=1000):
    img_input = Input(shape=input_shape)

    x = _conv_block(img_input, 32, alpha, strides=(2, 2))

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, strides=(2, 2), block_id=2)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)

    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=12)
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=13)

    x = GlobalAveragePooling2D()(x)

    shape = (1, 1, int(1024 * alpha))

    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(dropout, name='dropout')(x)

    x = Conv2D(classes, (1, 1), padding='same', name='conv_preds')(x)
    x = Activation('softmax', name='act_softmax')(x)
    x = Reshape((classes,), name='reshape_2')(x)

    inputs = img_input

    model = Model(inputs, x, name='mobilenet_%0.2f' % (alpha))
    return model


def relu6(x):
    return K.relu(x, max_value=6)


if __name__ == '__main__':
    model = MobileNet(input_shape=(224, 224, 3))
    model.summary()  #输出模型各层的参数状况
    plot_model(model, "net.svg", show_shapes=True)
    # weights_path = get_file('inception_v3_weights_tf_dim_ordering_tf_kernels.h5', WEIGHTS_PATH, cache_subdir='models',
    #                         md5_hash='9a0d58056eeedaa3f26cb7ebd46da564')
    # model.load_weights(weights_path)
    #
    # img_path = 'elephant.jpg'
    # img = image.load_img(img_path, target_size=(299, 299))
    # x = image.img_to_array(img)
    # x = np.expand_dims(x, axis=0)
    # x = preprocess_input(x)
    # print('Input image shape:', x.shape)
    #
    # preds = model.predict(x)
    # print('Predicted:', decode_predictions(preds))