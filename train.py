# -*- coding:utf-8 -*-
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import adam
from mobilenet import MobileNet
from utils.callbacks import ExponentDecayScheduler, LossHistory
from utils.dataloader import ClsDatasets
from utils.utils import get_classes


if __name__ == '__main__':
    classes_path = 'model_data/cls_classes.txt'
    input_shape = [224, 224]
    alpha = 0.25
    model_path = ''
    Freeze_layers = 81
    Freeze_Train = False
    annotation_path = 'cls_train.txt'
    val_split = 0.1     # 进行训练集和验证集的划分，默认使用10%的数据用于验证
    num_workers = 1     # 用于设置是否使用多线程读取数据，1代表关闭多线程; 开启后会加快数据读取速度，但是会占用更多内存; 在IO为瓶颈的时候再开启多线程，即GPU运算速度远大于读取图片的速度。
    class_name, num_classes = get_classes(classes_path)    # 获取classes
    model = MobileNet(input_shape=[input_shape[0], input_shape[0], 3], alpha=alpha, classes=num_classes)

    if model_path != '':
        # 载入预训练权重
        print('Load weights {}.'.format(model_path))
        model.load_weights(model_path, by_name=True, skip_mismatch=True)

    #-------------------------------------------------------------------------------#
    #   训练参数的设置
    #   logging表示tensorboard的保存地址
    #   checkpoint用于设置权值保存的细节，period用于修改多少epoch保存一次
    #   reduce_lr用于设置学习率下降的方式
    #   early_stopping用于设定早停，val_loss多次不下降自动结束训练，表示模型基本收敛
    #-------------------------------------------------------------------------------#
    logging = TensorBoard(log_dir='logs/')
    checkpoint = ModelCheckpoint('logs/ep{epoch:03d}-loss{loss:.3f}.h5',
                                 monitor='val_loss',
                                 save_weights_only=True, save_best_only=False, period=1)  # 回调函数
    reduce_lr = ExponentDecayScheduler(decay_rate=0.94, verbose=1)
    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1)
    loss_history = LossHistory('logs/')

    with open(annotation_path, 'a+') as f:
        f.seek(0, 0)  # 读取文件时发现f.readlines()读取内容为空,本来从最后一行读取，而最后一行为空;可以使用f.seek(0)将游标移动到文章开头再次调用f.read()即可获取内容.
        lines = f.readlines()
    np.random.seed(10101)  #seed( ) 用于指定随机数生成时所用算法开始的整数值，如果使用相同的seed( )值，则每次生成的随即数都相同，如果不设置这个值，则系统根据时间来自己选择这个值，此时每次生成的随机数因时间差异而不同。
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    if Freeze_Train:
        for i in range(Freeze_layers):
            model.layers[i].trainable = False

    if True:
        batch_size = 32
        lr = 1e-3
        Init_Epoch = 0
        Freeze_Epoch = 50
        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.compile(loss='categorical_crossentropy', optimizer=adam(lr=lr), metrics=['categorical_accuracy'])

        train_dataloader = ClsDatasets(lines[:num_train], input_shape, batch_size, num_classes, train=True)
        val_dataloader = ClsDatasets(lines[num_train:], input_shape, batch_size, num_classes, train=False)

        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=epoch_step,
            validation_data=val_dataloader,
            validation_steps=epoch_step_val,
            epochs=Freeze_Epoch,
            initial_epoch=Init_Epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )


    if Freeze_Train:
        for i in range(Freeze_layers):
            model.layers[i].trainable = True

    if True:
        batch_size = 32
        Lr = 1e-4
        Freeze_Epoch = 50
        Epoch = 100

        epoch_step = num_train // batch_size
        epoch_step_val = num_val // batch_size

        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError('数据集过小，无法进行训练，请扩充数据集。')

        print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))

        model.compile(loss='categorical_crossentropy', optimizer=adam(lr=lr), metrics=['categorical_accuracy'])

        train_dataloader = ClsDatasets(lines[:num_train], input_shape, batch_size, num_classes, train=True)
        val_dataloader = ClsDatasets(lines[num_train:], input_shape, batch_size, num_classes, train=False)

        model.fit_generator(
            generator=train_dataloader,
            steps_per_epoch=epoch_step,
            validation_data=val_dataloader,
            validation_steps=epoch_step_val,
            epochs=Epoch,
            initial_epoch=Freeze_Epoch,
            use_multiprocessing=True if num_workers > 1 else False,
            workers=num_workers,
            callbacks=[logging, checkpoint, reduce_lr, early_stopping, loss_history]
        )