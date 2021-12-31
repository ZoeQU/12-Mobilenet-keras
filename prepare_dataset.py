# -*- coding:utf-8 -*-
import os
import shutil
import numpy as np
import pandas as pd

# classes = ["cat", "dog"]
# sets = ["train", "test"]
# for se in sets:
#     os.mkdir('./datasets/' + str(se))
#     for class_ in classes:
#         os.mkdir('./datasets/' + str(se) + '/' + str(class_))


filenames = os.listdir('./datasets2/train')
for f_name in filenames:
    category = f_name.split('.')[0]
    old_path = './datasets2/train/' + f_name
    target_path = './datasets/train/' + category + '/' + f_name
    shutil.move(old_path, target_path)




