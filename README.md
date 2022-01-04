# 12-Mobilenet-keras

尝试用 Mobilenet 去 classification 的完整的train 和 test 过程。

数据集：cat-vs-dog，test数据集没有找到相应的ground truth所以没有用。
       需要注意train中文件夹的结构和格式。有可能需要用到 prepare_dataset.py 和 txt_annotation.py (用来生成 cls_classes.txt )
       
最终用来predict的权重，即是logs中loss最小的权重。
