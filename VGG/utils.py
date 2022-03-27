import os
import random

import matplotlib.pyplot as plt

def read_split_data(root: str, val_rate: float = 0.2): # 数据集路径和验证集划分比例
    random.seed(0) # 随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))

    train_images_path = [] # 存储训练集的所有图片路径
    train_images_label = [] # 存储训练集图片对应的索引信息
    val_images_path = [] # 存储验证集的所有图片路径
    val_images_label = [] # 存储验证集图片对应的索引信息
    every_class_num = [] # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"] # 支持的文件后缀类型

    # 遍历每个类别下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取该类别supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k = int(len(images) * val_rate))

        # 遍历某个类别的所有文件
        for img_path in images:
            if img_path in val_path: # 如果该路径在采样的验证集样本中，则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else: # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset".format(sum(every_class_num)))

    return train_images_path, train_images_label, val_images_path, val_images_label