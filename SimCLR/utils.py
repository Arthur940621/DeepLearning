from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
from torchvision.transforms.transforms import Resize

# 自定义数据集
class MyDataSet(Dataset):
    def __init__(self, images_path: list, image_class: list, transform1 = None, transform2 = None):
        self.images_path = images_path
        self.image_class = image_class
        self.transform1 = transform1
        self.transform2 = transform2

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        # RGB为彩色图像，L为灰度图像
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[index]))
        label = self.image_class[index]
        if self.transform1 is not None:
            img1 = self.transform1(img)
        if self.transform2 is not None:
            img2 = self.transform2(img)
        return img1, img2, label

import os

def read_split_data(root: str): # 数据集路径和验证集划分比例
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(flower_class))

    train_images_path = [] # 存储训练集的所有图片路径
    train_images_label = [] # 存储训练集图片对应的索引信息
    every_class_num = [] # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG", ".BMP", ".bmp"] # 支持的文件后缀类型

    # 遍历每个类别下的文件
    for cla in flower_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取该类别supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path) if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数
        every_class_num.append(len(images))

        # 遍历某个类别的所有文件
        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    return train_images_path, train_images_label

data_transform = {
    "train1":transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.RandomResizedCrop(32),
    transforms.RandomGrayscale(p = 0.2),
    transforms.RandomRotation(60),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
    "train2":transforms.Compose([
    transforms.Resize([128, 128]),
    transforms.RandomResizedCrop(32),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p = 0.8),
    
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ]),
    "val":transforms.Compose([
    transforms.Resize([32, 32]),
    transforms.ToTensor(),
    transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])
}
