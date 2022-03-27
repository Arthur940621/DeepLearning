
import os
import random
from PIL import Image
from torch.utils.data import Dataset
import torch
from torch.autograd import Variable

def read_split_data(root: str): # 数据集路径和验证集划分比例
    random.seed(0) # 随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹，一个文件夹对应一个类别
    flower_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    flower_class.sort()
    every_class_num = [] # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"] # 支持的文件后缀类型

    # 遍历每个类别下的文件
    for cla in flower_class:
        if cla == "trainA":
            cla_path_a = os.path.join(root, cla)
            # 遍历获取该类别supported支持的所有文件路径
            images_a = [os.path.join(root, cla, i) for i in os.listdir(cla_path_a) if os.path.splitext(i)[-1] in supported]
            # 记录该类别的样本数
            every_class_num.append(len(images_a))
        elif cla == "trainB":
            cla_path_b = os.path.join(root, cla)
            images_b = [os.path.join(root, cla, i) for i in os.listdir(cla_path_b) if os.path.splitext(i)[-1] in supported]
    return images_a, images_b

# 自定义数据集
class MyDataSet(Dataset):
    def __init__(self, images_path_a: list, images_path_b: list, transform = None):
        self.images_path_a = images_path_a
        self.images_path_b = images_path_b
        self.transform = transform

    def __len__(self):
        return max(len(self.images_path_a), len(self.images_path_b))

    def __getitem__(self, index):
        img_a = Image.open(self.images_path_a[index % len(self.images_path_a)])
        img_b = Image.open(self.images_path_b[random.randint(0, len(self.images_path_b) - 1)])
        # RGB为彩色图像，L为灰度图像
        if img_a.mode != 'RGB' or img_b.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[index]))
        if self.transform is not None:
            item_A = self.transform(img_a)
            item_B = self.transform(img_b)
        return {"A": item_A, "B": item_B}

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

class ReplayBuffer:
    def __init__(self, max_size=50):
        assert max_size > 0, "Empty buffer or trying to create a black hole. Be careful."
        self.max_size = max_size
        self.data = []

    def push_and_pop(self, data):
        to_return = []
        for element in data.data:
            element = torch.unsqueeze(element, 0)
            if len(self.data) < self.max_size:
                self.data.append(element)
                to_return.append(element)
            else:
                if random.uniform(0, 1) > 0.5:
                    i = random.randint(0, self.max_size - 1)
                    to_return.append(self.data[i].clone())
                    self.data[i] = element
                else:
                    to_return.append(element)
        return Variable(torch.cat(to_return))