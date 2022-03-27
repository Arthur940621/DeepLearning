from PIL import Image
import torch
from torch.utils.data import Dataset

# 自定义数据集
class MyDataSet(Dataset):
    def __init__(self, images_path: list, image_class: list, transform = None):
        self.images_path = images_path
        self.image_class = image_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        img = Image.open(self.images_path[index])
        # RGB为彩色图像，L为灰度图像
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[index]))
        label = self.image_class[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    # 无需实现，官方有默认方法
    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim = 0)
        labels = torch.as_tensor(labels)
        return images, labels
