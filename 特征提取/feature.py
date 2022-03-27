import torch
from torchvision import models
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
import numpy as np

# 设定模型为resnet34，预训练为True
model = models.resnet34(pretrained=True)
# 不启用Batch Normalization
model.eval()

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
     ])

# 载入图片
img = Image.open("./tulip.jpg")
# transform增强
img = data_transform(img)
# 扩展batch维度，[N,C,H,W]
img = torch.unsqueeze(img, dim=0)

# 特征图保存
def save_img(img, name, shape):
    feature_show = np.squeeze(img.detach().numpy())
    img_show = np.transpose(feature_show, [1, 2, 0])
    # 特征矩阵的每一个channel所对应的是一个二维的的特征矩阵，就像灰度图像一样，channel=1
    plt.figure(figsize=(shape[0] + shape[1], shape[0] + shape[1]))
    for i in range(shape[0] * shape[1]):
        ax = plt.subplot(shape[0], shape[1], i+1)
        plt.imshow(img_show[:,:,i])
        plt.xticks([]),plt.yticks([])
    plt.savefig(name + '.jpg')


# 抽取出某一层的特征图
def extract_feature(img, model, name):
    feature = img
    # named_children()返回子模块的迭代器
    for n, m in model.named_children():
        feature = m(feature)
        if n == name:
            break
    return feature


f1 = extract_feature(img, model, "conv1") # [1, 64, 112, 112]
save_img(f1, 'f1', [8, 8])
f2 = extract_feature(img, model, "layer1") # [1, 64, 56, 56]
save_img(f2, 'f2', [8, 8])
f3 = extract_feature(img, model, "layer2") # [1, 128, 28, 28]
save_img(f3, 'f3', [16, 8])
f4 = extract_feature(img, model, "layer3") # [1, 256, 14, 14]
save_img(f4, 'f4', [32, 8])
f5 = extract_feature(img, model, "layer4") # [1, 512, 7, 7]
save_img(f5, 'f5', [64, 8])







