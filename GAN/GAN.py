import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import load_minist_data
from torchvision.utils import save_image
import numpy as np
import os

os.makedirs("imgs", exist_ok=True)

img_shape = (1, 28, 28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), *img_shape)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # (x-mean) / std
])

 # 将数据集转换为Tensor的类
class DealDataset(Data.Dataset):
    def __init__(self, folder, data_name, label_name, transform = None):
        (dataSet, labels) = load_minist_data.load_data(folder, data_name, label_name)
        self.dataSet = dataSet
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.dataSet[index], int(self.labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.dataSet)

# 设置文件路径
data_folder = "./data/mnist"
train_data_name = "train-images-idx3-ubyte.gz"
train_label_name = "train-labels-idx1-ubyte.gz"

# 实例化这个类，然后我们就得到了Dataset类型的数据
trainDataset = DealDataset(data_folder, train_data_name, train_label_name, img_transform)

# 数据的装载
train_loader = Data.DataLoader(dataset = trainDataset, batch_size = 64, shuffle = True)

# 创建对象
D = Discriminator().to(device)
G = Generator().to(device)


# 损失函数
criterion = nn.BCELoss()

# 优化器
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)


for epoch in range(1, 31):
    for i, (img, _) in enumerate(train_loader):
        # ============================训练判别器============================
        img_num = img.shape[0]
        real_img = img.to(device)
        real_label = torch.ones(img_num, 1).to(device)  # 定义真实的图片label为1
        fake_label = torch.zeros(img_num, 1).to(device)  # 定义假的图片的label为0
        d_optimizer.zero_grad()  # 在反向传播之前，先将梯度归0
        # 判别器训练train
        # 分为两部分：1、真的图像判别为真；2、假的图像判别为假
        # 计算真实图片的损失
        real_out = D(real_img)  # 将真实图片放入判别器中
        d_loss_real = criterion(real_out, real_label)  # 得到真实图片的loss
        real_scores = real_out  # 得到真实图片的判别值，输出的值越接近1越好
        # 计算假的图片的损失
        z = torch.randn(img_num, 100).to(device)  # 随机生成一些噪声
        fake_img = G(z).detach()  # 随机噪声放入生成网络中，生成一张假的图片，避免梯度传到G，因为G不用更新, detach分离
        fake_out = D(fake_img)  # 判别器判断假的图片，
        d_loss_fake = criterion(fake_out, fake_label)  # 得到假的图片的loss
        fake_scores = fake_out  # 得到假图片的判别值，对于判别器来说，假图片的损失越接近0越好
        # 损失函数和优化
        d_loss = d_loss_real + d_loss_fake  # 损失包括判真损失和判假损失
        d_loss.backward()  # 将误差反向传播
        d_optimizer.step()  # 更新参数

        # ============================训练生成器============================
        # 目的是希望生成的假的图片被判别器判断为真的图片，在此过程中，将判别器固定，将假的图片传入判别器的结果与真实的label对应
        # 反向传播更新的参数是生成网络里面的参数，这样可以通过更新生成网络里面的参数，来训练网络，使得生成的图片让判别器以为是真的
        z = torch.randn(img_num, 100).to(device)  # 得到随机噪声
        g_optimizer.zero_grad()  # 梯度归0
        fake_img = G(z)  # 随机噪声输入到生成器中，得到一副假的图片
        output = D(fake_img)  # 经过判别器得到的结果
        g_loss = criterion(output, real_label)  # 得到的假的图片与真实的图片的label的loss
        g_loss.backward()  # 进行反向传播
        g_optimizer.step()  # .step()一般用在反向传播后面,用于更新生成网络的参数

        # 打印中间的损失
        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, 30, i, len(train_loader), d_loss.item(), g_loss.item()))

    save_image(fake_img, './imgs/fake_images-{}.png'.format(epoch))

# 保存模型
torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')