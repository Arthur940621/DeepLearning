import torch.autograd
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import load_minist_data
from torchvision.utils import save_image
import numpy as np
import os
from torch.autograd import Variable

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
        img = img.view(img.shape[0], *img_shape)
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
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


# 梯度惩罚损失权重
lambda_gp = 10

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

# 优化器
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.00005)
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.00005)

def compute_gradient_penalty(D, real_samples, fake_samples):
    # 随机权重初始化
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    # 得到real samples和fake samples之间的样本
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    
    d_interpolates = D(interpolates)
    fake = Variable(torch.Tensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False).to(device)
    # 对interpolates求梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

for epoch in range(1, 31):
    for i, (img, _) in enumerate(train_loader):
        img_num = img.shape[0]
        real_img = img.to(device)
        d_optimizer.zero_grad()

        z = torch.randn(img_num, 100).to(device)
        fake_img = G(z).detach()

        gradient_penalty = compute_gradient_penalty(D, real_img.data, fake_img.data)
        d_loss = -torch.mean(D(real_img)) + torch.mean(D(fake_img)) + lambda_gp * gradient_penalty
        d_loss.backward()
        d_optimizer.step()

        z = torch.randn(img_num, 100).to(device)
        g_optimizer.zero_grad()
        fake_img = G(z)
        g_loss = -torch.mean(D(fake_img))
        g_loss.backward()
        g_optimizer.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, 30, i, len(train_loader), d_loss.item(), g_loss.item()))

    save_image(fake_img, './imgs/fake_images-{}.png'.format(epoch))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')