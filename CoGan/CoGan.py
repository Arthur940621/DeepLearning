import argparse
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import torch.utils.data as Data
from torchvision.utils import save_image
import os
import numpy as np

os.makedirs("imgs", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=32, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=28, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval betwen image samples")
opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)

class CoupledGenerators(nn.Module):
    def __init__(self):
        super(CoupledGenerators, self).__init__()

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.Share = nn.Sequential(
            *block(100, 256),
            *block(256, 512),
        )
        self.G1 = nn.Sequential(
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
        self.G2 = nn.Sequential(
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )
    def forward(self, z):
        img1 = self.G1(self.Share(z))
        img1 = img1.view(img1.size(0), *img_shape)
        img2 = self.G2(self.Share(z))
        img2 = img2.view(img2.size(0), *img_shape)
        return img1, img2



class CoupledDiscriminators(nn.Module):
    def __init__(self):
        super(CoupledDiscriminators, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.Share = nn.Sequential(
            *block(int(np.prod(img_shape)), 512),
            *block(512, 256),
        )

        self.D1 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.D2 = nn.Sequential(
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img1, img2):
        img_flat_1 = img1.view(img1.size(0), -1)
        img_flat_2 = img2.view(img2.size(0), -1)
        validity1 = self.D1(self.Share(img_flat_1))
        validity2 = self.D2(self.Share(img_flat_2))
        return validity1, validity2


# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5), (0.5)),
])

import os
import gzip

def load_data(data_folder, data_name):

    with gzip.open(os.path.join(data_folder, data_name), 'rb') as imgpath:
        x = np.frombuffer(imgpath.read(), np.uint8, offset = 16).reshape(60000, 28, 28)
    return x

def load_invert_mnist(data_folder, data_name):
    trX = load_data(data_folder, data_name)
    invert_trX = 255 - trX
    invert_trX = invert_trX.astype(np.uint8)
    return invert_trX

# 将数据集转换为Tensor的类
class DealDataset(Data.Dataset):
    def __init__(self, folder, data_name, transform = None):
        x = load_data(folder, data_name)
        self.x = x
        self.transform = transform

    def __getitem__(self, index):
        img = self.x[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.x)

class DealDatasetInvert(Data.Dataset):
    def __init__(self, folder, data_name, transform = None):
        x = load_invert_mnist(folder, data_name)
        self.x = x
        self.transform = transform

    def __getitem__(self, index):
        img = self.x[index]

        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.x)

# 设置文件路径
data_folder = "./data/mnist"
train_data_name = "train-images-idx3-ubyte.gz"


# 实例化这个类，然后我们就得到了Dataset类型的数据
trainDataset = DealDataset(data_folder, train_data_name, img_transform)
trainDatasetInvert = DealDatasetInvert(data_folder, train_data_name, img_transform)
# 数据的装载
train_loader = Data.DataLoader(dataset = trainDataset, batch_size = opt.batch_size, shuffle = True)
train_loader_invert = Data.DataLoader(dataset = trainDatasetInvert, batch_size = opt.batch_size, shuffle = True)

# Initialize models
coupled_generators = CoupledGenerators().to(device)
coupled_discriminators = CoupledDiscriminators().to(device)

# Loss function
adversarial_loss = torch.nn.MSELoss()

# Optimizers
optimizer_G = torch.optim.Adam(coupled_generators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(coupled_discriminators.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

for epoch in range(opt.n_epochs):
    for i, (imgs1, imgs2) in enumerate(zip(train_loader, train_loader_invert)):
        batch_size = imgs1.shape[0]
        valid = torch.ones(batch_size, 1).to(device)
        fake = torch.zeros(batch_size, 1).to(device)

        imgs1 = imgs1.to(device)
        imgs2 = imgs2.to(device)

        optimizer_G.zero_grad()

        z = torch.randn(batch_size, opt.latent_dim).to(device)

        gen_imgs1, gen_imgs2 = coupled_generators(z)

        validity1, validity2 = coupled_discriminators(gen_imgs1, gen_imgs2)

        g_loss = (adversarial_loss(validity1, valid) + adversarial_loss(validity2, valid)) / 2

        g_loss.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        
        validity1_real, validity2_real = coupled_discriminators(imgs1, imgs2)
        validity1_fake, validity2_fake = coupled_discriminators(gen_imgs1.detach(), gen_imgs2.detach())

        d_loss = (
            adversarial_loss(validity1_real, valid)
            + adversarial_loss(validity1_fake, fake)
            + adversarial_loss(validity2_real, valid)
            + adversarial_loss(validity2_fake, fake)
        ) / 4

        d_loss.backward()
        optimizer_D.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch, opt.n_epochs, i, len(train_loader), d_loss.item(), g_loss.item())
        )

    gen_imgs = torch.cat((gen_imgs1.data, gen_imgs2.data), 0)
    save_image(gen_imgs, './imgs/gen_imgs{}.png'.format(epoch))