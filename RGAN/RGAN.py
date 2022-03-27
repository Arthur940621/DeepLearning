import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import load_minist_data
from torchvision.utils import save_image
import numpy as np
import os

os.makedirs("imgs", exist_ok=True)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 1, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def discriminator_block(in_filters, out_filters, bn=True):
            block = [nn.Conv2d(in_filters, out_filters, 3, 2, 1), nn.LeakyReLU(0.2, inplace=True), nn.Dropout2d(0.25)]
            if bn:
                block.append(nn.BatchNorm2d(out_filters, 0.8))
            return block

        self.model = nn.Sequential(
            *discriminator_block(1, 16, bn=False),
            *discriminator_block(16, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
        )

        ds_size = 32 // 2 ** 4
        self.adv_layer = nn.Sequential(nn.Linear(128 * ds_size ** 2, 1))

    def forward(self, img):
        out = self.model(img)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)

        return validity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

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

data_folder = "./data/mnist"
train_data_name = "train-images-idx3-ubyte.gz"
train_label_name = "train-labels-idx1-ubyte.gz"

trainDataset = DealDataset(data_folder, train_data_name, train_label_name, img_transform)

train_loader = Data.DataLoader(dataset = trainDataset, batch_size = 64, shuffle = True)

D = Discriminator().to(device)
G = Generator().to(device)

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0003)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0003)

loss = nn.BCEWithLogitsLoss().to(device)
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for epoch in range(1, 31):
    for i, (img, _) in enumerate(train_loader):
        img_num = img.shape[0]
        real_img = img.to(device)
        real_label = torch.ones(img_num, 1).to(device)
        fake_label = torch.zeros(img_num, 1).to(device)
        d_optimizer.zero_grad() 

        real_out = D(real_img)
        z = torch.randn(img_num, 100).to(device)
        fake_img = G(z).detach()
        fake_out = D(fake_img)
        real_loss = loss(real_out - fake_out, real_label)
        fake_loss = loss(fake_out - real_out, fake_label)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        z = torch.randn(img_num, 100).to(device)
        g_optimizer.zero_grad()
        fake_img = G(z)
        real_out = D(real_img)
        fake_out = D(fake_img)
        g_loss = loss(fake_out - real_out, real_label)
        g_loss.backward()
        g_optimizer.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, 30, i, len(train_loader), d_loss.item(), g_loss.item()))

    save_image(fake_img, './imgs/fake_images-{}.png'.format(epoch))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')