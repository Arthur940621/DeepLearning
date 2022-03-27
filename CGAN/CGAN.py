import torch
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import load_minist_data
from torchvision.utils import save_image
import os

os.makedirs("imgs", exist_ok=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(10, 10)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(100 + 10, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, 1 * 32 * 32),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), 1, 32, 32)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(10, 10)

        self.model = nn.Sequential(
            nn.Linear(10 + 1 * 32 * 32, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, img, labels):
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
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

d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002, betas = (0.5, 0.999))
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002, betas = (0.5, 0.999))

loss = nn.MSELoss()

import numpy as np
def sample_image(n_row, epoch):
    z = torch.randn(100, 100).to(device)
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = torch.LongTensor(labels).to(device)
    gen_imgs = G(z, labels)
    save_image(gen_imgs, "imgs/%d.png" % epoch, nrow=n_row, normalize=True)


for epoch in range(1, 31):
    for i, (img, label) in enumerate(train_loader):
        img_num = img.shape[0]

        valid = torch.ones(img_num, 1).to(device)
        fake = torch.zeros(img_num, 1).to(device)

        real_img = img.to(device)
        real_label = label.to(device)
        
        d_optimizer.zero_grad() 

        real_out = D(real_img, real_label)
        real_loss = loss(real_out, valid)

        z = torch.randn(img_num, 100).to(device)
        fake_label = torch.randint(1, 10, [img_num]).to(device)

        fake_img = G(z, fake_label).detach()
        fake_out = D(fake_img, fake_label)
        fake_loss = loss(fake_out, fake)
        d_loss = real_loss + fake_loss

        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()

        z = torch.randn(img_num, 100).to(device)
        fake_label = torch.randint(1, 10, [img_num]).to(device)
        fake_img = G(z, fake_label)
        out = D(fake_img, fake_label)

        g_loss = loss(out, valid)
        g_loss.backward()
        g_optimizer.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, 30, i, len(train_loader), d_loss.item(), g_loss.item()))
        
    sample_image(n_row=10, epoch=epoch)

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')