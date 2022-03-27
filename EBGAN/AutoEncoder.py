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

        self.init_size = 32 // 4
        self.l1 = nn.Sequential(nn.Linear(100, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
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

    def forward(self, noise):
        out = self.l1(noise)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.down = nn.Sequential(nn.Conv2d(1, 64, 3, 2, 1), nn.ReLU())
        self.down_size = 32 // 2
        down_dim = 64 * (32 // 2) ** 2

        self.embedding = nn.Linear(down_dim, 32)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(32, 0.8),
            nn.ReLU(inplace=True),
            nn.Linear(32, down_dim),
            nn.BatchNorm1d(down_dim),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Sequential(nn.Upsample(scale_factor=2), nn.Conv2d(64, 1, 3, 1, 1))

    def forward(self, img):
        out = self.down(img) # [B, 64, 16, 16]
        embedding = self.embedding(out.view(out.size(0), -1)) # [B, 32])
        out = self.fc(embedding) # [B, 16384]
        out = self.up(out.view(out.size(0), 64, self.down_size, self.down_size)) # [B, 1, 32, 32]
        return out, embedding

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

pixelwise_loss = nn.MSELoss()
def pullaway_loss(embeddings):
    norm = torch.sqrt(torch.sum(embeddings ** 2, -1, keepdim=True))
    normalized_emb = embeddings / norm
    similarity = torch.matmul(normalized_emb, normalized_emb.transpose(1, 0))
    batch_size = embeddings.size(0)
    loss_pt = (torch.sum(similarity) - batch_size) / (batch_size * (batch_size - 1))
    return loss_pt

lambda_pt = 0.1
margin = 1.0

for epoch in range(1, 31):
    for i, (img, _) in enumerate(train_loader):
        img_num = img.shape[0]
        real_img = img.to(device)

        d_optimizer.zero_grad()
        real_recon, _ = D(real_img)
        z = torch.randn(img_num, 100).to(device)
        fake_img = G(z).detach()
        fake_recon, _ = D(fake_img)
        d_loss_real = pixelwise_loss(real_recon, real_img)
        d_loss_fake = pixelwise_loss(fake_recon, fake_img)
        
        d_loss = d_loss_real
        if (margin - d_loss_fake.data).item() > 0:
            d_loss += (margin - d_loss_fake)

        d_loss.backward()
        d_optimizer.step()

        g_optimizer.zero_grad()
        z = torch.randn(img_num, 100).to(device)
        gen_imgs = G(z)
        recon_imgs, img_embeddings = D(gen_imgs)
        g_loss = pixelwise_loss(recon_imgs, gen_imgs.detach()) + lambda_pt * pullaway_loss(img_embeddings)
        g_loss.backward()
        g_optimizer.step()


        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, 30, i, len(train_loader), d_loss.item(), g_loss.item()))

    save_image(gen_imgs, './imgs/fake_images-{}.png'.format(epoch))

torch.save(G.state_dict(), './generator.pth')
torch.save(D.state_dict(), './discriminator.pth')