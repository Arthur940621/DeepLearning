
import torch
import torchvision.utils as vutils
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

# 定义生成器
class Generator(nn.Module):
    # ngf：生成器特征图的深度
    # nz：生成器的输入z的维度
    def __init__(self, ngf, nz): 
        super(Generator, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4，其中4=(1-1)*1-2*0+4
        self.layer1 = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(ngf * 8), # 对该批次每个通道分别进行归一化
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸(ngf*4)x8x8，8=(4-1)*2-2*1+4
        self.layer2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(ngf*2)x16x16，16=(8-1)*2-2*1+4
        self.layer3 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(ngf)x32x32，32=(16-1)*2-2*1+4
        self.layer4 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸3*64*64，64=(32-1)*2-2*1+4
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

# 定义鉴别器
class Discriminator(nn.Module):
    # ndf：判别器特征图的深度
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        # layer1输入3*64*64, 输出尺寸ndf*32*32，其中32=(64-4+2*1)/2+1
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2输出尺寸(ngf*2)x16x16，16=(32-4+2*1)/2+1
        self.layer2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3输出尺寸(ngf*4)x8x8，8=(16-4+2*1)/2+1
        self.layer3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4输出尺寸(ngf*8)x4x4，4=(8-4+2*1)/2+1
        self.layer4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5输出一个数，表示概率，1=(4-4+2*0)/1+1
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        return out

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读入图像与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
path = "./faces"
dataset = datasets.ImageFolder(path, transform)
dataloader = torch.utils.data.DataLoader(
    dataset = dataset,
    batch_size = 64,
    shuffle = True,
    drop_last = True
)

# 模型实例化
NetG = Generator(64, 100).to(device)
NetD = Discriminator(64).to(device)

# 损失函数
loss_fn = nn.BCELoss()

# 优化器
optimizerG = optim.Adam(NetG.parameters(), lr = 0.002, betas = (0.5, 0.999))
optimizerD = optim.Adam(NetD.parameters(), lr = 0.002, betas = (0.5, 0.999))

class Heuristic:
    def __init__(self):
        self.ls = []

    def get_heur_inf_dec(self, disc_out):
        h = torch.sign(disc_out).sum() / len(disc_out)
        self.ls.append(h)
        if h > 0:
            return True
        return False





for epoch in range(1, 501):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        # 让D尽可能把真图片判别为1
        output = NetD(imgs).view(-1)
        real_label = torch.ones(64).to(device)
        lossD_real = loss_fn(output, real_label)
        lossD_real.backward()
        # 让D尽可能把假图片判别为0
        noise = torch.randn(64, 100, 1, 1)
        noise = noise.to(device)
        fake_label = torch.zeros(64).to(device)
        fake = NetG(noise) # 生成假图
        output = NetD(fake.detach()).view(-1) # 避免梯度传到G，G不需要更新
        lossD_fake = loss_fn(output, fake_label)
        lossD_fake.backward()
        lossD = lossD_real + lossD_fake      
        optimizerD.step()
        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        output = NetD(fake).view(-1)
        lossG = loss_fn(output, real_label)
        lossG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, 500, i, len(dataloader), lossD.item(), lossG.item()))
    vutils.save_image(fake.data, './img/fake_images-{}.png'.format(epoch), normalize = True)

torch.save(NetG.state_dict(), "NetG.plk")
torch.save(NetD.state_dict(), "NetD.plk")
