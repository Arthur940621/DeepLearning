import torch
import torch.nn as nn
from torchvision.utils import save_image
import torch.optim as optim
from torchvision import datasets, transforms
import os

img_shape = (1, 28, 28)
os.makedirs("imgs", exist_ok=True)

class SpectralNorm(nn.Module):
    def __init__(self, layer, name = 'weight', power_iterations = 1):
        super(SpectralNorm, self).__init__()
        '''params:
        layer: 传入的需要使得参数谱归一化的网路层
        name : 谱归一化的参数
        power_iterations：幂迭代的次数,论文中提到，实际上迭代一次已足够
        '''
        self.layer = layer
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params(): # 如果迭代参数未初始化，则初始化
            self._make_params()
            
    def _update_u_v(self):
        u = getattr(self.layer, self.name+'_u')
        v = getattr(self.layer, self.name+'_v')
        w = getattr(self.layer, self.name+'_bar')
        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = self.l2Norm(torch.mv(torch.t(w.view(height, -1).data), u.data)) # 计算：v <- (W^t*u)/||W^t*u||   2范数
            u.data = self.l2Norm(torch.mv(w.view(height, -1).data, v.data)) # 计算：u <- (Wv)/||Wv||
        sigma = u.dot(w.view(height, -1).mv(v)) # 计算 W的谱范数 ≈ u^t * W * v
        setattr(self.layer, self.name, w/sigma.expand_as(w))
        
    def _made_params(self):
        # 存在这些参数则返回True, 否则返回False
        try:
            u = getattr(self.layer, self.name + '_u')
            v = getattr(self.layer, self.name + '_v')
            w = getattr(self.layer, self.name + '_bar')
            return True
        except AttributeError:
            return False
    def _make_params(self):
        w = getattr(self.layer, self.name)
        height = w.data.shape[0] # 输出的卷积核的数目
        width = w.view(height, -1).data.shape[1] # width为 in_feature*kernel*kernel 的值
        # .new()创建一个新的Tensor，该Tensor的type和device都和原有Tensor一致
        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad = False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad = False)
        u.data = self.l2Norm(u.data)
        v.data = self.l2Norm(v.data)
        w_bar = nn.Parameter(w.data)
        del self.layer._parameters[self.name] # 删除以前的weight参数
        # 注册参数
        self.layer.register_parameter(self.name+'_u', u) # 传入的值u，v必须是Parameter类型
        self.layer.register_parameter(self.name+'_v', v)
        self.layer.register_parameter(self.name+'_bar', w_bar)
        
    def l2Norm(self, v, eps = 1e-12): # 用于计算例如：v/||v||
        return v/(v.norm() + eps) 
    
    def forward(self, *args):
        self._update_u_v()
        return self.layer.forward(*args)


class Self_Attn(nn.Module): # 自注意力机制
    def __init__(self,in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
    
        self.query_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.key_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim//8, kernel_size= 1)
        self.value_conv = nn.Conv2d(in_channels = in_dim, out_channels = in_dim, kernel_size= 1)
        self.gamma = nn.Parameter(torch.zeros(1)) # gamma初值为0
        self.softmax = nn.Softmax(dim=-1) # 对每一行进行softmax
    
    def forward(self,x):
        batchsize, C, width, height = x.size() # x=[B,C,W,H]
        proj_query = self.query_conv(x).view(batchsize, -1, width*height).permute(0, 2, 1) # proj_query=[B,N,C^]
        proj_key = self.key_conv(x).view(batchsize, -1, width*height) # proj_key=[B,C^,N]
        energy = torch.bmm(proj_query, proj_key) # 矩阵乘法
        attention = self.softmax(energy) # attention=[B,N,N]
        proj_value = self.value_conv(x).view(batchsize, -1, width*height) # proj_value=[B,C,N]
        out = torch.bmm(proj_value, attention.permute(0, 2, 1)) # out=[B,C,N]
        out = out.view(batchsize, C, width, height) # out=[B,C,W,H]
        
        out = self.gamma * out + x
        return out, attention


class Generator(nn.Module): # 生成器
    # ngf：生成器特征图的深度
    # nz：生成器的输入z的维度
    def __init__(self, ngf, nz): 
        super(Generator, self).__init__()
        # layer1输入的是一个100x1x1的随机噪声, 输出尺寸(ngf*8)x4x4，其中4=(1-1)*1-2*0+4
        self.layer1 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(nz, ngf * 8, kernel_size=4, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(ngf * 8), # 对该批次每个通道分别进行归一化
            nn.ReLU(inplace=True)
        )
        # layer2输出尺寸(ngf*4)x8x8，8=(4-1)*2-2*1+4
        self.layer2 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(inplace=True)
        )
        # layer3输出尺寸(ngf*2)x16x16，16=(8-1)*2-2*1+4
        self.layer3 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(inplace=True)
        )
        # layer4输出尺寸(ngf)x32x32，32=(16-1)*2-2*1+4
        self.layer4 = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.BatchNorm2d(ngf),
            nn.ReLU(inplace=True)
        )
        # layer5输出尺寸3*64*64，64=(32-1)*2-2*1+4
        self.layer5 = nn.Sequential(
            nn.ConvTranspose2d(ngf, 3, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        self.attn1 = Self_Attn(128)
        self.attn2 = Self_Attn(64)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, p1 = self.attn1(out)
        out = self.layer4(out)
        out, p2 = self.attn2(out)
        out = self.layer5(out)
        return out, p1, p2


class Discriminator(nn.Module): # 判别器
    # ndf：判别器特征图的深度
    def __init__(self, ndf):
        super(Discriminator, self).__init__()
        # layer1输入3*64*64, 输出尺寸ndf*32*32，其中32=(64-4+2*1)/2+1
        self.layer1 = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, ndf, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer2输出尺寸(ngf*2)x16x16，16=(32-4+2*1)/2+1
        self.layer2 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf, ndf * 2, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer3输出尺寸(ngf*4)x8x8，8=(16-4+2*1)/2+1
        self.layer3 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 2, ndf * 4, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer4输出尺寸(ngf*8)x4x4，4=(8-4+2*1)/2+1
        self.layer4 = nn.Sequential(
            SpectralNorm(nn.Conv2d(ndf * 4, ndf * 8, kernel_size=4, stride=2, padding=1, bias=False)),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # layer5输出一个数，1=(4-4+2*0)/1+1
        self.layer5 = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
        )

        self.attn1 = Self_Attn(256)
        self.attn2 = Self_Attn(512)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out, p1 = self.attn1(out)
        out = self.layer4(out)
        out, p2 = self.attn2(out)
        out = self.layer5(out)
        out = out.view(-1)
        return out, p1, p2

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 读入图像与预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
path = "./data/mnist"
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

# 优化器
optimizerG = optim.Adam(NetG.parameters(), lr = 0.0001, betas = (0.5, 0.999))
optimizerD = optim.Adam(NetD.parameters(), lr = 0.0004, betas = (0.5, 0.999))

for epoch in range(1, 501):
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(device)
        # 固定生成器G，训练鉴别器D
        optimizerD.zero_grad()
        # 让D尽可能把真图片判别为1
        d_out_real, dr1, dr2 = NetD(imgs)
        real_label = torch.ones(64).to(device)
        lossD_real = torch.nn.ReLU()(1.0 - d_out_real).mean()
    
        # 让D尽可能把假图片判别为0
        noise = torch.randn(64, 100, 1, 1)
        noise = noise.to(device)
        fake_label = torch.zeros(64).to(device)
        fake_images, gf1, gf2 = NetG(noise) # 生成假图
        d_out_fake, df1, df2 = NetD(fake_images.detach()) # 避免梯度传到G，G不需要更新
        lossD_fake = torch.nn.ReLU()(1.0 + d_out_fake).mean()
        
        lossD = lossD_real + lossD_fake    
        lossD.backward()
        optimizerD.step()

        # 固定鉴别器D，训练生成器G
        optimizerG.zero_grad()
        # 让D尽可能把G生成的假图判别为1
        fake_images, gf1, gf2 = NetG(noise) # 生成假图
        g_out_fake, _, _ = NetD(fake_images)
        lossG = -g_out_fake.mean()
        lossG.backward()
        optimizerG.step()

        print('[%d/%d][%d/%d] Loss_D: %.3f Loss_G %.3f'
              % (epoch, 500, i, len(dataloader), lossD.item(), lossG.item()))
    save_image(fake_images, './imgs/fake_images-{}.png'.format(epoch))

torch.save(NetG.state_dict(), './generator.pth')
torch.save(NetD.state_dict(), './discriminator.pth')