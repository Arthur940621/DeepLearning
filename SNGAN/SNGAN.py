import torch.autograd
import torch.nn as nn
from torchvision import transforms
import torch.utils.data as Data
import load_minist_data
from torchvision.utils import save_image
import numpy as np
import os

os.makedirs("imgs", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_shape = (1, 28, 28)

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
            SpectralNorm(nn.Linear(int(np.prod(img_shape)), 512)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(512, 256)),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Linear(256, 1)),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
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

# 优化器
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.00005)
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.00005)



for epoch in range(1, 31):
    for i, (img, _) in enumerate(train_loader):
        img_num = img.shape[0]
        real_img = img.to(device)
        d_optimizer.zero_grad()

        z = torch.randn(img_num, 100).to(device)
        fake_img = G(z).detach()

        d_loss = -torch.mean(D(real_img)) + torch.mean(D(fake_img))
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

