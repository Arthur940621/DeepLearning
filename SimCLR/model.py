import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import batchnorm
from torchvision.models.resnet import resnet50

class Model(nn.Module):
    def __init__(self, feature_dim = 128):
        super(Model, self).__init__()
        # 修改resnet50作为f
        self.f = []
        for name, module in resnet50().named_children():
            # name=conv1,bn1,relu,maxpool,layer1,layer2,layer3,layer4,avgpool,fc
            if name == 'conv1':
            # 原conv1为Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # 现在output=(input-3+2*1)/1+1=input
                module = nn.Conv2d(3, 64, kernel_size = 3, stride = 1, padding = 1, bias = False)
            # 去掉线性层和maxpool层
            if not isinstance(module, nn.Linear) and not isinstance(module, nn.MaxPool2d):
                self.f.append(module)
        # 基础编码器f
        self.f = nn.Sequential(*self.f)
        # 投影头g
        self.g = nn.Sequential(
            nn.Linear(2048, 512, bias = False),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.Linear(512, feature_dim, bias = True)
        )
    
    def forward(self, x):
        x = self.f(x) # [b,3,64,64]=>[b,2048,1,1]
        feature = torch.flatten(x, start_dim = 1) # [b,2048,1,1]=>[b,2048]
        out = self.g(feature) # [b,2048]=>[b,feature_dim]
        return F.normalize(feature, dim = -1), F.normalize(out, dim = -1)