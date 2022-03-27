import torch
import torch.nn as nn
from torch.nn import functional as F

class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride = 1, downsample = None): # 下采样，即残差结构中的虚线部分
        super(BasicBlock, self).__init__()
        # 卷积核大小为3，步长为1，padding为1，卷积不改变输入大小，output=(input—3+2*1)/1+1=input
        # 卷积核大小为3，步长为2，padding为1，output=(input—3+2*1)/2+1=input/2+0.5=input/2
        # 第一层卷积步长可能为1，可以能为2
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace = True)
        # 第二层卷积步长一定为1
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out



class ResNet34(nn.Module):
    def __init__(self, num_classes = 1000):
        super(ResNet34, self).__init__()
       
        # 对输入图片前置处理
        self.pre = nn.Sequential(
            # [b,3,224,224]=>[b,64,112,112]，其中112.5=(224-7+2*3)/2+1
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            # [b,64,112,112] =>[b,64,56,56]，其中56.5=(112-3+2*1)/2+1
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        )
        # 重复的layer，分别有3,4,6,3个residual block
        # 56x56x64，layer1层输入输出一样，make_layer里，不用进行downsample
        self.layer1 = self.make_layer(64, 64, 3)
        # 第一个stride=2，剩下3个stride=1，28x28x128
        self.layer2 = self.make_layer(64, 128, 4, stride = 2)
        # 14x14x256
        self.layer3 = self.make_layer(128, 256, 6, stride = 2)
        # 7x7x512
        self.layer4 = self.make_layer(256, 512, 3, stride = 2)
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, in_ch, out_ch, block_num, stride = 1): # block_num代表该层包含多少个残差结构
        downsample = None
        if stride != 1: #第一层不需要downsample
            downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        layers = []
        layers.append(BasicBlock(in_ch, out_ch, stride, downsample))
        #后面的几个BasicBlock，不需要downsample
        for i in range(1, block_num):
            layers.append(BasicBlock(out_ch, out_ch))
        return nn.Sequential(*layers)

            
    def forward(self, x):
        out = self.pre(x) # [b,3,224,224]=>[b,64,56,56]
        out = self.layer1(out) # [b,64,56,56]=>[b,64,56,56]
        out = self.layer2(out) # [b,64,56,56]=>[b,28,28,128]
        out = self.layer3(out) # [b,28,28,128]=>[b,14,14,256]
        out = self.layer4(out) # [b,14,14,256]=>[b,7,7,512]
        out = F.avg_pool2d(out, 7) # [b,7,7,512]=>[b,1,1,512]
        out = out.view(out.size(0), -1) # 将输出拉伸为一行：[b,1,1,512]=>[b,512]
        out = self.fc(out) # [b,512]=>[b,num_class]
        return out
