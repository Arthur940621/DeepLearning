import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.modules import padding
from torch.nn.modules.module import Module

# 卷积模板,卷积与ReLu共同使用
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.relu = nn.ReLU(inplace = True) # 即对原值进行操作，然后将得到的值又直接复制到该值中
    
    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x

# 定义Inception
class Inception(nn.Module):
    """
    ch1x1:第一个分支的1x1的卷积的channels
    ch3x3red:第二个分支1x1卷积channels
    ch3x3:第二个分支3x3的卷积channels
    ch5x5red:第三个分支1x1的卷积channels
    ch5x5:第三个分支5x5的卷积channels
    poll_proj:第四个分支1x1的卷积channels
    """
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, poll_proj):
        super(Inception, self).__init__()

        self.brach1 = BasicConv2d(in_channels, ch1x1, kernel_size = 1)
        self.brach2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size = 1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size = 3, padding = 1) # 保证输出矩阵大小相同
        )
        self.brach3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size = 1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size = 5, padding = 2) # 保证输出矩阵大小相同
        )
        self.brach4 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 1, padding = 1), # 保证输出矩阵大小相同
            BasicConv2d(in_channels, poll_proj, kernel_size = 1)
        )

    def forward(self, x):
        branch1 = self.brach1(x)
        branch2 = self.brach2(x)
        branch3 = self.brach3(x)
        branch4 = self.brach4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1) # 在channels上进行合并

# 定义辅助分类器
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.averagePoll = nn.AvgPool2d(kernel_size = 5, stride = 3)
        self.conv = BasicConv2d(in_channels, 128, kernel_size = 1) # output[b, 128, 4, 4]
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        # aux1:[b, 512, 14, 14] aux2:[b, 528, 14, 14]
        x = self.averagePoll(x)
        # aux1:[b, 512, 4, 4] aux2:[b, 528, 4, 4]
        x = self.conv(x)
        # aux1:[b, 128, 4, 4] aux2:[b, 128, 4, 4]
        x = torch.flatten(x, 1) # 在channels上进行展平
        """
        当我们实例化一个模型model后，可以通过mode.train()和model.eval()来控制模型的状态
        在model.train()模式下self.training=True
        在model.eval()模式下self.traning=False
        """
        x = F.dropout(x, 0.5, training = self.training)
        # aux1:[b, 2048] aux2:[b, 2048]
        x = F.relu(self.fc1(x), inplace = True)
        # aux1:[b, 1024] aux2:[b, 1024]
        x = self.fc2(x)
        # aux1:[b, num_classes] aux2:[b, num_classes]
        return x


# 定义GoogLeNet
class GoogLeNet(nn.Module):
    def __init__(self, num_classes, aux_logits = True, init_weights = False):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits

        self.conv1 = BasicConv2d(3, 64, kernel_size = 7, stride = 2, padding = 3)
        self.maxpool1 = nn.MaxPool2d(3, stride = 2, ceil_mode = True) # ceil_mode，默认为False（地板模式），为True时是天花板模式

        self.conv2 = BasicConv2d(64, 64, kernel_size = 1)
        self.conv3 = BasicConv2d(64, 192, kernel_size = 3, padding = 1)
        self.maxpool2 = nn.MaxPool2d(3, stride = 2, ceil_mode = True)

        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride = 2, ceil_mode = True)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(3, stride = 2, ceil_mode = True)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) # 输出大小为1024@1x1
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)
        if init_weights:
            self._initialize_weights()
    
    def forward(self, x):
        # [b, 3, 224, 224]
        x = self.conv1(x)
        # [b, 64, 112, 112]:输入3层，输出64层，112=(224-7+2*3)/2+1
        x = self.maxpool1(x)
        # [b, 64, 56, 56]:输入64层，输出64层，56=(112+1-3)/2+1
        
        x = self.conv2(x)
        # [b, 64, 56, 56]:输入64层，输出64层，卷积核为1
        x = self.conv3(x)
        # [b, 192, 56, 56]:输入64层，输出192层，56=(56-3+2*1)/1+1
        x = self.maxpool2(x)
        # [b, 192, 28, 28]:输入192层，输出192层，28=(56+1-3)/2+1
        
        x = self.inception3a(x)
        # [b, 256, 28, 28]
        x = self.inception3b(x)
        # [b, 480, 28, 28]
        x = self.maxpool3(x)
        # [b, 480, 14, 14]:输入480层，输出480层，14=(28+1-3)/2+1

        x = self.inception4a(x)
        # [b, 512, 14, 14]
        if self.training and self.aux_logits:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        # [b, 512, 14, 14]
        x = self.inception4c(x)
        # [b, 512, 14, 14]
        x = self.inception4d(x)
        # [b, 528, 14, 14]
        if self.training and self.aux_logits:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        # [b, 832, 14, 14]
        x = self.maxpool4(x)
        # [b, 832, 7, 7]:输入832层，输出832层，7=(14+1-3)/2+1

        x = self.inception5a(x)
        # [b, 832, 7, 7]
        x = self.inception5b(x)
        # [b, 1024, 7, 7]
        x = self.avgpool(x)
        # [b, 1024, 1, 1]

        x = torch.flatten(x, 1)
        # [b, 1024]
        x = self.fc(x)
        # [b, num_classes]

        if self.training and self.aux_logits:
            return x, aux2, aux1
        return x

    # 定义模型权重初始化函数
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = 'fan_out', nonlinearity = 'relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)