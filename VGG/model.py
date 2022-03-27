import torch
import torch.nn as nn

# 卷积模板,卷积与ReLu共同使用
def BasicConv2d(in_channels,out_channels):
    
    basicconv2d=nn.Sequential(
        
        nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )
    return basicconv2d


class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        self.layer1 = self.make_layer(3, 64, 2)
        self.layer2 = self.make_layer(64, 128, 2)
        self.layer3 = self.make_layer(128, 256, 3)
        self.layer4 = self.make_layer(256, 512, 3)
        self.layer5 = self.make_layer(512, 512, 3)
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7 * 7 * 512, 4096),
            nn.ReLU(inplace = True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace = True),
            nn.Linear(4096, num_classes)
        )
    def make_layer(self, in_ch, out_ch, block_num):
        layer = []
        layer.append(BasicConv2d(in_ch, out_ch))
        for i in range(1, block_num):
            layer.append(BasicConv2d(out_ch, out_ch))
        layer.append(nn.MaxPool2d(kernel_size = 2, stride = 2))
        return nn.Sequential(*layer)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out
