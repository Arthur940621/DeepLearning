import torch
import torch.nn as nn

def _make_divisible(ch, divisor=8, min_ch=None): # ch为divisor的整数倍
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch

class ConvBNReLu(nn.Sequential):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, groups=1): # DW卷积也调用nn.Conv2d()，groups=1代表普通卷积，groups=in_channel代表DW卷积
        padding = (kernel_size - 1) // 2 # 如果kernel_size=3，padding=1，如果kernel_size=1，padding=0
        super(ConvBNReLu, self).__init__(
            nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU6(inplace=True)
        )

# 倒残差结构
class InvertedResidual(nn.Module):
    def __init__(self, in_channel, out_channel, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        hidden_channel = in_channel * expand_ratio # output=tk
        self.use_shotcut = stride == 1 and in_channel == out_channel # 判断是否使用shortcut

        layers = []
        if expand_ratio != 1:
            # 1x1pw卷积
            layers.append(ConvBNReLu(in_channel, hidden_channel, kernel_size=1))
        layers.extend([
            # 3x3dw卷积
            ConvBNReLu(hidden_channel, hidden_channel, stride=stride, groups=hidden_channel),
            # 1x1pw卷积(linear)
            nn.Conv2d(hidden_channel, out_channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channel)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_shotcut:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, alpha=1.0, round_nearest=8):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = _make_divisible(32 * alpha, round_nearest) # 将input_channel调整为最接近32*alpha的round_nearest的整数倍
        last_channel = _make_divisible(1280 * alpha, round_nearest)

        inverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        features = []
        # conv1 layer
        features.append(ConvBNReLu(3, input_channel, stride=2))
        # inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * alpha, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # last several layers
        features.append(ConvBNReLu(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        # classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(last_channel, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

model = MobileNetV2()

print(model)