import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary

class_num = 15

def _SplitChannels(channels, kernel_num):
    split_channels = [channels//kernel_num for _ in range(kernel_num)]
    split_channels[0] += channels - sum(split_channels)
    return split_channels

class MFC(nn.Module):
    def __init__(self, channels, spectral, kernel_size, kernel_num):
        super(MFC, self).__init__()
        self.channels = channels
        self.kernel_num = kernel_num
        self.k1 = kernel_size[0]
        self.k2 = kernel_size[1]
        self.k3 = kernel_size[2]
        self.sp = _SplitChannels(channels, self.kernel_num)
        self.conv = nn.ModuleList()
        self.conv_f = nn.ModuleList()
        for i in range(self.kernel_num):
            n = i+1
            self.conv_f.append(nn.Conv2d((self.sp[0] + self.sp[i] * i) * spectral,
                                         (self.sp[0] + self.sp[i] * i) * spectral, 1, 1))
            self.conv.append(nn.Conv3d(self.sp[0] + self.sp[i] * i, self.sp[0] + self.sp[i] * i,
                                       (self.k1, self.k2, self.k3), 1, ((self.k1-1)//2, (self.k2-1)//2, (self.k3-1)//2),
                                       groups=self.sp[0] + self.sp[i] * i, bias=False))

    def forward(self, x):
        if self.kernel_num == 1:
            return self.conv[0](x)
        x_split = torch.split(x, self.sp, dim=1)
        xs = []
        x = self.conv[0](x_split[0])
        xs.append(x)
        for i in range(1, self.kernel_num):
            x = torch.cat([x_split[i], xs[i - 1]], dim=1)
            # Feature Fusion Block
            x1 = x.view(x.shape[0], x.shape[1] * x.shape[2], x.shape[3], x.shape[4])
            x1 = self.conv_f[i](x1)
            x = x1.view(x.shape[0], -1, x.shape[2], x.shape[3], x.shape[4])
            x = self.conv[i](x)
            xs.append(x)
            x = torch.cat(xs, dim=1)
        return x

class ms2fenet(nn.Module):
    def __init__(self):
        super(ms2fenet, self).__init__()
        self.conv1 = nn.Conv3d(1, 8, (7, 3, 3))
        self.bn1 = nn.BatchNorm3d(8)
        self.pool1 = nn.AvgPool3d((2, 3, 3), stride=2)
        self.conv2 = MFC(8, 24, (5, 3, 3), 4)
        self.bn2 = nn.BatchNorm3d(20)
        self.pool2 = nn.AvgPool3d((2, 3, 3), stride=2)
        self.conv3 = MFC(20, 12, (5, 3, 3), 4)
        self.bn3 = nn.BatchNorm3d(50)
        self.pool3 = nn.AvgPool3d((2, 3, 3), stride=2)
        self.conv4 = MFC(50, 6, (3, 3, 3), 4)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.AdaptiveAvgPool3d((3, 2, 2))
        self.linear = nn.Sequential(
            nn.Linear(1536, class_num),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ms2fenet().to(device)
summary(model, input_size=(1, 30, 15, 15))

from thop import profile
model = ms2fenet()
input = torch.randn(1, 1, 30, 15, 15)
flops, params = profile(model, inputs=(input, ))
print(flops/1e6, params)