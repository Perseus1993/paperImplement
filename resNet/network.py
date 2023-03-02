import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self, ch_in, ch_out):
        # （size -3 +  2 *padding）/stride + 1 = size ,因此padding = 1
        super(BasicBlock, self).__init__()
        # 这里还是分两个conv，即使一样
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=3,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3,
                               padding=1, bias=False)

        self.block = nn.Sequential(
            self.conv1(),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(),
            self.conv2(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return nn.ReLU(self.block(x) + x)


class Bottleneck(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Bottleneck, self).__init__()
        # 1*1卷积
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=1,
                               stride=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=ch_in, out_channels=ch_in, kernel_size=3,
                               padding=1, stride=1, bias=False)
        self.conv3 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=1,
                               stride=1, bias=False)
        self.block = nn.Sequential(
            self.conv1(),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            self.conv2(),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(),
            self.conv3(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return nn.ReLU(self.block(x) + x)


class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
