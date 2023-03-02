from typing import Type, Union

import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, ch_in, ch_out):
        # （size -3 +  2 *padding）/stride + 1 = size ,因此padding = 1
        super().__init__()
        # 这里还是分两个conv，即使一样
        self.conv1 = nn.Conv2d(in_channels=ch_in, out_channels=ch_out, kernel_size=3,
                               padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=ch_out, out_channels=ch_out, kernel_size=3,
                               padding=1, bias=False)

        self.block = nn.Sequential(
            self.conv1(),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            self.conv2(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return nn.ReLU(self.block(x) + x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, ch_in):
        super().__init__()
        ch_out = ch_in * self.expansion
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
            nn.ReLU(inplace=True),
            self.conv2(),
            nn.BatchNorm2d(ch_in),
            nn.ReLU(inplace=True),
            self.conv3(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        return nn.ReLU(self.block(x) + x)


class ResNet(nn.Module):
    into_block_channel = 64

    def __init__(
            self,
            block: Type[Union[BasicBlock,Bottleneck]]
    ):
        super().__init__()
        # in 224*224 out 112*112  (224 - 7 + 2 * padding)/2 = 112, padding = floor(3.5)
        self.conv1 = nn.Conv2d(3,
                               self.into_block_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False
                               )
    def make_layer(self,
                   block: Type[Union[BasicBlock,Bottleneck]],
                   stride
                   ):
        # 判断identity是否需要下采样
        if stride != 1 or