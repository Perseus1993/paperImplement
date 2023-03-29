from typing import Type, Union, Optional, Callable, List

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 block_ch_in,
                 block_ch_out,
                 stride=1,
                 down_sample=None):
        # （size -3 +  2 *padding）/stride + 1 = size
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=block_ch_in,
                               out_channels=block_ch_out,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=block_ch_out,
                               out_channels=block_ch_out,
                               kernel_size=3,
                               padding=1,
                               stride=1,
                               bias=False)
        self.down_sample = down_sample
        self.relu = nn.ReLU(inplace=True)

        self.block = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(block_ch_out),
            self.relu,
            self.conv2,
            nn.BatchNorm2d(block_ch_out)
        )

    def forward(self, x: Tensor):
        if self.down_sample is not None:
            identity = self.down_sample(x)
        else:
            identity = x
        x = self.block(x)
        # print("identity shape", identity.shape)
        # print("x shape ", x.shape)
        x = x + identity
        return self.relu(x)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 block_ch_in,
                 block_ch_out,
                 stride=1,
                 down_sample=None):
        super().__init__()
        # 1*1卷积
        self.conv1 = nn.Conv2d(in_channels=block_ch_in,
                               out_channels=block_ch_out,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=block_ch_out,
                               out_channels=block_ch_out,
                               kernel_size=3,
                               padding=1,
                               stride=stride,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=block_ch_out,
                               out_channels=block_ch_out * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.block = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(block_ch_out),
            nn.ReLU(inplace=True),
            self.conv2,
            nn.BatchNorm2d(block_ch_out),
            nn.ReLU(inplace=True),
            self.conv3,
            nn.BatchNorm2d(block_ch_out * self.expansion)
        )
        self.down_sample = down_sample

    def forward(self, x):
        if self.down_sample is not None:
            identity = self.down_sample(x)
        else:
            identity = x
        return self.relu(self.block(x) + identity)


class ResNet(nn.Module):

    def __init__(
            self,
            layers: List[int],
            block: Type[Union[BasicBlock, Bottleneck]],
            num_classes=1000,
    ):
        super().__init__()
        self.in_channel = 64
        # in 224*224 out 112*112  (224 - 7 + 2 * padding)/2 = 112, padding = floor(3.5)
        self.conv1 = nn.Conv2d(3,
                               self.in_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False
                               )
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.ly1 = self._make_layer(block, 64, layers[0])
        self.ly2 = self._make_layer(block, 128,  layers[1], stride=2)
        self.ly3 = self._make_layer(block, 256, layers[2], stride=2)
        self.ly4 = self._make_layer(block, 512, layers[3], stride=2)
        self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.feature = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.max_pool,
            self.ly1,
            self.ly2,
            self.ly3,
            self.ly4,
            self.av_pool
        )

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    channel,
                    block_nums,
                    stride=1
                    ):
        # 判断identity是否需要下采样
        down_sample = None
        if stride != 1 or self.in_channel != block.expansion :
            down_sample = nn.Sequential(
                nn.Conv2d(self.in_channel,
                          channel * block.expansion,
                          stride=stride,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(channel * block.expansion)
            )

        layer_list = [block(self.in_channel, channel, stride, down_sample)]
        self.in_channel = channel * block.expansion

        for _ in range(1, block_nums):
            layer_list.append(
                block(
                    self.in_channel,
                    channel
                )
            )

        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = self.feature(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def resnet34(num_classes=1000):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet([3, 4, 6, 3], BasicBlock, num_classes=num_classes)
def resnet50(num_classes=1000):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet([3, 4, 6, 3], Bottleneck, num_classes=num_classes)

if __name__ == '__main__':
    nt = resnet34()
    for name, parameters in nt.named_parameters():
        print(name, ':', parameters.size())
