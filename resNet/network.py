from typing import Type, Union, Optional, Callable, List

import torch
import torch.nn as nn
from torch import Tensor


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 block_ch_in,
                 # block_ch_out,
                 stride_first_conv=1,
                 down_sample=None):
        # （size -3 +  2 *padding）/stride + 1 = size
        super().__init__()
        block_ch_out = block_ch_in
        self.conv1 = nn.Conv2d(in_channels=block_ch_in,
                               out_channels=block_ch_in,
                               kernel_size=3,
                               stride=stride_first_conv,
                               padding=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=block_ch_in,
                               out_channels=block_ch_out,
                               kernel_size=3,
                               padding=1,
                               bias=False)
        self.down_sample = down_sample

        self.block = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(block_ch_in),
            nn.ReLU(inplace=True),
            self.conv2,
            nn.BatchNorm2d(block_ch_out)
        )

    def forward(self, x: Tensor):
        if self.down_sample is not None:
            identity = self.down_sample(x)
        else:
            identity = x
        return nn.ReLU(self.block(x) + identity)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self,
                 block_ch_in,
                 # block_ch_out,
                 stride_second_conv=1,
                 down_sample=None):
        super().__init__()
        # 1*1卷积
        self.conv1 = nn.Conv2d(in_channels=block_ch_in,
                               out_channels=block_ch_in,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.conv2 = nn.Conv2d(in_channels=block_ch_in,
                               out_channels=block_ch_in,
                               kernel_size=3,
                               padding=1,
                               stride=stride_second_conv,
                               bias=False)
        self.conv3 = nn.Conv2d(in_channels=block_ch_in,
                               out_channels=block_ch_in * self.expansion,
                               kernel_size=1,
                               stride=1,
                               bias=False)
        self.block = nn.Sequential(
            self.conv1(),
            nn.BatchNorm2d(block_ch_in),
            nn.ReLU(inplace=True),
            self.conv2(),
            nn.BatchNorm2d(block_ch_in),
            nn.ReLU(inplace=True),
            self.conv3(),
            nn.BatchNorm2d(block_ch_in * self.expansion)
        )
        self.down_sample = down_sample

    def forward(self, x):
        if self.down_sample is not None:
            identity = self.down_sample(x)
        else:
            identity = x
        return nn.ReLU(self.block(x) + identity)


class ResNet(nn.Module):
    into_res_channel = 64

    def __init__(
            self,
            layers: List[int],
            block: Type[Union[BasicBlock, Bottleneck]],
            num_classes=1000,
    ):
        super().__init__()
        # in 224*224 out 112*112  (224 - 7 + 2 * padding)/2 = 112, padding = floor(3.5)
        self.conv1 = nn.Conv2d(3,
                               self.into_res_channel,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False
                               )
        self.ly1 = self._make_layer(block, 64, layers[0])
        self.ly2 = self._make_layer(block, 128, layers[1], stride=2)
        self.ly3 = self._make_layer(block, 256, layers[2], stride=2)
        self.ly4 = self._make_layer(block, 512, layers[3], stride=2)
        self.av_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.feature = nn.Sequential(
            self.conv1,
            nn.BatchNorm2d(self.into_res_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self.ly1,
            self.ly2,
            self.ly3,
            self.ly4,
            self.av_pool
        )

    def _make_layer(self,
                    block: Type[Union[BasicBlock, Bottleneck]],
                    in_layer_channel,
                    block_nums,
                    stride=1
                    ):
        # 判断identity是否需要下采样
        if stride != 1 or block.expansion != 1:
            down_sample = nn.Sequential(
                nn.Conv2d(in_layer_channel,
                          in_layer_channel * block.expansion,
                          kernel_size=1),
                nn.BatchNorm2d(in_layer_channel * block.expansion)
            )

        layer_list = [block(in_layer_channel, stride)]
        tmp_channel = block.expansion * in_layer_channel

        for _ in range(1, block_nums):
            layer_list.append(
                block(
                    tmp_channel,
                    stride
                )
            )

        return nn.Sequential(*layer_list)

    def forward(self, x):
        x = torch.flatten(self.feature(x), 1)
        x = self.fc(x)
        return x


def resnet34(num_classes=1000):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet([3, 4, 6, 3], BasicBlock, num_classes=num_classes)


if __name__ == '__main__':
    nt = resnet34()
    for name, parameters in nt.named_parameters():
        print(name, ':', parameters.size())
