from collections import OrderedDict
from typing import Iterable

import torch
from torch import Tensor, nn as nn
import torch.nn as nn


class RegressionConnector(nn.Module):

    def __init__(self, input_channels: int, output_size: int, pool_size: int = 4, bias: bool = False):
        super().__init__()
        self.input_channels = input_channels

        self.avgpool = nn.AdaptiveAvgPool2d(pool_size)
        self.fc = nn.Linear(input_channels*pool_size**2, output_size, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class AtrousPiramidSumator(nn.Module):
    def __init__(self, in_channels, out_channels, piramid_levels):
        super().__init__()
        self.layers = nn.ModuleList([nn.Conv2d((in_channels >> i),
                                      out_channels,
                                      kernel_size=3,
                                      stride=(1 << i),
                                      dilation=i+1,
                                      padding=i+1,
                                      bias=True) for i in range(piramid_levels)])
        self.pooler = nn.Sequential(
            nn.Conv2d(out_channels * piramid_levels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.LeakyReLU(inplace=True))

    def _forward_impl(self, inputs: Iterable[Tensor]) -> Tensor:
        x = torch.cat([layer(input) for input, layer in zip(inputs, self.layers)], 1)
        x = self.pooler(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)
