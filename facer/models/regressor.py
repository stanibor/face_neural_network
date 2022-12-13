from collections import OrderedDict
from typing import Iterable

import torch
from torch import Tensor, nn as nn
import torch.nn as nn

from facer.models.blocks import CoordConv2d


class RegressionLink(nn.Module):

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


class PyramidPooler(nn.Module):
    @staticmethod
    def _block(channels, hidden_channels, level):
        in_channels = channels >> level
        pool_window = 1 << level
        block = nn.Sequential(CoordConv2d(in_channels=in_channels,
                                          out_channels=hidden_channels,
                                          kernel_size=3,
                                          stride=1,
                                          padding=1,
                                          bias=False),
                              nn.BatchNorm2d(hidden_channels),
                              nn.AvgPool2d(kernel_size=pool_window, stride=pool_window))
        return block

    def __init__(self, in_channels, out_channels, levels=2, hidden_channels=256):
        super().__init__()
        self.layers = nn.ModuleList([self._block(in_channels, hidden_channels, level) for level in range(levels)])
        self.pooler = nn.Sequential(nn.ReLU(),
                                    nn.Conv2d(hidden_channels * levels, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels),
                                    nn.LeakyReLU(inplace=True))

    def _forward_impl(self, inputs: Iterable[Tensor]) -> Tensor:
        x = torch.cat([layer(input) for input, layer in zip(inputs, self.layers)], 1)
        x = self.pooler(x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

