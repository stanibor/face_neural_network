from typing import Tuple

from torch import nn as nn, Tensor

from facer.models.blocks import DoubleConv


class UnetEncoder(nn.Module):
    def __init__(self, in_channels, start_planes=64):
        super().__init__()
        self.start_planes = start_planes
        self.in_channels = in_channels
        self.layer1 = DoubleConv(in_channels, start_planes)
        self.layer2 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    DoubleConv(start_planes, start_planes << 1), )
        self.layer3 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    DoubleConv(start_planes << 1, start_planes << 2), )
        self.layer4 = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                    DoubleConv(start_planes << 2, start_planes << 3), )

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = x1 = self.layer1(x)
        x = x2 = self.layer2(x)
        x = x3 = self.layer3(x)
        x = self.layer4(x)
        return x, x3, x2, x1

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._forward_impl(x)
