from functools import lru_cache

import torch
import torch.nn as nn
from torch import nn as nn


class CoordConv2d(nn.Module):
    def __init__(self, in_channels, *args, **kwargs):
        super().__init__()
        self.coord = nn.Conv2d(in_channels=2, *args, **kwargs)
        self.conv = nn.Conv2d(in_channels=in_channels, *args, **kwargs)

    @staticmethod
    @lru_cache(4)
    def make_coord_channels(size: torch.Size):
        xc = torch.linspace(-1., 1., size[-1]).expand(1, 1, *size[-2:])
        yc = torch.linspace(-1., 1., size[-2]).unsqueeze(-1).expand(1, 1, *size[-2:])
        return torch.cat((xc, yc), 1)

    def forward(self, x: torch.Tensor):
        coords = self.make_coord_channels(x.shape[-2:]).to(x.device, x.dtype)
        return self.conv(x) + self.coord(coords)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, inplace_activation=True):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=inplace_activation),
                                  nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.BatchNorm2d(out_channels),
                                  nn.ReLU(inplace=inplace_activation),
                                  )

    def forward(self, x):
        return self.conv(x)
