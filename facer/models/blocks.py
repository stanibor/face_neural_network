from functools import lru_cache

import torch
import torch.nn as nn


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