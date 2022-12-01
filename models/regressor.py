from collections import OrderedDict

import torch
from torch import Tensor
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

