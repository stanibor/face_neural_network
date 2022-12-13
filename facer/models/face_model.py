import math
from collections import OrderedDict
from typing import Union, Iterable

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet, resnet34, ResNet34_Weights
from facer.models.backbone import ResnetBackbone
from facer.models.blocks import CoordConv2d
from facer.models.regressor import RegressionConnector, PyramidPooler

DEFAULT_RESNET = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
# DEFAULT_RESNET = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)


class LandmarkRegressionModel(nn.Module):
    def __init__(self,
                 output_shape: Union[Iterable[int], torch.Size],
                 backbone: ResNet = DEFAULT_RESNET,
                 pool_size: int = 2,
                 hidden_channels: int = 1024,
                 *,
                 dropout=0.2):
        super().__init__()
        self.output_shape = torch.Size(output_shape)
        self.hidden_channels = hidden_channels
        self.backbone = ResnetBackbone(backbone=backbone)
        self.connector = self._connector_class()
        self.regressor = nn.Sequential(OrderedDict({
            "connector": RegressionConnector(input_channels=self.hidden_channels,
                                             output_size=self.hidden_channels * 4,
                                             pool_size=pool_size),
            "bn1": nn.BatchNorm1d(self.hidden_channels * 4, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            "dropout": nn.Dropout(dropout, inplace=False),
            "lrelu": nn.LeakyReLU(inplace=True),
            "fc": nn.Linear(self.hidden_channels * 4, self.output_shape.numel(), bias=True),
        }))

    def _connector_class(self):
        return CoordConv2d(in_channels=self.backbone.inplanes,
                           out_channels=self.hidden_channels,
                           kernel_size=(3, 3),
                           padding=(1, 1),
                           bias=False)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.backbone(x)[0]
        x = self.connector(x)
        x = self.regressor(x)
        return x.view(-1, *self.output_shape)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


class PyramidRegressionModel(LandmarkRegressionModel):
    def __init__(self, *args, levels=2, **kwargs):
        self.levels = levels
        super().__init__(*args, **kwargs)

    def _connector_class(self):
        return PyramidPooler(self.backbone.inplanes,
                             self.hidden_channels,
                             self.levels,
                             self.hidden_channels//2)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.backbone(x)[:self.levels]
        x = self.connector(x)
        x = self.regressor(x)
        return x.view(-1, *self.output_shape)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

