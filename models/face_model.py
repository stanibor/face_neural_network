from collections import OrderedDict

import torch
import torch.nn as nn
from torch import Tensor
from torchvision.models import ResNet, resnet34, ResNet34_Weights
from models.backbone import ResnetBackbone
from models.regressor import RegressionConnector

DEFAULT_RESNET = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)


class LandmarkRegressionModel(nn.Module):
    def __init__(self, output_shape: torch.Size, backbone: ResNet = DEFAULT_RESNET, pool_size: int = 4, *, dropout=0.2):
        super().__init__()
        self.output_shape = torch.Size(output_shape)
        self.backbone = ResnetBackbone(backbone=backbone)
        self.regressor = nn.Sequential(OrderedDict({
            "connector": RegressionConnector(input_channels=self.backbone.inplanes,
                                             output_size=self.backbone.inplanes,
                                             pool_size=pool_size),
            "bn1": nn.BatchNorm1d(self.backbone.inplanes),
            "dropout": nn.Dropout(dropout),
            "lrelu": nn.LeakyReLU(),
            "fc": nn.Linear(self.backbone.inplanes, self.output_shape.numel(), bias=True),
            "relu": nn.ReLU()
        }))

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.backbone(x)
        x = self.regressor(x)
        return x.view(-1, *self.output_shape)

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)