from collections import OrderedDict

from torch import Tensor
import torch.nn as nn
from torchvision.models import ResNet


class ResnetBackbone(nn.Module):
    def __init__(self, backbone: ResNet):
        assert isinstance(backbone, ResNet), "This class requires backbone class to be ResNet"
        super().__init__()
        self.start_conv = nn.Sequential(OrderedDict({
            "conv1": backbone.conv1,
            "bn1": backbone.bn1,
            "relu": backbone.relu,
            "maxpool": backbone.maxpool,
        }))
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.inplanes = backbone.inplanes

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.start_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)