from collections import OrderedDict
from typing import Tuple, Optional

from torch import Tensor
import torch.nn as nn
from torchvision.models import resnet34, ResNet34_Weights
from torchvision.models.resnet import ResNet


def resnet_by_name(name: str, weights: Optional[str] = None):
    from torchvision.models.resnet import __dict__ as resnet_dict
    assert name in resnet_dict and 'resnet' in name, "Can't use this as backbone"
    return resnet_dict[name](weights=weights)


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

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        x = self.start_conv(x)
        x = x1 = self.layer1(x)
        x = x2 = self.layer2(x)
        x = x3 = self.layer3(x)
        x = self.layer4(x)
        return x, x3, x2, x1

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        return self._forward_impl(x)


DEFAULT_RESNET = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
