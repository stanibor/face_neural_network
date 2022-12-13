import torch.nn as nn
from torchvision.models import ResNet

from facer.models.backbone import ResnetBackbone, DEFAULT_RESNET
from facer.models.decoder import ResnetUnetDecoder


class SegmentationModel(nn.Module):
    def __init__(self, backbone: ResNet = DEFAULT_RESNET, out_channels=1):
        super().__init__()
        self.backbone = ResnetBackbone(backbone)
        self.decoder = ResnetUnetDecoder(self.backbone.inplanes, out_channels=out_channels)

    def _forward_impl(self, x):
        x = self.backbone(x)
        x = self.decoder(*x)[-1]
        return x

    def forward(self, x):
        return self._forward_impl(x)