from torch import nn as nn

from facer.models.decoder import UnetDecoder
from facer.models.encoder import UnetEncoder


class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, start_planes=64):
        super().__init__()
        self.encoder = UnetEncoder(in_channels=in_channels, start_planes=start_planes)
        self.decoder = UnetDecoder(self.encoder.start_planes << 3, out_channels=out_channels)

    def _forward_impl(self, x):
        x = self.encoder(x)
        x = self.decoder(*x)[-1]
        return x

    def forward(self, x):
        return self._forward_impl(x)
