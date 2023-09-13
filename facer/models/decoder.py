import torch
import torch.nn as nn

from facer.models.blocks import DoubleConv


class UnetDecoder(nn.Module):
    def __init__(self, inplanes, out_channels):
        super().__init__()
        self.inplanes = inplanes << 1
        self.bottleneck = nn.Sequential(nn.MaxPool2d(kernel_size=2, stride=2),
                                        DoubleConv(inplanes, inplanes << 1),
                                        nn.ConvTranspose2d(inplanes << 1, inplanes, kernel_size=2, stride=2))
        self.layer1 = nn.Sequential(DoubleConv(inplanes << 1, inplanes),
                                    nn.ConvTranspose2d(inplanes, inplanes >> 1, kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(DoubleConv(inplanes, inplanes >> 1),
                                    nn.ConvTranspose2d(inplanes >> 1, inplanes >> 2, kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(DoubleConv(inplanes >> 1, inplanes >> 2),
                                    nn.ConvTranspose2d(inplanes >> 2, inplanes >> 3, kernel_size=2, stride=2))
        self.layer4 = nn.Sequential(DoubleConv(inplanes >> 2, inplanes >> 3),
                                    nn.LeakyReLU(),
                                    nn.Conv2d(inplanes >> 3, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.Sigmoid())

    def _forward_impl(self, x, x3, x2, x1):
        y = y0 = torch.cat((x, self.bottleneck(x)), dim=1)
        y = y1 = torch.cat((x3, self.layer1(y)), dim=1)
        y = y2 = torch.cat((x2, self.layer2(y)), dim=1)
        y = y3 = torch.cat((x1, self.layer3(y)), dim=1)
        y = self.layer4(y)
        return y0, y1, y2, y3, y

    def forward(self, x, x3, x2, x1):
        return self._forward_impl(x, x3, x2, x1)


class ResnetUnetDecoder(UnetDecoder):
    def __init__(self, inplanes, out_channels):
        super().__init__(inplanes, out_channels)
        self.layer4 = nn.Sequential(DoubleConv(inplanes >> 2, inplanes >> 3),
                                    nn.ConvTranspose2d(inplanes >> 3, inplanes >> 4, kernel_size=2, stride=2),
                                    nn.ReLU(inplace=True))

        self.deconv = nn.Sequential(nn.ConvTranspose2d(inplanes >> 4, inplanes >> 4, kernel_size=2, stride=2),
                                    nn.LeakyReLU(inplace=True),
                                    nn.Conv2d(inplanes >> 4, out_channels, kernel_size=3, stride=1, padding=1),
                                    nn.Sigmoid())

    def _forward_impl(self, x, x3, x2, x1):
        y = y0 = torch.cat((x, self.bottleneck(x)), dim=1)
        y = y1 = torch.cat((x3, self.layer1(y)), dim=1)
        y = y2 = torch.cat((x2, self.layer2(y)), dim=1)
        y = y3 = torch.cat((x1, self.layer3(y)), dim=1)
        y = self.layer4(y)
        y = self.deconv(y)
        return y0, y1, y2, y3, y



