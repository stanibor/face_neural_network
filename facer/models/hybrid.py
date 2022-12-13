from typing import Tuple

from torch import Tensor

from facer.models.decoder import ResnetUnetDecoder
from facer.models.landmarks import PyramidRegressionModel


class CoupledFaceModel(PyramidRegressionModel):
    def __init__(self, *args, out_channels=1, **kwargs):
        super().__init__(*args, **kwargs)
        self.decoder = ResnetUnetDecoder(inplanes=self.backbone.inplanes, out_channels=out_channels)

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.backbone(x)
        mask = self.decoder(*x)[-1]
        landmarks = self.regressor(x[:self.levels])
        return mask, landmarks.view(-1, *self.output_shape)


class TightlyCoupledFaceModel(CoupledFaceModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.regressor[0] = self._connector_class(self.backbone.inplanes*2)

    def _forward_impl(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.backbone(x)
        y = self.decoder(*x)
        landmarks = self.regressor(y[:self.levels])
        mask = y[-1]
        return mask, landmarks.view(-1, *self.output_shape)
