from abc import ABC, abstractmethod

from torch import nn


class ModelAdapterWrapper(nn.Module, ABC):
    def __init__(self, model: nn.Module, adapter: nn.Module, freeze_model: bool = True):
        super().__init__()
        self.model = model
        if freeze_model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.adapter = adapter

    @abstractmethod
    def _forward_impl(self, x):
        return None

    def forward(self, x):
        return self._forward_impl(x)


class LandmarkAdapterWrapper(ModelAdapterWrapper):
    def _forward_impl(self, x):
        l, m = self.model(x)

        return self.adapter(l.flatten(start_dim=1)).view(l.shape)


class MaskAdapterWrapper(ModelAdapterWrapper):
    def _forward_impl(self, x):
        l, m = self.model(x)
        return self.adapter(m)

