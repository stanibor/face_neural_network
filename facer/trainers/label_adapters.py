from abc import ABC, abstractmethod
from typing import Iterator

from torch import nn
from torch.nn import Parameter


class ModelAdapterWrapper(nn.Module, ABC):
    def __init__(self, model: nn.Module, adapter: nn.Module, freeze_model: bool = True):
        super().__init__()
        self.model = model
        if freeze_model:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.adapter = adapter

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        yield from self.adapter.parameters(recurse=recurse)

    @abstractmethod
    def _forward_impl(self, x):
        return None

    def forward(self, x):
        return self._forward_impl(x)


class LandmarkAdapterWrapper(ModelAdapterWrapper):
    def _forward_impl(self, x):
        m, l = self.model(x)

        return self.adapter(l.flatten(start_dim=1)).view(l.shape)


class MaskAdapterWrapper(ModelAdapterWrapper):
    def _forward_impl(self, x):
        m, l = self.model(x)
        return self.adapter(m)

