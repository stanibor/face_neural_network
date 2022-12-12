from dataclasses import dataclass, astuple
from os import PathLike
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import default_collate


def to_device_collate(batch, device: Union[torch.device, str]):
    device = torch.device(device)
    return list(x.to(device) for x in default_collate(batch))


@dataclass
class DatasetPaths:
    directory: PathLike
    images: PathLike = Path("images")
    landmarks: PathLike = Path("landmarks")
    masks: PathLike = Path("masks")

    def __post_init__(self):
        self.directory = Path(self.directory)

    def with_directory(self, directory: Path):
        return DatasetPaths(directory, *astuple(self)[1:])

    @property
    def images_directory(self):
        return Path(self.directory) / self.images

    @property
    def landmarks_directory(self):
        return Path(self.directory) / self.landmarks

    @property
    def masks_directory(self):
        return Path(self.directory) / self.masks


