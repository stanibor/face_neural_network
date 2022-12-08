from os import PathLike
from pathlib import Path
from typing import Type

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from facer.datasets.face_datasets import ImageDataset, LandmarkLocalizationDataset
from torch.utils.data.dataset import random_split


class FaceSyntheticsModule(pl.LightningDataModule):
    dataset_class: Type[ImageDataset] = ImageDataset

    def __init__(self, root: PathLike[str], batch_size=32, seed: int = 42):
        super().__init__()
        self.root = Path(root)
        self.batch_size = batch_size
        self.seed = seed
        self.dataset_train, self.dataset_val, self.dataset_test = (None,)*3

    def setup(self, stage=None):
        dataset = self.dataset_class(self.root)
        generator = torch.Generator().manual_seed(self.seed)
        train_len = 0.8 * len(dataset)
        valid_len = len(dataset) - train_len
        self.dataset_train, self.dataset_val = random_split(dataset, (train_len, valid_len), generator=generator)
        self.dataset_test = self.dataset_class(self.root / "test")

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)


class LandmarkDataModule(FaceSyntheticsModule):
    dataset_class: Type[ImageDataset] = LandmarkLocalizationDataset
