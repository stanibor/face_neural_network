from os import PathLike
from pathlib import Path
import albumentations as A

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from facer.datasets.face_datasets import ImageDataset, LandmarkLocalizationDataset, SegmentationAndLandmarkDataset
from torch.utils.data.dataset import random_split

from facer.datasets.transforms import TO_TENSOR_TRANSFORM


class FaceSyntheticsModule(pl.LightningDataModule):
    def __init__(self,
                 root: PathLike,
                 test_root: PathLike = None,
                 transform: A.Compose = TO_TENSOR_TRANSFORM,
                 *,
                 batch_size=32,
                 seed: int = 42):
        super().__init__()
        self.root = Path(root)
        self.test_root = self.root.parent / "test" if test_root is None else Path(test_root)
        self.batch_size = batch_size
        self.seed = seed
        self.transform = transform
        self.dataset_train, self.dataset_val, self.dataset_test = (None,)*3

    def setup(self, stage=None):
        dataset = self._get_dataset(self.root, self.transform)
        generator = torch.Generator().manual_seed(self.seed)
        train_len = int(0.8 * len(dataset))
        valid_len = len(dataset) - train_len
        self.dataset_train, self.dataset_val = random_split(dataset, (train_len, valid_len), generator=generator)
        self.dataset_test = self._get_dataset(self.test_root)

    @staticmethod
    def _get_dataset(root, transform=TO_TENSOR_TRANSFORM):
        return ImageDataset(root, transform=transform)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size, shuffle=True, pin_memory=True, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.dataset_test, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size, shuffle=False, pin_memory=True, num_workers=4)


class LandmarkDataModule(FaceSyntheticsModule):
    @staticmethod
    def _get_dataset(root, transform=TO_TENSOR_TRANSFORM):
        return LandmarkLocalizationDataset(root, transform=transform)


class MasksAndLandmarksDataModule(FaceSyntheticsModule):
    @staticmethod
    def _get_dataset(root, transform=TO_TENSOR_TRANSFORM):
        return SegmentationAndLandmarkDataset(root, masks="bin_masks", transform=transform)

