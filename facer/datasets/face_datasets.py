from ctypes import Union
from os import PathLike

import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF

from facer.datasets.utils import DatasetPaths


class ImageDataset(data.Dataset):
    def __init__(self, directory: PathLike[str], **kwargs):
        super().__init__()
        self.paths = DatasetPaths(Path(directory), **kwargs)
        self.img_files = [Path(img) for img in self.paths.image_directory.glob('*.png')]

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = Image.open(img_path)
        return TF.to_tensor(data)

    def __len__(self):
        return len(self.img_files)


class MasksDataset(data.Dataset):
    def __init__(self, directory: PathLike[str], **kwargs):
        super().__init__()
        self.paths = DatasetPaths(Path(directory), **kwargs)
        self.masks_files = [Path(img) for img in self.paths.masks_directory.glob('*.png')]

    def _get_mask(self, index):
        mask_path = self.masks_files[index]
        mask = Image.open(mask_path)
        return torch.tensor(mask.getdata(), dtype=torch.uint8).view(1, *mask.size)

    def __getitem__(self, index):
        return self._get_mask(index)

    def __len__(self):
        return len(self.masks_files)


class SegmentationDataset(ImageDataset, MasksDataset):
    def __init__(self, directory: PathLike[str], **kwargs):
        super().__init__(directory, **kwargs)
        self.mask_files = [self.paths.masks_directory / img.name for img in self.img_files]
        assert len(self.mask_files) == len(self.img_files)

    def __getitem__(self, index):
        image = super().__getitem__(index)
        mask = self._get_mask(index)
        return image, mask


class LandmarkLocalizationDataset(ImageDataset):
    def __init__(self, directory: PathLike[str], **kwargs):
        super().__init__(directory, **kwargs)
        self.landmark_files = [(self.paths.landmark_directory / img.name).with_suffix(".txt") for img in self.img_files]
        assert len(self.landmark_files) == len(self.img_files)

    def _get_landmarks(self, index):
        ldmk_path = self.landmark_files[index]
        return torch.from_numpy(np.genfromtxt(ldmk_path, dtype='float32'))

    def __getitem__(self, index):
        image = super().__getitem__(index)
        landmarks = self._get_landmarks(index)
        return image, landmarks


class SegmentationAndLandmarkDataset(SegmentationDataset, LandmarkLocalizationDataset):
    def __getitem__(self, index):
        image_and_landmarks, mask = super().__getitem__(index)
        image, landmarks = image_and_landmarks
        return image, mask, landmarks

