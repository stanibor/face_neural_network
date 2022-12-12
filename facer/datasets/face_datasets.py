from os import PathLike

import cv2
import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
import albumentations as A

from facer.datasets.utils import DatasetPaths
from facer.datasets.transforms import NO_TRANSFORM


class ImageDataset(data.Dataset):
    def __init__(self,
                 directory: PathLike,
                 *,
                 images: PathLike = DatasetPaths.images,
                 transform: A.Compose = NO_TRANSFORM):
        super().__init__()
        self.paths = DatasetPaths(Path(directory), images=images)
        self.transform: A.Compose = transform
        self.img_files = [Path(img) for img in self.paths.images_directory.glob('*.png')]

    def _get_image(self, index):
        img_path = self.img_files[index]
        image = cv2.imread(str(img_path))
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    def __getitem__(self, index):
        image = self._get_image(index)
        return self.transform(image=image)['image']

    def __len__(self):
        return len(self.img_files)


class SegmentationDataset(ImageDataset):
    def __init__(self,
                 directory: PathLike,
                 *,
                 images: PathLike = DatasetPaths.images,
                 masks: PathLike = DatasetPaths.masks,
                 transform: A.Compose = NO_TRANSFORM):
        super().__init__(directory, images=images, transform=transform)
        self.paths.masks = masks
        self.masks_files = [self.paths.masks_directory / img.name for img in self.img_files]
        assert len(self.masks_files) == len(self.img_files)

    def _get_mask(self, index):
        mask_path = self.masks_files[index]
        return cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    def __getitem__(self, index):
        image = self._get_image(index)
        mask = self._get_mask(index)
        item = self.transform(image=image, mask=mask)
        return item['image'], item['mask']


class LandmarkLocalizationDataset(ImageDataset):
    def __init__(self,
                 directory: PathLike,
                 *,
                 images: PathLike = DatasetPaths.images,
                 landmarks: PathLike = DatasetPaths.landmarks,
                 transform: A.Compose = NO_TRANSFORM):
        super().__init__(directory, images=images, transform=transform)
        self.paths.landmarks = landmarks
        self.landmark_files = [(self.paths.landmarks_directory / img.name).with_suffix(".txt") for img in self.img_files]
        assert len(self.landmark_files) == len(self.img_files)

    def _get_landmarks(self, index):
        ldmk_path = self.landmark_files[index]
        return torch.from_numpy(np.genfromtxt(ldmk_path, dtype='float32'))

    @staticmethod
    def _get_bbox(landmarks):
        bbox = torch.empty(4)
        torch.aminmax(landmarks, dim=0, out=(bbox[::2], bbox[1::2]))
        return bbox

    def __getitem__(self, index):
        image = self._get_image(index)
        landmarks = self._get_landmarks(index)
        item = self.transform(image=image, keypoints=landmarks)
        return item['image'], torch.tensor(item['keypoints'])


class SegmentationAndLandmarkDataset(SegmentationDataset, LandmarkLocalizationDataset):
    def __init__(self,
                 directory: PathLike,
                 *,
                 images: PathLike = DatasetPaths.images,
                 masks: PathLike = DatasetPaths.masks,
                 landmarks: PathLike = DatasetPaths.landmarks,
                 transform: A.Compose = NO_TRANSFORM):
        LandmarkLocalizationDataset.__init__(self, directory, images=images, landmarks=landmarks)
        SegmentationDataset.__init__(self, directory, images=images, masks=masks, transform=transform)

    def __getitem__(self, index):
        image = self._get_image(index)
        mask = self._get_mask(index)
        landmarks = self._get_landmarks(index)
        item = self.transform(image=image, mask=mask, keypoints=landmarks)
        return item['image'], item['mask'], torch.tensor(item['keypoints'])

