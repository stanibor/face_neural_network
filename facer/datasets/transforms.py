from typing import Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

NO_TRANSFORM = A.Compose([])
TO_TENSOR_TRANSFORM = A.Compose([A.ToFloat(), ToTensorV2()])


def spatial_transforms(min_max_height: Tuple[int, int] = (300, 320), output_size: Tuple[int, int] = (256, 256)):
    transform = A.Compose([A.Rotate(limit=10, p=0.4),
                           A.RandomSizedCrop(min_max_height, output_size[0], output_size[1]),
                           A.HorizontalFlip(p=0.5)])
    return transform


def photometric_transforms():
    transform = A.Compose([A.RandomBrightnessContrast(p=0.5),
                           A.OneOf([A.RandomGamma(), A.RandomToneCurve()], p=0.05),
                           A.RGBShift(p=0.2),
                           A.OneOf([A.GaussianBlur(), A.Defocus()], p=0.05),
                           A.GaussNoise(p=0.2),
                           A.ISONoise(p=0.8)])
    return transform


def default_transform(min_max_height: Tuple[int, int] = (300, 320),
                      output_size: Tuple[int, int] = (256, 256),
                      has_landmarks: bool = False):
    keypoint_params = A.KeypointParams(format='xy', remove_invisible=False) if has_landmarks else None
    transform = A.Compose([*spatial_transforms(min_max_height, output_size),
                           *photometric_transforms(),
                           *TO_TENSOR_TRANSFORM],
                          keypoint_params=keypoint_params)
    return transform


