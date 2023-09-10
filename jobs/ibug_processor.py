from os import PathLike
from pathlib import Path
from typing import Optional

import PIL.Image
import cv2
import numpy as np
import torch
from tqdm import tqdm

from facer.datasets.utils import DatasetPaths
import albumentations as A


class DatasetIBUGProcessor:
    def __init__(self,
                 directory: PathLike,
                 *,
                 images: PathLike = DatasetPaths.images,
                 landmarks: PathLike = DatasetPaths.landmarks,
                 masks: PathLike = DatasetPaths.masks,
                 verbose: bool = True):
        self.dataset_paths = DatasetPaths(Path(directory), images=images, landmarks=landmarks, masks=masks)
        self._verbose = bool(verbose)

    def _print(self, *args, **kwargs):
        if self._verbose:
            print(*args, **kwargs)

    def _tqdm(self, *args, **kwargs):
        return tqdm(*args, **kwargs, disable=(not self._verbose))

    def reformat(self, output: Optional[PathLike] = None,
                 *,
                 p_margin = 0.7,
                 p_offset = (1.0, 0.90),
                 size = 256,
                 glob_expr = "*.jpg"):
        output = self.dataset_paths.directory if output is None else Path(output)
        output_paths = self.dataset_paths.with_directory(output)
        self._print(f"Creating output landmarks directory: {output_paths.landmarks_directory}")
        output_paths.landmarks_directory.mkdir(parents=True, exist_ok=True)
        self._print(f"Creating output images directory: {output_paths.images_directory}")
        output_paths.images_directory.mkdir(parents=True, exist_ok=True)
        self._print(f"Creating output masks directory: {output_paths.masks_directory}")
        output_paths.masks_directory.mkdir(parents=True, exist_ok=True)

        p_offset = torch.tensor(p_offset).sub(1)
        p_multiplier = 1 + p_margin
        kparams = A.KeypointParams(format='xy', remove_invisible=False)

        for image_file in self._tqdm(list(self.dataset_paths.directory.glob(glob_expr))):
            landmarks_file = image_file.parent / (image_file.stem + "_landmark.txt")
            mask_file = image_file.with_suffix(".png")
            img = cv2.imread(str(image_file))
            mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            imsize = torch.tensor(img.shape[:-1]).flip(0)
            landmarks = torch.from_numpy(np.loadtxt(landmarks_file))

            amin, amax = torch.aminmax(landmarks, dim=-2)
            span = amax.sub(amin).mul_(p_multiplier).max()
            span = torch.minimum(span, imsize.amin())
            center = amax.add(amin).div_(2)
            min_corner = center.sub(span.div(2)).addcmul_(p_offset, span).int()
            min_corner.sub_(min_corner.clamp_max(0))

            max_corner = min_corner.add(span).ceil().int()
            residual = max_corner.sub(max_corner.clamp_max(imsize)).max()
            max_corner.sub_(residual)

            transform = A.Compose([A.Crop(*min_corner, *max_corner), A.Resize(size, size)], keypoint_params=kparams)
            output = transform(image=img, keypoints=landmarks.numpy(), mask=mask)
            img, landmarks, mask = output["image"], output["keypoints"], output["mask"]
            landmarks = np.array(landmarks)

            cv2.imwrite(str(output_paths.images_directory / image_file.with_suffix(".png").name), img)
            cv2.imwrite(str(output_paths.masks_directory / image_file.with_suffix(".png").name), mask)
            np.savetxt(str(output_paths.landmarks_directory / image_file.with_suffix('.txt').name),
                       landmarks,
                       fmt="%.3f")





if __name__ == "__main__":
    processor = DatasetIBUGProcessor("/data/Datasets/ibugmask_release/test")
    processor.reformat("/data/Datasets/ibugmask/test")
    processor = DatasetIBUGProcessor("/data/Datasets/ibugmask_release/train")
    processor.reformat("/data/Datasets/ibugmask/train", glob_expr="*_00.jpg")

