from dataclasses import asdict
from os import PathLike
from pathlib import Path
import random
from typing import Optional, Tuple

import PIL
import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from facer.datasets.face_datasets import SegmentationAndLandmarkDataset, ImageDataset
from facer.datasets.transforms import TO_TENSOR_TRANSFORM
from facer.datasets.utils import DatasetPaths


class FaceDatasetProcessor:
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

    def organize(self, output: Optional[PathLike] = None):
        output = self.dataset_paths.directory if output is None else Path(output)
        output_paths = self.dataset_paths.with_directory(output)
        self._print("Creating output directories")
        self._print(f"Images: {output_paths.images_directory}")
        output_paths.images_directory.mkdir(parents=True, exist_ok=True)
        self._print(f"Masks: {output_paths.masks_directory}")
        output_paths.masks_directory.mkdir(parents=True, exist_ok=True)
        self._print(f"Landmarks: {output_paths.landmarks_directory}")
        output_paths.landmarks_directory.mkdir(parents=True, exist_ok=True)
        self._print(f"Reorganizing files from: {self.dataset_paths.directory}")
        for image_file in self._tqdm(list(self.dataset_paths.directory.glob("*[0-9].png"))):
            mask_file = image_file.with_stem(image_file.stem + '_seg')
            ldmks_file = image_file.with_name(image_file.stem + '_ldmks.txt')

            image_file.replace(output_paths.images_directory / image_file.name)
            mask_file.replace(output_paths.masks_directory / image_file.name)
            ldmks_file.replace(output_paths.landmarks_directory / image_file.with_suffix('.txt').name)

    def split(self,
              test: PathLike = Path("test"),
              split: float = 0.1,
              *,
              seed: int = 42):

        root = Path(self.dataset_paths.directory)

        stems = [f"*{file.stem}.*" for file in self.dataset_paths.images_directory.iterdir() if file.is_file()]
        np.random.default_rng(seed=seed).shuffle(stems)
        test_len = int(float(split) * len(stems))
        train_stems, test_stems = stems[test_len:], stems[:test_len]

        dirs = [directory for directory in root.iterdir() if directory.is_dir()]

        self._print("Creating output directories")
        train_directory = root
        test_directory = root / test
        for directory in dirs:
            relative_dir = directory.relative_to(root)
            (test_directory / relative_dir).mkdir(parents=True, exist_ok=True)

        self._print("Collecting test files")
        relative_test_files = [file.relative_to(root)for stem in self._tqdm(test_stems) for file in root.rglob(stem)]

        self._print("Moving test files")
        for relative_file in self._tqdm(relative_test_files):
            file = root / relative_file
            new_path = test_directory / relative_file
            file.replace(new_path)

        # self._print("Cleaning old empty directories")
        # for directory in dirs:
        #     directory.rmdir()

    def crop(self, crop_width: int,
             output: Optional[PathLike] = None,
             *,
             p_offset: Tuple[float, float] = (1.0, 0.9),
             batch_size: int = 64):
        output = self.dataset_paths.directory if output is None else Path(output)
        output_paths = self.dataset_paths.with_directory(output)

        output_paths.directory.mkdir(parents=True, exist_ok=True)

        output_paths.images_directory.mkdir(parents=True, exist_ok=True)
        output_paths.masks_directory.mkdir(parents=True, exist_ok=True)
        output_paths.landmarks_directory.mkdir(parents=True, exist_ok=True)

        dataset = SegmentationAndLandmarkDataset(**asdict(self.dataset_paths), transform=TO_TENSOR_TRANSFORM)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        digits = len(str(len(dataset) - 1))
        edges = torch.empty(batch_size, 2, 2)
        translator = torch.tensor((crop_width // 2,) * 2)

        p_offset = torch.tensor(p_offset)

        for i, batch in enumerate(self._tqdm(dataloader)):
            images, masks, landmarks = batch
            samples = len(landmarks)
            torch.aminmax(landmarks, dim=-2, out=(edges[:samples, 1], edges[:samples, 0]))
            centers = edges[:samples].mean(-2).mul_(p_offset)
            corners = centers.sub_(translator).round_().int()
            new_landmarks = landmarks.sub(corners.unsqueeze(1))
            for k, corner in enumerate(corners):
                cropped_img = TF.to_pil_image(TF.crop(images[k], corner[1], corner[0], crop_width, crop_width))
                cropped_mask = TF.to_pil_image(TF.crop(masks[k], corner[1], corner[0], crop_width, crop_width))
                name = f"{(i * batch_size + k):0{digits}}"
                cropped_img.save(output_paths.images_directory / f"{name}.png")
                cropped_mask.save(output_paths.masks_directory / f"{name}.png")
                np.savetxt(str(output_paths.landmarks_directory / f"{name}.txt"), new_landmarks[k].numpy(), fmt="%.3f")

    def binary_masks(self,
                     bin_masks: PathLike = Path("bin_masks"),
                     classes: Tuple[int] = (1, 2, 7, 8, 10, 11),
                     *,
                     batch_size: int = 64):
        output_directory = self.dataset_paths.directory / bin_masks
        output_directory.mkdir(parents=True, exist_ok=True)
        classes = torch.tensor(classes)
        one = torch.ones(1).long()
        class_mask = one.bitwise_left_shift(classes).sum(-1)

        dataset = ImageDataset(self.dataset_paths.directory, images=self.dataset_paths.masks)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i, masks in enumerate(self._tqdm(dataloader)):
            masks = masks.select(-1, 0)
            long_masks = one.bitwise_left_shift(masks).bitwise_and_(class_mask).ne_(0)
            byte_masks = long_masks.mul_(255).squeeze().byte()
            names = (output_directory / m_file.name for m_file in dataset.img_files[i*batch_size:(i+1)*batch_size])
            for name, byte_mask in zip(names, byte_masks):
                PIL.Image.fromarray(byte_mask.numpy()).save(name)

    def transform_dataset(self,
                          transform_json: PathLike,
                          *,
                          batch_size: int = 64,
                          seed: int = 42):

        random.seed(seed)
        transform = A.load(str(transform_json))
        keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
        transform = A.Compose(transform.transforms, keypoint_params=keypoint_params)

        dataset = SegmentationAndLandmarkDataset(**asdict(self.dataset_paths), transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i, batch in enumerate(self._tqdm(dataloader)):
            images, masks, landmarks = batch















