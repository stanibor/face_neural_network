from dataclasses import asdict
from os import PathLike
from pathlib import Path
from typing import Optional, Tuple

import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
from tqdm import tqdm

from facer.datasets.face_datasets import SegmentationAndLandmarkDataset, MasksDataset
from facer.datasets.utils import DatasetPaths


class FaceDatasetProcessor:
    def __init__(self, directory):
        self.dataset_paths = DatasetPaths(Path(directory))

    def organize(self, output: Optional[PathLike[str]] = None):
        output = self.dataset_paths.directory if output is None else Path(output)
        output_paths = self.dataset_paths.with_directory(output)
        output_paths.images_directory.mkdir(parents=True, exist_ok=True)
        output_paths.masks_directory.mkdir(parents=True, exist_ok=True)
        output_paths.landmarks_directory.mkdir(parents=True, exist_ok=True)

        for image_file in tqdm(list(self.dataset_paths.directory.glob("*[0-9].png"))):
            mask_file = image_file.with_stem(image_file.stem + '_seg')
            ldmks_file = image_file.with_name(image_file.stem + '_ldmks.txt')

            image_file.rename(output_paths.images_directory / image_file.name)
            mask_file.rename(output_paths.masks_directory / image_file.name)
            ldmks_file.rename(output_paths.landmarks_directory / image_file.with_suffix('.txt').name)

    def crop(self, crop_width: int,
             output: Optional[PathLike[str]] = None,
             *,
             p_offset: Tuple[float, float] = (1.0, 0.9),
             batch_size: int = 64):
        output = self.dataset_paths.directory if output is None else Path(output)
        output_paths = self.dataset_paths.with_directory(output)

        output_paths.images_directory.mkdir(parents=True, exist_ok=True)
        output_paths.masks_directory.mkdir(parents=True, exist_ok=True)
        output_paths.landmarks_directory.mkdir(parents=True, exist_ok=True)

        dataset = SegmentationAndLandmarkDataset(**asdict(output_paths))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        digits = len(str(len(dataset) - 1))
        edges = torch.empty(batch_size, 2, 2)
        translator = torch.tensor((crop_width // 2,) * 2)

        p_offset = torch.tensor(p_offset)

        for i, batch in enumerate(tqdm(dataloader)):
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
                cropped_img.save(output_paths.image_directory / f"{name}.png")
                cropped_mask.save(output_paths.masks_directory / f"{name}.png")
                np.savetxt(str(output_paths.landmark_directory / f"{name}.txt"), new_landmarks[k].numpy(), fmt="%.3f")

    def binary_masks(self,
                     bin_masks: PathLike[str] = Path("bin_masks"),
                     classes: Tuple[int] = (1, 2, 7, 8, 10, 11),
                     *,
                     batch_size: int = 64):
        output_directory = self.dataset_paths.directory / bin_masks
        output_directory.mkdir(parents=True, exist_ok=True)
        classes = torch.tensor(classes)
        one = torch.ones(1).long()
        class_mask = one.bitwise_left_shift(classes).sum(-1)

        dataset = MasksDataset(**asdict(self.dataset_paths))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for i, masks in enumerate(tqdm(dataloader)):
            long_masks = one.bitwise_left_shift(masks).bitwise_and_(class_mask).ne_(0)
            byte_masks = long_masks.mul_(255).permute(0, 2, 3, 1).squeeze().byte()
            names = (output_directory / m_file.name for m_file in dataset.masks_files[i*batch_size:(i+1)*batch_size])
            for name, byte_mask in zip(names, byte_masks):
                PIL.Image.fromarray(byte_mask.numpy()).save(name)











