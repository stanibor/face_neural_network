from abc import ABC

import torch
import torchvision
from pytorch_lightning import Callback
import wandb

from facer.utils.visualisation import apply_landmarks, apply_masks



class LandmarkLogger(Callback):
    def __init__(self, val_samples, num_samples=32, connectivity=None):
        super().__init__()
        self.num_samples = num_samples
        self.connectivity = connectivity
        self.val_imgs, self.gt_landmarks = val_samples
        self.resolution = self.gt_landmarks.new_tensor(self.val_imgs.shape[-2:])

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        gt_landmarks = self.gt_landmarks.to(device=pl_module.device)
        resolution = self.resolution.to(device=pl_module.device)
        # Get model prediction
        output = pl_module(val_imgs)
        landmarks = output.add(1).mul(resolution / 2)
        n = self.num_samples
        # Log the images as wandb Image
        val_imgs = apply_landmarks(val_imgs[:n], landmarks[:n], gt_landmarks[:n], connectivity=self.connectivity)
        trainer.logger.experiment.log({"examples": [wandb.Image(val_img) for val_img in val_imgs]})


class MasksLogger(Callback):
    def __init__(self, val_samples, num_samples=32, mask_color=(0., 0, 1.), mask_alpha=0.5):
        super().__init__()
        self.num_samples = num_samples
        self.mask_color = torch.tensor(mask_color).view(1, 3, 1, 1)
        self.mask_alpha = mask_alpha
        self.val_imgs, self.gt_masks = val_samples

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        mask_color = self.mask_color.to(device=pl_module.device)
        # Get model prediction
        output = pl_module(val_imgs)
        masks = output

        # Log the images as wandb Image
        val_imgs = apply_masks(val_imgs, masks, mask_color, self.mask_alpha)
        trainer.logger.experiment.log({"examples": [wandb.Image(val_img) for val_img in val_imgs]})


class FaceImagesLogger(LandmarkLogger):
    def __init__(self, *args, mask_color=(0., 0, 1.), mask_alpha=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mask_color = torch.tensor(mask_color).view(1, 3, 1, 1)
        self.mask_alpha = mask_alpha

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        gt_landmarks = self.gt_landmarks.to(device=pl_module.device)
        resolution = self.resolution.to(device=pl_module.device)
        mask_color = self.mask_color.to(device=pl_module.device)
        # Get model prediction
        output = pl_module(val_imgs)
        masks, landmarks = output

        landmarks = landmarks.add(1).mul(resolution / 2)
        n = self.num_samples
        # Log the images as wandb Image
        val_imgs = apply_masks(val_imgs, masks, mask_color, self.mask_alpha)
        val_imgs = apply_landmarks(val_imgs[:n], landmarks[:n], gt_landmarks[:n], connectivity=self.connectivity)
        trainer.logger.experiment.log({"examples": [wandb.Image(val_img) for val_img in val_imgs]})