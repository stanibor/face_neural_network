import torch
import torchvision
from pytorch_lightning import Callback
import wandb


class LandmarkLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.gt_landmarks = val_samples
        self.resolution = self.gt_landmarks.new_tensor(self.val_imgs.shape[-2:])

    @staticmethod
    def _apply_landmarks(imgs, pred_landmarks, gt_landmarks):
        imgs = imgs.mul(255).byte()
        for x, ldmks, gt_ldmks in zip(imgs, pred_landmarks, gt_landmarks):
            x[:] = torchvision.utils.draw_keypoints(x, gt_ldmks.unsqueeze(0), colors="#FF0000", radius=1)
            x[:] = torchvision.utils.draw_keypoints(x, ldmks.unsqueeze(0), colors="#00FF00", radius=1)
        return imgs.float().div_(255)

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
        val_imgs = self._apply_landmarks(val_imgs[:n], landmarks[:n], gt_landmarks[:n])
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

        val_imgs = torch.lerp(val_imgs, mask_color, masks * self.mask_alpha)
        landmarks = landmarks.add(1).mul(resolution / 2)
        n = self.num_samples
        # Log the images as wandb Image
        val_imgs = self._apply_landmarks(val_imgs[:n], landmarks[:n], gt_landmarks[:n])
        trainer.logger.experiment.log({"examples": [wandb.Image(val_img) for val_img in val_imgs]})