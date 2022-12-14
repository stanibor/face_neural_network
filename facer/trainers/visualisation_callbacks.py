import torch
import torchvision
from pytorch_lightning import Callback
import wandb


class FaceImagesLogger(Callback):
    def __init__(self, val_samples, num_samples=32, mask_color = (0., 0, 1.)):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, _, self.gt_landmarks = val_samples
        self.resolution = self.gt_landmarks.new_tensor(self.val_imgs.shape[-2:])
        self.mask_color = torch.tensor(mask_color).view(1, 3, 1, 1)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        gt_landmarks = self.gt_landmarks.to(device=pl_module.device)
        resolution = self.resolution.to(device=pl_module.device)
        mask_color = self.mask_color.to(device=pl_module.device)
        # Get model prediction
        output = pl_module(val_imgs)
        masks, landmarks = output

        val_imgs = torch.lerp(val_imgs, mask_color, masks).mul_(255).byte()
        landmarks = landmarks.add(1).mul(resolution / 2)
        n = self.num_samples
        for x, ldmks, gt_ldmks in zip(val_imgs[:n], landmarks[:n], gt_landmarks[:n]):
            x[:] = torchvision.utils.draw_keypoints(x, gt_ldmks.unsqueeze(0), colors="#FF0000", radius=1)
            x[:] = torchvision.utils.draw_keypoints(x, ldmks.unsqueeze(0), colors="#00FF00", radius=1)
        # Log the images as wandb Image
        val_imgs = val_imgs.float().div_(255)
        trainer.logger.experiment.log({"examples": [wandb.Image(val_img) for val_img in val_imgs]})