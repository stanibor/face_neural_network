import torch
import torchvision
from pytorch_lightning import Callback
import wandb


class FaceImagesLogger(Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, _, self.gt_landmarks = val_samples
        self.resolution = self.gt_landmarks.new_tensor(self.val_imgs.shape[-2:])

    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        gt_landmarks = self.gt_landmarks.to(device=pl_module.device)
        resolution = self.resolution.to(device=pl_module.device)
        # Get model prediction
        output = pl_module(val_imgs)
        masks, landmarks = output

        val_imgs = val_imgs.clone()
        landmarks = landmarks.add(1).mul(resolution / 2)
        n = self.num_samples
        for x, mask, ldmks, gt_ldmks in zip(val_imgs[:n], masks[:n], landmarks[:n], gt_landmarks[:n]):
            x *= mask
            byte_x = x.mul(255).byte()
            x[:] = torchvision.utils.draw_keypoints(byte_x, gt_ldmks.unsqueeze(0), colors="#FF0000", radius=1).div(255)
            x[:] = torchvision.utils.draw_keypoints(byte_x, ldmks.unsqueeze(0), colors="#00FF00", radius=1).div(255)
        # Log the images as wandb Image
        trainer.logger.experiment.log({"examples" :[wandb.Image(val_img.div(255).float())
                                                    for val_img in val_imgs]})