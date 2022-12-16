from typing import Optional

import torch
import torchvision
from torch import Tensor


def apply_landmarks(imgs: Tensor, pred_landmarks: Tensor, gt_landmarks: Optional[Tensor] = None) -> Tensor:
    imgs = imgs.mul(255).byte()
    gt_landmarks = torch.full_like(pred_landmarks, -1.) if gt_landmarks is None else gt_landmarks
    for x, ldmks, gt_ldmks in zip(imgs, pred_landmarks, gt_landmarks):
        x[:] = torchvision.utils.draw_keypoints(x, gt_ldmks.unsqueeze(0), colors="#00FF00", radius=1)
        x[:] = torchvision.utils.draw_keypoints(x, ldmks.unsqueeze(0), colors="#FF0000", radius=1)
    return imgs.float().div_(255)


def apply_masks(imgs: Tensor,
                masks: Tensor,
                mask_color: Tensor = torch.tensor((0., 0., 1.)),
                alpha: float = 0.5) -> Tensor:
    mask_color = mask_color.view(1, 3, 1, 1)
    val_imgs = torch.lerp(imgs, mask_color, masks * alpha)
    return val_imgs