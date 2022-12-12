from typing import Tuple

import torch
import torch.nn as nn


class OcularNMELoss(nn.Module):
    def __init__(self, ocular_indices: Tuple[int, int]):
        super().__init__()
        self.ocular_indices = torch.tensor(ocular_indices)

    def ocular_distances(self, landmarks):
        self.ocular_indices = self.ocular_indices.to(landmarks.device)
        eyes = landmarks.index_select(-2, self.ocular_indices)
        eyes.select(-2, 0).mul_(-1)
        return eyes.sum(-2).pow(2).sum(-1).sqrt()

    def forward(self, x, y):
        dists = self.ocular_distances(y)
        point_distances = (y - x).pow(2).sum(-1).sqrt()
        return point_distances.sum(-1).div_(dists).mean()