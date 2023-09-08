from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics

from facer.trainers.utils import OcularNMELoss


class FaceModel(pl.LightningModule, ABC):
    def __init__(self, model: nn.Module, learning_rate=1e-3, weight_decay=0):
        super().__init__()
        self.save_hyperparameters(ignore=['model'])
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay

    @abstractmethod
    def compute_losses(self, x, y):
        pass

    def update_validation_metrics(self, outputs, gt):
        pass

    def update_test_metrics(self, outputs, gt):
        pass

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx):
        imgs, gt = batch
        outputs = self(imgs)
        losses = self.compute_losses(outputs, gt)
        return losses, outputs, gt

    @abstractmethod
    def _log_losses(self, losses, stage: str):
        pass

    def training_step(self, batch, batch_idx):
        losses, _, _ = self.common_step(batch, batch_idx)
        self._log_losses(losses, "train")
        loss = losses[0]
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        losses, outputs, gt = self.common_step(batch, batch_idx)
        self._log_losses(losses, "val")
        self.update_validation_metrics(outputs, gt)
        loss = losses[0]
        return {'val_loss': loss}

    def test_step(self, batch, batch_idx):
        losses, outputs, gt = self.common_step(batch, batch_idx)
        self._log_losses(losses, "test")
        self.update_test_metrics(outputs, gt)
        loss = losses[0]
        return {'loss': loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer], [lr_scheduler]


class LandmarkRegressor(FaceModel):
    def __init__(self, model: nn.Module, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.mse = nn.MSELoss()
        self.nme = OcularNMELoss(ocular_indices=(36, 45))

    def compute_losses(self, x, y):
        return self.mse(x, y), self.nme(x, y)

    def common_step(self, batch, batch_idx):
        imgs, ldmks = batch
        resolution = ldmks.new_tensor(imgs.shape[-2:])/2
        ldmks = ldmks.div(resolution).sub_(1.)
        outputs = self(imgs)
        losses = self.compute_losses(outputs, ldmks)
        return losses, outputs, ldmks

    def _log_losses(self, losses, stage: str):
        loss, nme = losses
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_nme', nme, on_step=True, on_epoch=True, prog_bar=True, logger=True)


class CoupledSegmentationRegressor(LandmarkRegressor):
    def __init__(self, model: nn.Module, *args, mse_weight=1., **kwargs):
        super().__init__(model, *args, **kwargs)
        self.mse_weight = mse_weight
        self.bce = nn.BCELoss()
        self.val_f1 = torchmetrics.F1Score(task='binary')
        self.test_f1 = torchmetrics.F1Score(task='binary')

    def compute_losses(self, x, y):
        p_masks, p_ldmks = x
        masks, ldmks = y
        bce = self.bce(p_masks, masks)
        mse = self.mse(p_ldmks, ldmks)
        nme = self.nme(p_ldmks, ldmks)
        loss = (self.mse_weight * mse) + bce
        return loss, mse, nme, bce

    def common_step(self, batch, batch_idx):
        imgs, masks, ldmks = batch
        resolution = ldmks.new_tensor(imgs.shape[-2:]) / 2
        ldmks = ldmks.div(resolution).sub_(1.)
        masks = masks.div(255).unsqueeze(-3)
        gt = (masks, ldmks)
        outputs = self(imgs)
        losses = self.compute_losses(outputs, gt)
        return losses, outputs, gt

    def _log_losses(self, losses, stage: str):
        loss, mse, nme, bce = losses
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_nme', nme, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log(f'{stage}_bce', bce, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def update_validation_metrics(self, outputs, gt):
        masks, _ = outputs
        gt_masks, _ = gt
        self.val_f1.update(masks, gt_masks)

    def update_test_metrics(self, outputs, gt):
        masks, _ = outputs
        gt_masks, _ = gt
        self.test_f1.update(masks, gt_masks)

    def on_validation_epoch_end(self):
        super().on_validation_epoch_end()
        val_f1 = self.val_f1.compute()
        self.log('val_f1', val_f1.item())
        self.val_f1.reset()

    def on_test_epoch_end(self):
        super().on_validation_epoch_end()
        test_f1 = self.test_f1.compute()
        self.log('test_f1', test_f1.item())
        self.test_f1.reset()




