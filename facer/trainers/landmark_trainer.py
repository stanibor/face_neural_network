import torch
import torch.nn as nn
import pytorch_lightning as pl

from facer.trainers.utils import OcularNMELoss


class LandmarkRegressor(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate=1e-3, weight_decay=0):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = learning_rate
        self.weight_decay = weight_decay
        self.mse = nn.MSELoss()
        self.nme = OcularNMELoss(ocular_indices=(36, 45))

    def compute_losses(self, x, y):
        return self.mse(x, y), self.nme(x, y)

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx):
        imgs, ldmks = batch
        resolution = ldmks.new_tensor(imgs.shape[-2:])/2
        ldmks = ldmks.div(resolution).sub_(1.)
        outputs = self(imgs)
        losses = self.compute_losses(outputs, ldmks)
        return losses, outputs

    def training_step(self, batch, batch_idx):
        losses, _ = self.common_step(batch, batch_idx)
        loss, nme = losses
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_nme', nme, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'nme': nme}

    def validation_step(self, batch, batch_idx):
        losses, _ = self.common_step(batch, batch_idx)
        loss, nme = losses
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_nme', nme, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_nme': nme}

    def test_step(self, batch, batch_idx):
        losses, _ = self.common_step(batch, batch_idx)
        loss, nme = losses
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_nme', nme, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'nme': nme}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer], [lr_scheduler]

    # def configure_optimizers(self):
    #     optimizer = torch.optim.AdamW([{"params": self.model.backbone.parameters(), "lr": self.lr*1e-1},
    #                                    {"params": self.model.connector.parameters()},
    #                                    {"params": self.model.regressor.parameters()}], lr=self.lr)
    #     # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    #     return [optimizer]#, [lr_scheduler]



