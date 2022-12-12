import torch
import torch.nn as nn
import pytorch_lightning as pl


class LandmarkRegressor(pl.LightningModule):
    def __init__(self, model: nn.Module, learning_rate = 1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.lr = learning_rate
        self.current_epoch_training_losses = torch.zeros(2)
        self.gnll = nn.GaussianNLLLoss()
        self.mse = nn.MSELoss()

    def compute_losses(self, x, y, v):
        return self.gnll(x, y, v), self.mse(x,y)

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx):
        imgs, ldmks = batch
        resolution = ldmks.new_tensor(imgs.shape[-2:])
        ldmks = ldmks.div(resolution)
        outputs = self(imgs)
        pred = outputs[..., :-1]
        var = outputs[..., -1:].clamp_min(self.gnll.eps)
        losses = self.compute_losses(pred, ldmks, var)
        return losses, outputs

    def training_step(self, batch, batch_idx):
        losses, _ = self.common_step(batch, batch_idx)
        loss, mse = losses
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'mse': mse}

    def validation_step(self, batch, batch_idx):
        losses, _ = self.common_step(batch, batch_idx)
        loss, mse = losses
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_mse': mse}

    def test_step(self, batch, batch_idx):
        losses, _ = self.common_step(batch, batch_idx)
        loss, mse = losses
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW([{"params": self.model.backbone.parameters(), "lr": self.lr*1e-1},
                                       {"params": self.model.regressor.parameters()}], lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer], [lr_scheduler]



