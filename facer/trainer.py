import torch
import torch.nn as nn
import pytorch_lightning as pl


class LandmarkRegressor(pl.LightningModule):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model
        self.lr = 1e-3
        self.current_epoch_training_loss = torch.zeros(1)
        self.current_epoch_training_mse = torch.zeros(1)
        self.gnll = nn.GaussianNLLLoss()
        self.mse = nn.MSELoss()

    def compute_loss(self, x, y, v):
        return self.gnll(x, y,), self.mse(x,y)

    def forward(self, x):
        return self.model(x)

    def common_step(self, batch, batch_idx):
        imgs, ldmks = batch
        resolution = ldmks.new_tensor(imgs.shape[-2:])
        ldmks = ldmks.div(resolution)
        outputs = self(imgs)
        pred = outputs[..., :-1]
        var = outputs[..., -1:].clamp_min(self.gnll.eps)
        loss, mse = self.compute_loss(pred, ldmks, var)
        return loss, mse, outputs, ldmks

    def training_step(self, batch, batch_idx):
        loss, mse, _, _ = self.common_step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss, 'mse': mse}

    def training_epoch_end(self, outs):
        self.current_epoch_training_loss = torch.stack([o["loss"] for o in outs]).mean()
        self.current_epoch_training_mse = torch.stack([o['mse'] for o in outs]).mean()

    def validation_step(self, batch, batch_idx):
        loss, mse, _, _ = self.common_step(batch, batch_idx)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return {'val_loss': loss, 'val_mse': mse}

    def validation_epoch_end(self, outs):
        avg_loss = torch.stack([o["val_loss"] for o in outs]).mean()
        avg_mse = torch.stack([o["val_mse"] for o in outs]).mean()
        self.logger.experiment.add_scalars('train and val losses',
                                           {'train': self.current_epoch_training_loss.item(), 'val': avg_loss.item()},
                                           self.current_epoch)
        self.logger.experiment.add_scalars('train and val MSE',
                                           {'train': self.current_epoch_training_mse.item(), 'val': avg_mse.item()},
                                           self.current_epoch)

    def test_step(self, batch, batch_idx):
        loss, mse, _, _ = self.common_step(batch, batch_idx)
        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_mse', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
        return [optimizer], [lr_scheduler]



