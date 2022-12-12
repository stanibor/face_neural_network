from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from omegaconf import OmegaConf
import pytorch_lightning as pl
import albumentations as A
from pytorch_lightning.loggers import WandbLogger

from facer.datasets.face_datasets import LandmarkLocalizationDataset
from facer.models.face_model import LandmarkRegressionModel, PyramidRegressionModel
from facer.trainers.callbacks import checkpoint_callback, early_stop_callback
from facer.trainers.landmark_trainer import LandmarkRegressor

if __name__ == "__main__":
    conf = OmegaConf.load("params.yaml")
    dataset_conf = conf.dataset
    train_conf = conf.training
    transform = A.load("transforms/transform.json")
    dataset_path = Path("../data/datasets") / dataset_conf.directory
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    transform = A.Compose(transform.transforms, keypoint_params=keypoint_params)

    model = PyramidRegressionModel(output_shape=(70, 3), pool_size=train_conf.landmarks.pool_size)

    wandb_logger = WandbLogger(project='wandb-landmark-regression', job_type='train')

    regressor = LandmarkRegressor(model)
    regressor.lr = train_conf.optimizer.lr
    callbacks = [checkpoint_callback, early_stop_callback]

    dataset = LandmarkLocalizationDataset(directory=dataset_path, transform=transform)
    train_len = int(0.9 * len(dataset))
    valid_len = len(dataset) - train_len
    generator = torch.Generator().manual_seed(42)
    train_dataset, valid_dataset = random_split(dataset, (train_len, valid_len), generator=generator)

    train_loader = DataLoader(train_dataset,
                              batch_size=train_conf.batch_size,
                              num_workers=4,
                              pin_memory=True,
                              shuffle=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=train_conf.batch_size,
                              num_workers=4,
                              pin_memory=True,
                              shuffle=False)

    trainer = pl.Trainer(check_val_every_n_epoch=2,
                         gpus=1,
                         max_epochs=train_conf.epochs,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         )

    trainer.fit(regressor, train_loader, valid_loader)




