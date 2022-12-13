from pathlib import Path

from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import OmegaConf
import pytorch_lightning as pl
import albumentations as A
from pytorch_lightning.loggers import WandbLogger

from facer.datasets.data_module import MasksAndLandmarksDataModule
from facer.models.backbone import resnet_by_name
from facer.models.hybrid import TightlyCoupledFaceModel
from facer.trainers.callbacks import checkpoint_callback, early_stop_callback
from facer.trainers.trainers import CoupledSegmentationRegressor
from facer.trainers.visualisation_callbacks import FaceImagesLogger

if __name__ == "__main__":
    conf = OmegaConf.load("params.yaml")
    dataset_conf = conf.dataset
    train_conf = conf.training
    transform = A.load("transforms/transform.json")
    dataset_path = Path("../data/datasets") / dataset_conf.directory
    test_path = Path("../data/datasets") / conf.dataset.test.name
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    transform = A.Compose(transform.transforms, keypoint_params=keypoint_params)

    backbone = resnet_by_name(**train_conf.model.backbone)

    model = TightlyCoupledFaceModel(backbone=backbone, output_shape=(70, 2), **train_conf.model.params)

    wandb_logger = WandbLogger(project='wandb-face-mask-and-points', job_type='train')

    regressor = CoupledSegmentationRegressor(model, **train_conf.optimizer)
    callbacks = [checkpoint_callback, early_stop_callback, LearningRateMonitor()]

    data_module = MasksAndLandmarksDataModule(dataset_path, batch_size=train_conf.batch_size, seed=42)
    data_module.setup()

    trainer = pl.Trainer(check_val_every_n_epoch=2,
                         gpus=1,
                         max_epochs=train_conf.epochs,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         accumulate_grad_batches=train_conf.grad_batches
                         )

    trainer.fit(regressor, data_module)
    trainer.test(regressor, data_module)




