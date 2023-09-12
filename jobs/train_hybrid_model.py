import shutil
from pathlib import Path

from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import OmegaConf
import pytorch_lightning as pl
import albumentations as A
from pytorch_lightning.loggers import WandbLogger

from facer.datasets.data_module import MasksAndLandmarksDataModule
from facer.datasets.transforms import TO_TENSOR_TRANSFORM
from facer.models.backbone import resnet_by_name
from facer.models.hybrid import TightlyCoupledFaceModel, CoupledFaceModel
from facer.trainers.callbacks import checkpoint_callback, early_stop_callback
from facer.trainers.trainers import CoupledSegmentationRegressor
from facer.trainers.visualisation_callbacks import FaceImagesLogger
from facer.utils.faces import FaceIndices300W

model_type_dict = {"tight": TightlyCoupledFaceModel, "loose": CoupledFaceModel}

if __name__ == "__main__":
    conf = OmegaConf.load("params.yaml")
    dataset_conf = conf.dataset
    train_conf = conf.training
    dataset_path = Path("../data/datasets") / dataset_conf.directory
    test_path = Path("../data/datasets") / Path(train_conf.test_dataset.directory)
    transform = A.load("transforms/transform.json")
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    transform = A.Compose(transform.transforms, keypoint_params=keypoint_params)

    backbone = resnet_by_name(**train_conf.model.backbone)
    model_type = model_type_dict[train_conf.model.type]
    model = model_type(backbone=backbone, output_shape=(68, 2), **train_conf.model.params)

    wandb_logger = WandbLogger(project='pmgr-face-mask-and-points', job_type='train')

    experiment_checkpoint_callback = checkpoint_callback(wandb_logger.experiment.name)

    regressor = CoupledSegmentationRegressor(model, **train_conf.optimizer,
                                             backbone_=train_conf.model.backbone.name,
                                             model_type=model_type,
                                             **train_conf.model.params,
                                             )

    data_module = MasksAndLandmarksDataModule(dataset_path, test_path,
                                              batch_size=train_conf.batch_size,
                                              seed=42,
                                              transform=transform)
    data_module.setup()

    # val_images, _, val_landmarks = next(iter(data_module.val_dataloader()))
    test_images, _, test_landmarks = next(iter(data_module.test_dataloader()))
    image_logger = FaceImagesLogger((test_images, test_landmarks), connectivity=FaceIndices300W.connectivity)

    data_module.dataset_val.transform = A.Compose([A.Resize(*test_images.shape[:-2]), *TO_TENSOR_TRANSFORM])

    callbacks = [experiment_checkpoint_callback, early_stop_callback, LearningRateMonitor(), image_logger]
    trainer = pl.Trainer(check_val_every_n_epoch=2,
                         #gpus=1,
                         accelerator='gpu',
                         max_epochs=train_conf.epochs,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         accumulate_grad_batches=train_conf.grad_batches
                         )

    trainer.fit(regressor, data_module)
    best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
    shutil.copyfile(best_ckpt, best_ckpt.with_stem("model-best"))
    trainer.test(regressor, data_module)
    trainer.test(regressor, data_module, ckpt_path=str(best_ckpt))







