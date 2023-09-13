import argparse
import gc
import shutil
from pathlib import Path
from time import sleep

import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import OmegaConf
import pytorch_lightning as pl
import albumentations as A
from pytorch_lightning.loggers import WandbLogger

from facer.datasets.data_module import LandmarkDataModule, MasksDataModule
from facer.datasets.transforms import TO_TENSOR_TRANSFORM
from facer.models.hybrid import TightlyCoupledFaceModel, CoupledFaceModel
from facer.models.perceptron import MLPerceptron
from facer.models.unet import Unet
from facer.trainers.callbacks import checkpoint_callback, early_stop_callback
from facer.trainers.label_adapters import LandmarkAdapterWrapper, MaskAdapterWrapper
from facer.trainers.trainers import LandmarkRegressor, MaskSegmentator
from facer.trainers.utils import load_model_from_training_checkpoint
from facer.trainers.visualisation_callbacks import LandmarkLogger, MasksLogger
from facer.utils.faces import FaceIndices300W

model_type_dict = {"tight": TightlyCoupledFaceModel, "loose": CoupledFaceModel}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str)
    args = parser.parse_args()

    model_path = Path(args.model_path)
    assert model_path.exists(), f"No path to model exists: {model_path.absolute()}"
    if model_path.is_dir():
        model_path = model_path / "model-best.ckpt"

    conf = OmegaConf.load("params.yaml")
    dataset_conf = conf.dataset
    fine_tune_conf = conf.label_adaptation
    dataset_path = Path("../data/datasets") / Path(fine_tune_conf.train_dataset.directory)
    test_path = Path("../data/datasets") / Path(fine_tune_conf.test_dataset.directory)
    transform = TO_TENSOR_TRANSFORM
    transform = A.Compose(transform.transforms)

    experiment = model_path.parent.name
    experiment_name = experiment+"-masks-adaptation"
    adapted_model = load_model_from_training_checkpoint(model_path)
    masks_adapter = Unet(in_channels=1, out_channels=1, start_planes=16)

    model = MaskAdapterWrapper(adapted_model, masks_adapter)

    wandb_logger = WandbLogger(project='pmgr-face-mask-and-points', job_type='masks-adaptation')

    experiment_checkpoint_callback = checkpoint_callback(experiment_name)

    regressor = MaskSegmentator(model, **fine_tune_conf.optimizer,
                                tunned_model=experiment)

    data_module = MasksDataModule(dataset_path, test_path,
                                      batch_size=fine_tune_conf.batch_size,
                                      seed=42,
                                      transform=transform)
    data_module.setup()

    test_images, test_masks = next(iter(data_module.test_dataloader()))
    image_logger = MasksLogger((test_images, test_masks))
    callbacks = [experiment_checkpoint_callback, LearningRateMonitor(), image_logger]

    gc.collect()
    torch.cuda.empty_cache()
    sleep(1)

    trainer = pl.Trainer(check_val_every_n_epoch=2,
                         #gpus=1,
                         accelerator='gpu',
                         max_epochs=fine_tune_conf.epochs,
                         logger=wandb_logger,
                         callbacks=callbacks,
                         accumulate_grad_batches=fine_tune_conf.grad_batches
                         )

    trainer.fit(regressor, data_module)
    best_ckpt = Path(trainer.checkpoint_callback.best_model_path)
    shutil.copyfile(best_ckpt, best_ckpt.with_stem("model-best"))
    trainer.test(regressor, data_module)
    trainer.test(regressor, data_module, ckpt_path=str(best_ckpt))







