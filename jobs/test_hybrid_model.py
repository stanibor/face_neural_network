import argparse
import shutil
from pathlib import Path

from pytorch_lightning.callbacks import LearningRateMonitor
from omegaconf import OmegaConf
import pytorch_lightning as pl
import albumentations as A
from pytorch_lightning.loggers import WandbLogger

from facer.datasets.data_module import MasksAndLandmarksDataModule
from facer.datasets.transforms import TO_TENSOR_TRANSFORM
from facer.models.hybrid import TightlyCoupledFaceModel, CoupledFaceModel
from facer.trainers.callbacks import checkpoint_callback, early_stop_callback
from facer.trainers.trainers import CoupledSegmentationRegressor
from facer.trainers.utils import load_model_from_training_checkpoint
from facer.trainers.visualisation_callbacks import FaceImagesLogger
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
    train_conf = conf.training
    dataset_path = Path("../data/datasets") / dataset_conf.directory
    test_path = Path("../data/datasets") / Path(train_conf.test_dataset.directory)
    transform = A.load("transforms/transform.json")
    keypoint_params = A.KeypointParams(format="xy", remove_invisible=False)
    transform = A.Compose(transform.transforms, keypoint_params=keypoint_params)

    experiment = model_path.parent.name
    experiment_name = experiment+"-tested"
    model = load_model_from_training_checkpoint(model_path)

    model.eval()

    regressor = CoupledSegmentationRegressor(model, **train_conf.optimizer,
                                             tunned_model=experiment)

    data_module = MasksAndLandmarksDataModule(dataset_path, test_path,
                                              batch_size=train_conf.batch_size,
                                              seed=42,
                                              transform=transform)
    data_module.setup()

    trainer = pl.Trainer(check_val_every_n_epoch=2,
                         #gpus=1,
                         accelerator='gpu',
                         max_epochs=train_conf.epochs,
                         )

    trainer.test(regressor, data_module)







