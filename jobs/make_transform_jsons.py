from os import PathLike
from pathlib import Path
from fire import Fire

import albumentations as A
from facer.datasets.transforms import spatial_transforms, photometric_transforms, TO_TENSOR_TRANSFORM


def make_transform_json(crop_size, output_size, output: PathLike[str], photometric_json: PathLike[str] = None):
    s_transform = spatial_transforms((output_size, crop_size), (output_size, output_size))
    p_transform = photometric_transforms() if photometric_json is None else A.load(photometric_json)
    transform = A.Compose([*s_transform, *p_transform, *TO_TENSOR_TRANSFORM])
    A.save(transform, output)
    if photometric_json is None:
        A.save(p_transform, Path(output).parent / "photometric.json")


if __name__ == "__main__":
    Fire(make_transform_json)
