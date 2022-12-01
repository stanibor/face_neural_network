from pathlib import Path
from fire import Fire
from tqdm.auto import tqdm


def reorganize(directory: Path,
               output_directory: Path = None,
               image_subdir: Path = Path('images'),
               mask_subdir: Path = Path('masks'),
               landmark_subdir: Path = Path('landmarks')):

    _directory = Path(directory)
    _output_directory = _directory if output_directory is None else Path(output_directory)

    image_dir = _output_directory / image_subdir
    mask_dir = _output_directory / mask_subdir
    landmark_dir = _output_directory / landmark_subdir

    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    landmark_dir.mkdir(parents=True, exist_ok=True)

    for image_file in tqdm(list(_directory.glob("*[0-9].png"))):
        mask_file = image_file.with_stem(image_file.stem + '_seg')
        ldmks_file = image_file.with_name(image_file.stem + '_ldmks.txt')

        image_file.rename(image_dir / image_file.name)
        mask_file.rename(mask_dir / image_file.name)
        ldmks_file.rename(landmark_dir / image_file.with_suffix('.txt').name)


if __name__ == "__main__":
    Fire(reorganize)



