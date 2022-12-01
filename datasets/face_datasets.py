import numpy as np
import torch
import torch.utils.data as data
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as TF


class ImageDataset(data.Dataset):
    def __init__(self, root: Path, img_subdir: Path = Path('images')):
        super().__init__()
        self.root = Path(root)
        self.img_files = [Path(img) for img in (self.root / img_subdir).glob('*.png')]

    def __getitem__(self, index):
        img_path = self.img_files[index]
        data = Image.open(img_path)
        return TF.to_tensor(data)

    def __len__(self):
        return len(self.img_files)


class SegmentationDataset(ImageDataset):
    def __init__(self, root: Path, img_subdir: Path = Path('images'), mask_subdir: Path = Path('masks')):
        super().__init__(root, img_subdir)
        self.mask_files = [self.root / mask_subdir / img.name for img in self.img_files]
        assert len(self.mask_files) == len(self.img_files)

    def _get_mask(self, index):
        mask_path = self.mask_files[index]
        mask = Image.open(mask_path)
        return torch.tensor(mask.getdata(), dtype=torch.uint8).view(1, *mask.size)

    def __getitem__(self, index):
        image = super().__getitem__(index)
        mask = self._get_mask(index)
        return image, mask


class LandmarkLocalizationDataset(ImageDataset):
    def __init__(self, root: Path, img_subdir: Path = Path('images'), landmark_subdir: Path = Path('landmarks')):
        super().__init__(root, img_subdir)
        self.landmark_files = [(self.root / landmark_subdir / img.name).with_suffix(".txt") for img in self.img_files]
        assert len(self.landmark_files) == len(self.img_files)

    def _get_landmarks(self, index):
        ldmk_path = self.landmark_files[index]
        return torch.from_numpy(np.genfromtxt(ldmk_path, dtype='float32'))

    def __getitem__(self, index):
        image = super().__getitem__(index)
        landmarks = self._get_landmarks(index)
        return image, landmarks


class SegmentationAndLandmarkDataset(SegmentationDataset, LandmarkLocalizationDataset):
    def __init__(self,
                 root: Path,
                 img_subdir: Path = Path('images'),
                 mask_subdir: Path = Path('masks'),
                 landmark_subdir: Path = Path('landmarks')):
        SegmentationDataset.__init__(self, root, img_subdir, mask_subdir)
        LandmarkLocalizationDataset.__init__(self, root, img_subdir, landmark_subdir)

    def __getitem__(self, index):
        image_n_landmarks, mask = super().__getitem__(index)
        image, landmarks = image_n_landmarks
        return image, mask, landmarks


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from tqdm.auto import tqdm

    # dataset = SegmentationAndLandmarkDataset("/data/Datasets/dataset_100000")
    dataset = ImageDataset("/data/Datasets/dataset_fg", img_subdir=".")
    loader = DataLoader(dataset, batch_size=16, pin_memory=True, num_workers=4)

    for batch in tqdm(loader):
        continue

