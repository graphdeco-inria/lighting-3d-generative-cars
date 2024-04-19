import os
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.transforms import Compose, ToTensor

Image.init()
def is_image_ext(filename: str):
   ext = str(filename).split('.')[-1].lower()
   return f'.{ext}' in Image.EXTENSION 


class PathDataset(Dataset):
    def __init__(self, source_dir):
        super().__init__()
        self._image_paths = self._get_image_paths(source_dir)
        self.transform = Compose([ToTensor()])

    def _get_image_paths(self, source_dir):
        paths = [str(f) for f in Path(source_dir).rglob('*') if is_image_ext(f) and os.path.isfile(f)]
        if not len(paths) > 0:
            raise ValueError(f"No images found in {source_dir}")
        return paths

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image = self.transform(Image.open(self._image_paths[idx]))
        return image

class PathDataset2(Dataset):
    def __init__(self, source_dir):
        super().__init__()
        self._image_paths = self._get_image_paths(source_dir)

    def _get_image_paths(self, source_dir):
        paths = sorted([str(f) for f in Path(source_dir).rglob('*') if is_image_ext(f) and os.path.isfile(f)])
        if not len(paths) > 0:
            raise ValueError(f"No images found in {source_dir}")
        return paths

    def __len__(self):
        return len(self._image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self._image_paths[idx]))
        return image, self._image_paths[idx]
            