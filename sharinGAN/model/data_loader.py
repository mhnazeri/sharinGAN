from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset

from utils import get_conf


class SharinganDataset(Dataset):
    def __init__(self, transform) -> None:
        # get data directory
        data_dir = get_conf("conf/dirs").train_data
        # store filenames
        self.filenames = list(Path(data_dir).iterdir())
        self.transform = transform

    def __len__(self) -> int:
        """return size of dataset"""
        return len(self.filenames)

    def __getitem__(self, idx) -> torch.tensor:
        image = Image.open(self.filenames[idx])
        image = self.transform(image)
        return image


if __name__ == "__main__":
    dataset = SharinganDataset(None)
    print(len(dataset))