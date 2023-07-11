import json
import os
from pathlib import Path
import typing as ty

import addict
import albumentations as album
import jpeg4py as jpeg
import numpy as np
from numpy.typing import NDArray
from source.utils import get_albumentation_augs
import torch
from torch.utils.data import DataLoader, Dataset

from .utils import DatasetMode, fix_worker_seeds, read_image


class DataPoint(ty.TypedDict):
    image: NDArray[np.uint8] | torch.Tensor
    label: int


class ImageNetDataset(Dataset):
    def __init__(
            self, config: addict.Dict, mode: DatasetMode = DatasetMode.TRAIN, transforms: album.Compose | None = None
    ) -> None:
        super().__init__()
        self.mode = mode
        self.root_dir = Path(config.path.dataset.root_dir) / f"{self.mode}"
        with Path(config.path.dataset.meta_file_path).open() as jfs:
            self.class_mapper = json.load(jfs)
        self.classname2index = {classname: index for index, classname in enumerate(self.class_mapper.keys())}
        self.index2classname = {index: classname for classname, index in self.classname2index.items()}
        self.image_names = os.listdir(self.root_dir)
        self.transforms = transforms

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, index: int) -> DataPoint:
        try:
            image_name = self.image_names[index]
            image = self._get_image(index)
        except jpeg.JPEGRuntimeError:
            image_name = self.image_names[index]
            image = self._get_image(index - 1)
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        label = self.get_classidx_from_classname(self.get_classname_from_filename(image_name))
        return {"image": image, "label": label}

    def _get_image(self, index: int) -> NDArray[np.uint8]:
        image_name = self.image_names[index]
        image_path = self.root_dir / image_name
        return read_image(image_path.as_posix())

    def get_classname_from_filename(self, filename: str) -> str:
        return filename.split("_")[-1].split(".")[0]

    def get_classidx_from_classname(self, classname: str) -> int:
        return self.classname2index[classname]


def build_dataloaders(config: addict.Dict) -> dict[str, DataLoader]:
    augs = get_albumentation_augs(config)
    dataloaders = {}
    for subset_name in augs:
        dataset = ImageNetDataset(config, mode=DatasetMode(subset_name), transforms=augs[subset_name])
        dataloaders[subset_name] = DataLoader(
            dataset=dataset,
            batch_size=config.training.batch_size,
            shuffle=subset_name == "train",
            pin_memory=torch.cuda.is_available(),
            num_workers=config.training.dataloader_num_workers,
            worker_init_fn=fix_worker_seeds
        )
    return dataloaders
