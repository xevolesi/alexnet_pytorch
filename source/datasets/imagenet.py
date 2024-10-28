import os
import typing as ty

import addict
import albumentations as album
import numpy as np
from numpy.typing import NDArray
from source.utils.augmentation import get_albumentation_augs
import torch
from torch.utils.data import DataLoader, Dataset

from .stats import get_imagenet_class_id_from_name
from .utils import read_image

ImageT: ty.TypeAlias = NDArray[np.uint8] | torch.Tensor

class DataPoint(ty.TypedDict):
    image: ImageT
    label: int


class ImageNetDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        subset: ty.Literal["train", "val"],
        transforms: album.Compose | None = None,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.root_dir = root_dir
        self.transforms = transforms

        subset_folder_path = os.path.join(self.root_dir, f"{self.subset}_images")
        image_paths = [file.path for file in os.scandir(subset_folder_path) if file.name.endswith(".JPEG")]

        # Get labels. Let's store it as NumPy array to avoid issues
        # with DataLoader multiprocessing.
        self.image_labels = np.array([get_imagenet_class_id_from_name(os.path.split(path)[-1]) for path in image_paths])

        # Get images.
        self.image_collection = np.array(image_paths)

    def __len__(self) -> int:
        return len(self.image_labels)

    def __getitem__(self, index: int) -> DataPoint:
        image = read_image(self.image_collection[index])
        label = self.image_labels[index]

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return {"image": image, "label": label}


def build_dataloaders(config: addict.Dict) -> dict[str, DataLoader[ImageNetDataset]]:
    dataloaders = {}
    transforms = get_albumentation_augs(config)
    for subset, subset_transforms in transforms.items():
        dataset = ImageNetDataset(
            config.path.dataset_root_dir,
            subset=subset,
            transforms=subset_transforms,
        )
        dataloaders[subset] = DataLoader(
            dataset,
            batch_size=config.training.batch_size,
            shuffle=subset == "train",
            pin_memory=True,
            num_workers=config.training.dataloader_num_workers,
        )
    return dataloaders
