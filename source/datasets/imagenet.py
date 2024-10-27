import multiprocessing as mp
from pathlib import Path
import typing as ty
import warnings

import albumentations as album
import numpy as np
from numpy.typing import NDArray
import psutil
import torch
from torch.utils.data import Dataset

from .stats import IMAGE_SIZE_IN_BYTES_80_PERCENTILE, get_imagenet_class_id_from_name
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
        num_cached_images: int = 0,
        transforms: album.Compose | None = None,
    ) -> None:
        super().__init__()
        self.subset = subset
        self.root_dir = root_dir
        self.transforms = transforms

        subset_folder_path = Path(self.root_dir) / f"{self.subset}_images"
        image_paths = [path for path in subset_folder_path.iterdir() if path.name.endswith(".JPEG")]

        # Get labels. Let's store it as NumPy array to avoid issues
        # with DataLoader multiprocessing.
        self.image_labels = np.array([get_imagenet_class_id_from_name(path.name) for path in image_paths])

        # Get images.
        availalbe_ram = round(0.9 * psutil.virtual_memory().available)
        num_images_avail = round(availalbe_ram / IMAGE_SIZE_IN_BYTES_80_PERCENTILE)

        # Let's check if all images fit in RAM. If no - turn of
        # caching completely.
        self.num_cached_images = min(len(self.image_labels), num_cached_images)
        if self.num_cached_images > num_images_avail:
            message = (
                f"You requested to cache {self.num_cached_images} but there is no enough RAM for it. "
                "So no images will be cached."
            )
            warnings.warn(message, category=UserWarning, stacklevel=2)
            self.num_cached_images = 0

        self.images_in_ram_indices = set(range(self.num_cached_images))
        to_load_into_ram = image_paths[:self.num_cached_images]
        to_stay_as_paths = image_paths[self.num_cached_images:]

        with mp.Pool(mp.cpu_count() - 2) as pool:
            self.image_collection = pool.map(read_image, to_load_into_ram)
        self.image_collection.extend(to_stay_as_paths)

    def __len__(self) -> int:
        return len(self.image_collection)

    def __getitem__(self, index: int) -> DataPoint:
        image = self.image_collection[index]
        if index not in self.images_in_ram_indices:
            image = read_image(self.image_collection[index])
        label = self.image_labels[index]

        if self.transforms is not None:
            image = self.transforms(image=image)["image"]
        return {"image": image, "label": label}
