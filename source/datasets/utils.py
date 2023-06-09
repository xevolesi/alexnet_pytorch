from enum import Enum

import jpeg4py as jpeg
import numpy as np
from numpy.typing import NDArray
from source.utils.general import reseed


class DatasetMode(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


def fix_worker_seeds(worker_id: int) -> None:
    """Fix seeds inside single worker."""
    seed = np.random.default_rng().bit_generator.state["state"]["state"] + worker_id
    reseed(seed)


def read_image(image_path: str) -> NDArray[np.uint8] | None:
    return jpeg.JPEG(image_path).decode()
