r"""This script contains preprocessing steps for ImageNet according to
AlexNet paper. In the paper on page 2, section 2 authors said: "ImageNet
consists of variable-resolution images, while our system requires a
constant input dimensionality. Therefore, we down-sampled the images to a
fixed resolution of (256, 256). Given a rectangular image, we first rescaled
the image such that the shorter side was of length 256, and then cropped
out the central (256, 256) patch from the resulting image."

So, they did the following steps:
1) Rescale images such that the shortest side of the image is equal to 256
   and other side is being kept so the aspect ratio isn't changed;
2) Center crop image from step 1.

Since these steps were applied to train and validation subsets of the ImageNet
it's quite beneficial to do them before actual training to save minutes per epoch.

This scripts performs such steps using Python's multiprocessing module to speed
up preprocessing.

The usage is quite simple. Just pass source directory with images and destination
directory for processed images:

```
python preprocess_imagenet.py -s %path_to_source_dir_with_images% -d %path_to_destination_dir%
```

I used this script as follows:

```
# For train images.
python preprocess_imagenet.py -s D:\\datasets\\imagenet\\train_images -d D:\\datasets\\imagenet_processed_256\\train_images

# For val images.
python preprocess_imagenet.py -s D:\\datasets\\imagenet\\val_images -d D:\\datasets\\imagenet_processed_256\\val_images
```

On my setups it saves ~2 minutes per epoch. So, if we train 90 epochs like authors in paper we will totally save
180 minutes which is quite good.
"""
import argparse as ap
from functools import partial
import multiprocessing as mp
import os

import cv2
import numpy as np
from numpy.typing import NDArray
from source.datasets.utils import read_image


def center_crop(image: NDArray[np.uint8], crop_height: int = 256, crop_width: int = 256) -> NDArray[np.uint8]:
    image_height, image_width = image.shape[:2]
    center_x, center_y = image_width // 2, image_height // 2
    x_slice = slice(center_x - crop_width // 2, center_x + crop_width // 2)
    y_slice = slice(center_y - crop_height // 2, center_y + crop_height //2)
    return image[y_slice, x_slice, :]


def resize_smallest_max_size(image: NDArray[np.uint8], smalles_max_size: int) -> NDArray[np.uint8]:
    image_height, image_width = image.shape[:2]
    min_size = min(image_height, image_width)
    ratio = smalles_max_size / min_size
    new_height = round(image_height * ratio)
    new_width = round(image_width * ratio)
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_CUBIC)


def process_image(
    src_image_path: str,
    dst_dir_path: str,
    smalles_max_size: int = 256,
    apply_center_crop: bool = False,
) -> None:
    image = read_image(src_image_path)
    image = resize_smallest_max_size(image, smalles_max_size)
    if apply_center_crop:
        image = center_crop(image, smalles_max_size, smalles_max_size)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _, image_name = os.path.split(src_image_path)
    target_path = os.path.join(dst_dir_path, image_name)
    cv2.imwrite(target_path, image)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-s", "--src_path", required=True, type=str)
    parser.add_argument("-d", "--dst_path", required=True, type=str)
    parser.add_argument("-z", "--smallest_max_size", required=False, type=int, default=256)
    parser.add_argument("-c", "--center_crop", required=False, type=bool, default=False, action=ap.BooleanOptionalAction)
    args = parser.parse_args()

    os.makedirs(args.dst_path, exist_ok=True)
    process_image_parallel = partial(
        process_image,
        dst_dir_path=args.dst_path,
        smallest_max_size=args.smallest_max_size,
        apply_center_crop=args.center_crop,
    )
    image_paths = [image_file.path for image_file in os.scandir(args.src_path)]
    with mp.Pool(processes=mp.cpu_count()) as pool:
        _ = pool.map(process_image_parallel, image_paths)
