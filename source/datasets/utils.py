import cv2
import jpeg4py as jpeg
import numpy as np
from numpy.typing import NDArray


def read_image(image_path: str) -> NDArray[np.uint8]:
    try:
        image = jpeg.JPEG(image_path).decode()
    except jpeg.JPEGRuntimeError:
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image
