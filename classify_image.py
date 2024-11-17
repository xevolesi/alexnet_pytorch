import argparse as ap

import numpy as np
from numpy.typing import NDArray
from preprocess_imagenet import center_crop, resize_smallest_max_size
from source.datasets.stats import IMAGENET2012_CLASS_ID_TO_CLASS, IMAGENET2012_CLASSES, RGB_MEAN
from source.datasets.utils import read_image
from source.models import AlexNet
import torch


def tensorize_numpy_image(image: NDArray[np.uint8], device: torch.device) -> torch.Tensor:
    image = image.astype(np.float32)
    rgb_mean_uint = (255 * np.array(RGB_MEAN)).reshape(1, 1, -1)
    image -= rgb_mean_uint
    image = np.expand_dims(image.transpose(2, 0, 1), axis=0)
    return torch.from_numpy(image).to(device)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-i", "--image_path", required=True, type=str, help="Path to image to be classified")
    parser.add_argument("-w", "--weights_path", required=True, type=str, help="Path to model weights to use")
    parser.add_argument("-d", "--device", required=False, type=str, default="cpu", help="PyTorch device")
    parser.add_argument("-k", "--topk", required=False, default=1, type=int, help="Top-K results of classification")
    args = parser.parse_args()

    device = torch.device(args.device)

    # Prepare image.
    image = read_image(args.image_path)
    crop = center_crop(resize_smallest_max_size(image, 256), 224, 224)
    crop_tensor = tensorize_numpy_image(crop, device)

    # Prepare model.
    model_state_dict = torch.load(args.weights_path, weights_only=True)
    model = AlexNet()
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()

    # Do inference.
    with torch.no_grad():
        logits = model(crop_tensor).detach().cpu()
        probas = logits.softmax(dim=1)
        topk_class_indices = torch.argsort(probas, descending=True)[0, :args.topk].tolist()
        topk_class_probas = probas[:, topk_class_indices][0].tolist()

    # Print classes.
    print(f"Classes for {args.image_path}") # noqa: T201
    for class_id, class_proba in zip(topk_class_indices, topk_class_probas):
        class_name = IMAGENET2012_CLASSES[IMAGENET2012_CLASS_ID_TO_CLASS[class_id]]
        print(f"'{class_name}' with proba: {class_proba:.5f}") # noqa: T201




