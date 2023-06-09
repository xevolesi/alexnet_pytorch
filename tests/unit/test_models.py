from copy import deepcopy
from itertools import product

import pytest
from source.models import AlexNet
import torch


@pytest.mark.parametrize(("n_classes", "in_channels"), product((1, 2, 1000), (3, 4, 9)))
def test_model(n_classes, in_channels, get_test_config):
    config = deepcopy(get_test_config)
    config.model.n_classes = n_classes
    config.model.in_channels = in_channels
    model = AlexNet(**config.model)
    model.eval()

    assert model.block1[0].in_channels == config.model.in_channels
    assert model.head[-1].out_features == config.model.n_classes

    bulk_image_batch = torch.randn(
        (config.training.batch_size, config.model.in_channels, config.training.image_size, config.training.image_size)
    )
    with torch.no_grad():
        output = model(bulk_image_batch)
    assert output.shape == (config.training.batch_size, config.model.n_classes)

