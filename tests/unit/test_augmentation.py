from copy import deepcopy
import pydoc

import pytest
import albumentations as album
from albumentations.pytorch.transforms import ToTensorV2
from source.utils.augmentation import get_albumentation_augs, _IMAGENET_MEAN, _IMAGENET_STD


def _assert_default_transform_set(subset_transforms):
    assert len(subset_transforms) == 2
    for transform_idx, transform in enumerate(subset_transforms):
        if transform_idx == 0:
            assert isinstance(transform, album.Normalize)
            assert transform.mean == _IMAGENET_MEAN
            assert transform.std == _IMAGENET_STD
        elif transform_idx == 1:
            assert isinstance(transform, ToTensorV2)
        else:
            assert False


def _assert_augs_from_config(config, subset_name, actual_augs):
    config_augs = config.augmentations[subset_name]
    assert config_augs["transform"]["__class_fullname__"] == "Compose"
    config_augs = config_augs["transform"]["transforms"]
    assert len(config_augs) == len(actual_augs)
    for config_aug, actual_aug in zip(config_augs, actual_augs):
        assert isinstance(actual_aug, pydoc.locate(config_aug["__class_fullname__"]))


@pytest.mark.parametrize("default", [True, False])
def test_get_albumentation_augs(default, get_test_config):
    config = deepcopy(get_test_config)
    if default:
        config.augmentations = {}
    augs = get_albumentation_augs(config)

    if default:
        assert set(augs.keys()) == set(("train", "val", "test")) 
        for subset_name in augs:
            _assert_default_transform_set(augs[subset_name])
    else:
        assert set(augs.keys()) == set(config.augmentations.keys())
        for subset_name in augs:
            _assert_augs_from_config(config, subset_name, augs[subset_name])
