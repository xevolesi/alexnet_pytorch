import typing as ty

import addict
import albumentations as album


class TransformTypedDict(ty.TypedDict):
    train: album.Compose | None
    val: album.Compose | None
    test: album.Compose | None


def get_albumentation_augs(config: addict.Dict) -> TransformTypedDict:
    """Build albumentations's augmentation pipelines from configuration file."""
    return TransformTypedDict(**{subset: album.from_dict(config.augmentations[subset]) for subset in config.augmentations})
