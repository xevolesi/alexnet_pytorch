import torch
from source.datasets import ImageNetDataset
from source.utils.augmentation import get_albumentation_augs
from source.utils.general import read_config
from torch.utils.data import DataLoader
from source.models import AlexNet


def main():
    config = read_config("config.yml")
    
    device = torch.device(config.training.device)
    transforms = get_albumentation_augs(config)


    val_dataset = ImageNetDataset(
        config.path.dataset_root_dir, subset="val", num_cached_images=50_000, transforms=transforms["val"]
    )
    val_dataloader = DataLoader(val_dataset, config.training.batch_size, shuffle=False, pin_memory=True, num_workers=config.training.dataloader_num_workers)
    train_dataset = ImageNetDataset(
        config.path.dataset_root_dir, subset="train", num_cached_images=70_000, transforms=transforms["train"]
    )
    train_dataloader = DataLoader(train_dataset, config.training.batch_size, shuffle=True, pin_memory=True, num_workers=config.training.dataloader_num_workers)


if __name__ == "__main__":
    main()
