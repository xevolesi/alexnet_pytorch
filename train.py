import argparse as ap
from collections import defaultdict
import os
import time

from coolname import generate_slug
from loguru import logger
from source.datasets import ImageNetDataset, build_dataloaders
from source.metrics import calculate_batch_top_k_error_rate
from source.utils.general import (
    get_cpu_state_dict,
    get_object_from_dict,
    read_config,
    seed_everything,
    tensor_dict_to_float_dict,
)
from source.utils.tb_logger import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torchvision.transforms.functional import ten_crop

# Some additional settings.
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
os.environ["ALBUMENTATIONS_DISABLE_VERSION_CHECK"] = "1"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = True
torch.backends.cuda.matmul.allow_tf32 = True


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[ImageNetDataset],
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
) -> tuple[dict[str, float], dict[str, torch.Tensor]]:
    model.train()
    running_losses = defaultdict(lambda: torch.as_tensor(0.0, device=device))
    grad_dict = {name: torch.zeros_like(param, device=device) for name, param in model.named_parameters()}
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        outputs = model(images)
        loss: torch.Tensor = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # Track avg. of the gradients.
        for param_name, param_grad in model.named_parameters():
            grad_dict[param_name] += param_grad

        optimizer.zero_grad(set_to_none=True)
        running_losses["train_loss"] += loss

    grad_dict = {name: grad * 1/len(dataloader) for name, grad in grad_dict.items()}
    return tensor_dict_to_float_dict(running_losses, 1/len(dataloader)), grad_dict


@torch.no_grad()
def validate_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader[ImageNetDataset],
    criterion: torch.nn.modules.loss._Loss,
    device: torch.device,
    subset: str = "val",
) -> tuple[dict[str, float], dict[str, float]]:
    model.eval()
    running_losses = defaultdict(lambda: torch.as_tensor(0.0, device=device))
    running_metrics = defaultdict(lambda: torch.as_tensor(0.0, device=device))
    for batch in dataloader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        # Do ten crop validation on batch as in paper.
        crops = torch.stack(ten_crop(images, size=(224, 224)))
        n_crops, batch_size, channels, height, width = crops.size()
        outputs = model(crops.view(-1, channels, height, width)).view(n_crops, batch_size, -1).mean(dim=0)

        running_losses[f"{subset}_loss"] += criterion(outputs, labels)
        running_metrics[f"{subset}_error_rate@1"] += calculate_batch_top_k_error_rate(outputs, labels, k=1)
        running_metrics[f"{subset}_error_rate@5"] += calculate_batch_top_k_error_rate(outputs, labels, k=5)

    running_losses = tensor_dict_to_float_dict(running_losses, 1/len(dataloader))
    running_metrics = tensor_dict_to_float_dict(running_metrics, 1/len(dataloader))

    return running_losses, running_metrics


def main(args: ap.Namespace) -> None:
    config = read_config(args.config_path)
    seed_everything(config)

    # Dir for weights.
    run_name = generate_slug(2)
    weights_path = os.path.join(config.path.checkpoint_save_path, run_name)
    os.makedirs(weights_path, exist_ok=True)
    logger.info("Run name: {rn}", rn=run_name)
    logger.info("Created weights path: {wp}", wp=weights_path)

    dataloaders = build_dataloaders(config)
    logger.info(
        "Created dataloaders: {dls}",
        dls={subset: (len(dataloader.dataset), len(dataloader)) for subset, dataloader in dataloaders.items()}
    )

    device = torch.device(config.training.device)
    criterion: torch.nn.modules.loss._Loss = get_object_from_dict(config.criterion)
    model: torch.nn.Module = get_object_from_dict(config.model).to(device)
    optimizer: torch.optim.Optimizer = get_object_from_dict(config.optimizer, params=model.parameters())
    scheduler: torch.optim.lr_scheduler.LRScheduler = get_object_from_dict(config.scheduler, optimizer=optimizer)

    start_epoch = 0
    if config.training.start_from_this_ckpt is not None:
        full_checkpoint = torch.load(config.training.start_from_this_ckpt, weights_only=True)
        start_epoch = 16
        model.load_state_dict(full_checkpoint["model"])
        optimizer.load_state_dict(full_checkpoint["optimizer"])
        scheduler.load_state_dict(full_checkpoint["scheduler"])
        logger.info("Loading checkpoint from {sp}", sp=config.training.start_from_this_ckpt)

    tb_logger = TensorBoardLogger(
        log_dir=os.path.join(config.path.tb_log_dir, run_name),
        init_model=model,
    )
    logger.info("Initialized training ingredients")

    best_model_weights = None
    best_metric = float("inf")
    for epoch in range(start_epoch, config.training.n_epochs):
        epoch_start = time.perf_counter()
        training_losses, gradient_dict = train_one_epoch(model, dataloaders["train"], optimizer, criterion, device)
        training_end = time.perf_counter()
        validation_losses, validation_metrics = validate_one_epoch(model, dataloaders["val"], criterion, device, subset="val")
        validation_end = time.perf_counter()

        # Authors decreased LR if validation error rate is not
        # improving. They did it manually but i'll hope that
        # automation is okay here. :)
        # Didn't get which one of validation error rate is it.
        # Was it top-1 error rate or top-5 error rate?
        scheduler.step(metrics=validation_metrics["val_error_rate@5"])

        # Log results to TB.
        tb_logger.log(
            epoch=epoch,
            metric_dict=validation_metrics,
            losses_dict=validation_losses | training_losses,
            grad_dict=gradient_dict,
            optimizer=optimizer,
            optimizer_param_names=("lr", "weight_decay"),
            model=model,
        )

        # Save weights every N epochs.
        if config.training.save_every_epoch is not None and (epoch + 1) % config.training.save_every_epoch == 0:
            model_weights = get_cpu_state_dict(model)
            full_checkpoint = {
                "epoch": epoch,
                "model": model_weights,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            save_name = f"alexnet_{epoch}_{round(validation_metrics['val_error_rate@5'], 4)}_full.pt"
            save_path = os.path.join(weights_path, save_name)
            torch.save(full_checkpoint, save_path)
            logger.info("Saved epoch {epoch} weights to {wp}", epoch=epoch+1, wp=save_path)

        training_time = training_end - epoch_start
        validation_time = validation_end - training_end
        logger.info(
            (
                "[EPOCH {epoch}]: tloss: {tl:.5f}, vloss: {vl:.5f}, "
                "train time: {tts:.2f} sec. ({ttm:.2f} min), "
                "val time: {vts:.2f} sec. ({vtm:.2f} min)"
            ),
            epoch=epoch+1,
            tl=training_losses["train_loss"],
            vl=validation_losses["val_loss"],
            tts=training_time,
            ttm=training_time / 60,
            vts=validation_time,
            vtm=validation_time / 60,
        )

        # Determine best model.
        # < since we are looking for error rate, not accuracy.
        if validation_metrics["val_error_rate@5"] < best_metric:
            best_metric = validation_metrics["val_error_rate@5"]
            best_model_weights = get_cpu_state_dict(model)

    # Save best model if any.
    if best_model_weights is not None:
        save_name = f"alexnet_best_{epoch}_{round(best_metric, 4)}.pt"
        best_model_path = os.path.join(weights_path, save_name)
        torch.save(best_model_weights, best_model_path)
        logger.info("Saved best weights to {sp}", sp=best_model_path)


if __name__ == "__main__":
    parser = ap.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, default="config.yml", help="Path to configuration YAML file")
    args = parser.parse_args()
    main(args)
