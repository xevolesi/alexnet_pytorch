from copy import deepcopy
import warnings

from source.models import PAPER_ERROR_RATE_AT_1, PAPER_ERROR_RATE_AT_5
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


class TensorBoardLogger:
    def __init__(self, log_dir: str, init_model: nn.Module | None = None) -> None:
        self.log_dir = log_dir
        self.init_model = deepcopy(init_model).cpu()
        self.summary_writer = SummaryWriter(log_dir=log_dir)

    @torch.no_grad()
    def log_model_weights_hist(self, model: nn.Module, epoch: int) -> None:
        for module_name, module in model.named_modules():
            if hasattr(module, "weight"):
                tag = f"Weights/{module_name}_{module.__class__.__name__.lower()}"
                self.summary_writer.add_histogram(tag, module.weight.data, global_step=epoch)
            if hasattr(module, "bias"):
                tag = f"Biases/{module_name}_{module.__class__.__name__.lower()}"
                self.summary_writer.add_histogram(tag, module.bias.data, global_step=epoch)

    @torch.no_grad()
    def log_weights_diff_on_epoch(self, model: nn.Module, epoch: int) -> None:
        """Log the L2-norm of the difference of the weights between 2
        consecutive epochs.
        Log nothing if the initial model state was not provided in
        __init__ method.

        Args:
            model: Model after current epoch;
            epoch: Current epoch.

        Warnings:
            Warn you about the impossibility to log the difference if
            you didn't provide initial model's weights.

        """
        if self.init_model is None:
            message = (
                "You request to log difference between weights from epoch to epoch, "
                "but didn't provide initial model to the __init__ method, so skipping."
            )
            warnings.warn(message, category=UserWarning, stacklevel=2)
            return
        model = deepcopy(model).cpu()
        scalars = {}
        for (module_name, prev_module), cur_module in zip(self.init_model.named_modules(), model.modules()):
            if hasattr(prev_module, "weight"):
                tag = f"{module_name}_{prev_module.__class__.__name__.lower()}"
                scalars[tag] = torch.linalg.norm(prev_module.weight.data - cur_module.weight.data)
            if hasattr(prev_module, "bias"):
                tag = f"{module_name}_{prev_module.__class__.__name__.lower()}"
                scalars[tag] = torch.linalg.norm(prev_module.bias.data - cur_module.bias.data)
        self.summary_writer.add_scalars("Diff", scalars, global_step=epoch)

        # Current model state is now considered as previous model
        # state.
        self.init_model = model

    def log_optimizer_params(
            self,
            optimizer: torch.optim.Optimizer,
            param_names: tuple[str, ...] = ("lr", "weight_decay"),
            epoch: int = 0,
        ) -> None:
        """Log provided optimizer's parameters for all parameter's groups."""
        scalars = {}
        for group_idx, param_group in enumerate(optimizer.param_groups):
            for param_name in param_names:
                tag = f"Group{group_idx}_{param_name}"
                scalars[tag] = param_group[param_name]
        self.summary_writer.add_scalars("Optimizer", scalars, global_step=epoch)

    def log_gradient_hist(self, grad_dict: dict[str, torch.Tensor], epoch: int) -> None:
        for param_name, param_grad in grad_dict.items():
            tag = f"Gradients/{param_name}"
            self.summary_writer.add_histogram(tag, param_grad, epoch)

    def log_metrics(self, metric_dict: dict[str, float], epoch: int) -> None:
        paper_metrics = {"paper_error_rate@1": PAPER_ERROR_RATE_AT_1, "paper_error_rate@5": PAPER_ERROR_RATE_AT_5}
        self.summary_writer.add_scalars("Metrics", metric_dict | paper_metrics, epoch)

    def log_losses(self, losses_dict: dict[str, float], epoch: int) -> None:
        self.summary_writer.add_scalars("Losses", losses_dict, epoch)

    def log(
        self,
        epoch: int,
        metric_dict: dict[str, float],
        losses_dict: dict[str, float],
        grad_dict: dict[str, torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_param_names: tuple[str, ...] | None = None,
        model: torch.nn.Module | None = None,
    ) -> None:
        self.log_metrics(metric_dict, epoch)
        self.log_losses(losses_dict, epoch)

        if grad_dict is not None:
            self.log_gradient_hist(grad_dict, epoch)

        if optimizer is not None:
            if optimizer_param_names is None:
                message = "`optimizer_param_names` must not be `None` if you provide optimizer to log."
                raise ValueError(message)
            self.log_optimizer_params(optimizer, optimizer_param_names, epoch)

        if model is not None:
            self.log_model_weights_hist(model, epoch)
            self.log_weights_diff_on_epoch(model, epoch)

    def close(self) -> None:
        self.summary_writer.close()
