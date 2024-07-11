from nerfstudio.engine.trainer import TrainerConfig, Trainer, TRAIN_INTERATION_OUTPUT, TORCH_DEVICE
import dataclasses
import functools
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Dict, List, Literal, Optional, Tuple, Type, cast
from nerfstudio.configs.experiment_config import ExperimentConfig
import torch
import torch.nn as nn
from nerfstudio.utils import profiler, writer

                    
@dataclass 
class MERFTrainerConfig(ExperimentConfig):
    """Configuration for training regimen"""

    _target: Type = field(default_factory=lambda: MERFTrainer)
    """target class to instantiate"""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    use_grad_scaler: bool = False
    """Use gradient scaler even if the automatic mixed precision is disabled."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Path to config YAML file."""
    load_checkpoint: Optional[Path] = None
    """Path to checkpoint file."""
    log_gradients: bool = True
    """Optionally log gradients during training"""
    clip_gradients: bool = False
    """Clip gradients during training"""
    grad_max_val: float = 0.0
    """Clip gradients with max value"""
    grad_max_norm: float = 0.001
    """Clip gradients with max value"""
    gradient_accumulation_steps: int = 1
    """Number of steps to accumulate gradients over."""
    
    
    
class MERFTrainer(Trainer):
    
    @profiler.time_function
    def train_iteration(self, step: int) -> TRAIN_INTERATION_OUTPUT:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """

        self.optimizers.zero_grad_all()
        cpu_or_cuda_str: str = self.device.split(":")[0]
        assert (
            self.gradient_accumulation_steps > 0
        ), f"gradient_accumulation_steps must be > 0, not {self.gradient_accumulation_steps}"
        for _ in range(self.gradient_accumulation_steps):
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                loss = functools.reduce(torch.add, loss_dict.values())
                loss /= self.gradient_accumulation_steps
            self.grad_scaler.scale(loss).backward()  # type: ignore
        
        if self.config.clip_gradients:
            for tag, param in self.pipeline.model.named_parameters():
                if param.grad is not None:
                    # Clip by value
                    if self.config.grad_max_val > 0:
                        param.grad.data.clamp_(-self.config.grad_max_val, self.config.grad_max_val)
                    
                    # Clip by norm
                    if self.config.grad_max_norm > 0:
                        grad_norm = param.grad.data.norm()
                        scale = min(1, self.config.grad_max_norm / (grad_norm + 1e-6))
                        param.grad.data.mul_(scale)
                        
        for tag, param in self.pipeline.model.named_parameters():
            if param.grad is not None:
                param.grad.data = torch.nan_to_num(param.grad.data)
                        
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)

        if self.config.log_gradients:
            total_grad = 0
            for tag, value in self.pipeline.model.named_parameters():
                assert tag != "Total"
                if value.grad is not None:
                    grad = value.grad.norm()
                    metrics_dict[f"Gradients/{tag}"] = grad  # type: ignore
                    total_grad += grad

            metrics_dict["Gradients/Total"] = cast(torch.Tensor, total_grad)  # type: ignore

        scale = self.grad_scaler.get_scale()
        self.grad_scaler.update()
        # If the gradient scaler is decreased, no optimization step is performed so we should not step the scheduler.
        if scale <= self.grad_scaler.get_scale():
            self.optimizers.scheduler_step_all(step)

        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict  # type: ignore