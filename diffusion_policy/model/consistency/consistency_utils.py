import math
from typing import Optional

import torch
import torch.nn.functional as F


def get_karras_sigmas(
        num_scales: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if num_scales < 2:
        raise ValueError("num_scales must be at least 2 for training")

    ramp = torch.linspace(0, 1, num_scales, device=device, dtype=dtype)
    min_inv_rho = sigma_min ** (1.0 / rho)
    max_inv_rho = sigma_max ** (1.0 / rho)
    return (min_inv_rho + ramp * (max_inv_rho - min_inv_rho)) ** rho


def get_sampling_sigmas(
        num_steps: int,
        sigma_min: float,
        sigma_max: float,
        rho: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None) -> torch.Tensor:
    if num_steps <= 1:
        return torch.tensor([sigma_max], device=device, dtype=dtype)
    return torch.flip(
        get_karras_sigmas(
            num_scales=num_steps,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            rho=rho,
            device=device,
            dtype=dtype),
        dims=(0,))


def sample_training_indices(
        batch_size: int,
        num_scales: int,
        device: torch.device) -> torch.Tensor:
    return torch.randint(0, num_scales - 1, (batch_size,), device=device)


def consistency_error(
        pred: torch.Tensor,
        target: torch.Tensor,
        loss_type: str) -> torch.Tensor:
    if loss_type == 'l1':
        return torch.abs(pred - target)
    if loss_type == 'l2':
        return F.mse_loss(pred, target, reduction='none')
    if loss_type == 'pseudo_huber':
        dim = math.prod(pred.shape[1:])
        delta = 0.00054 * math.sqrt(dim)
        return torch.sqrt((pred - target) ** 2 + delta ** 2) - delta
    raise ValueError(f"Unsupported consistency loss type: {loss_type}")
