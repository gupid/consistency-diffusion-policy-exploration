from typing import Dict, Optional, Tuple

import torch
from einops import reduce

from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.model.consistency.consistency_utils import (
    consistency_error,
    get_karras_sigmas,
    get_sampling_sigmas,
    sample_training_indices,
)
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from diffusion_policy.policy.base_lowdim_policy import BaseLowdimPolicy


class ConsistencyUnetLowdimPolicy(BaseLowdimPolicy):
    def __init__(self,
            model: ConditionalUnet1D,
            horizon,
            obs_dim,
            action_dim,
            n_action_steps,
            n_obs_steps,
            num_inference_steps=4,
            num_train_scales=150,
            sigma_min=0.002,
            sigma_max=80.0,
            sigma_data=0.5,
            rho=7.0,
            loss_type='l2',
            reconstruction_loss_weight=0.25,
            scale_consistency_loss=True,
            clip_sample=True,
            obs_as_local_cond=False,
            obs_as_global_cond=True,
            pred_action_steps_only=False,
            oa_step_convention=False):
        super().__init__()
        if obs_as_local_cond:
            raise NotImplementedError("Consistency lowdim policy only supports global conditioning")
        if not obs_as_global_cond:
            raise NotImplementedError("Consistency lowdim policy only supports obs_as_global_cond=True")
        if pred_action_steps_only:
            raise NotImplementedError("Consistency lowdim policy currently assumes full-horizon action prediction")

        self.model = model
        self.normalizer = LinearNormalizer()
        self.horizon = horizon
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.n_action_steps = n_action_steps
        self.n_obs_steps = n_obs_steps
        self.num_inference_steps = num_inference_steps
        self.num_train_scales = num_train_scales
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.sigma_data = sigma_data
        self.rho = rho
        self.loss_type = loss_type
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.scale_consistency_loss = scale_consistency_loss
        self.clip_sample = clip_sample
        self.obs_as_local_cond = False
        self.obs_as_global_cond = True
        self.pred_action_steps_only = False
        self.oa_step_convention = oa_step_convention

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def _sigma_batch(self,
            sigma: torch.Tensor,
            batch_size: int,
            device: torch.device,
            dtype: torch.dtype) -> torch.Tensor:
        if not torch.is_tensor(sigma):
            sigma = torch.full((batch_size,), float(sigma), device=device, dtype=dtype)
        else:
            sigma = sigma.to(device=device, dtype=dtype).reshape(-1)
            if sigma.shape[0] == 1 and batch_size > 1:
                sigma = sigma.expand(batch_size)
        if sigma.shape[0] != batch_size:
            raise ValueError(f"Expected sigma batch of size {batch_size}, got {sigma.shape[0]}")
        return sigma

    def _sigma_dims(self, sigma: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return sigma.reshape(sigma.shape[0], *([1] * (reference.ndim - 1)))

    def _prepare_global_condition(self, nobs: torch.Tensor) -> torch.Tensor:
        return nobs[:, :self.n_obs_steps].reshape(nobs.shape[0], -1)

    def _prepare_training_batch(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        assert 'valid_mask' not in batch
        nbatch = self.normalizer.normalize(batch)
        obs = nbatch['obs']
        action = nbatch['action']
        global_cond = self._prepare_global_condition(obs)
        return action, global_cond

    def _prepare_action_sampling(self, obs_dict: Dict[str, torch.Tensor]) -> Tuple[Tuple[int, int, int], torch.Tensor]:
        assert 'obs' in obs_dict
        assert 'past_action' not in obs_dict

        nobs = self.normalizer['obs'].normalize(obs_dict['obs'])
        batch_size, _, obs_dim = nobs.shape
        if obs_dim != self.obs_dim:
            raise ValueError(f"Expected obs_dim={self.obs_dim}, got {obs_dim}")

        global_cond = self._prepare_global_condition(nobs)
        shape = (batch_size, self.horizon, self.action_dim)
        return shape, global_cond

    def predict_consistency(self,
            trajectory: torch.Tensor,
            sigma: torch.Tensor,
            global_cond: torch.Tensor) -> torch.Tensor:
        sigma = self._sigma_batch(
            sigma=sigma,
            batch_size=trajectory.shape[0],
            device=trajectory.device,
            dtype=trajectory.dtype)
        sigma_model = sigma
        sigma_skip = torch.clamp(sigma - self.sigma_min, min=0.0)

        model_output = self.model(
            trajectory,
            sigma_model,
            local_cond=None,
            global_cond=global_cond)

        sigma_expanded = self._sigma_dims(sigma, trajectory)
        sigma_skip_expanded = self._sigma_dims(sigma_skip, trajectory)
        sigma_data_sq = self.sigma_data ** 2
        c_skip = sigma_data_sq / (sigma_skip_expanded ** 2 + sigma_data_sq)
        c_out = self.sigma_data * sigma_skip_expanded / torch.sqrt(sigma_expanded ** 2 + sigma_data_sq)
        denoised = c_skip * trajectory + c_out * model_output

        if self.clip_sample:
            denoised = denoised.clamp(-1.0, 1.0)
        return denoised

    def conditional_sample(self,
            shape,
            global_cond: torch.Tensor,
            generator=None) -> torch.Tensor:
        device = global_cond.device
        dtype = global_cond.dtype
        sigmas = get_sampling_sigmas(
            num_steps=self.num_inference_steps,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
            dtype=dtype)

        trajectory = torch.randn(
            shape,
            device=device,
            dtype=dtype,
            generator=generator) * sigmas[0]
        trajectory = self.predict_consistency(trajectory, sigmas[0], global_cond)

        for sigma in sigmas[1:]:
            sigma_value = float(sigma.item())
            noise_scale = max(sigma_value ** 2 - self.sigma_min ** 2, 0.0) ** 0.5
            if noise_scale > 0:
                trajectory = trajectory + torch.randn(
                    trajectory.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator) * noise_scale
            trajectory = self.predict_consistency(trajectory, sigma, global_cond)

        if self.clip_sample:
            trajectory = trajectory.clamp(-1.0, 1.0)
        return trajectory

    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        shape, global_cond = self._prepare_action_sampling(obs_dict)
        nsample = self.conditional_sample(shape=shape, global_cond=global_cond)

        action_pred = self.normalizer['action'].unnormalize(nsample)
        start = self.n_obs_steps
        if self.oa_step_convention:
            start = self.n_obs_steps - 1
        end = start + self.n_action_steps
        action = action_pred[:, start:end]

        return {
            'action': action,
            'action_pred': action_pred
        }

    def compute_loss(self,
            batch,
            ema_model: Optional['ConsistencyUnetLowdimPolicy'] = None,
            num_scales: Optional[int] = None):
        trajectory, global_cond = self._prepare_training_batch(batch)
        batch_size = trajectory.shape[0]
        device = trajectory.device
        dtype = trajectory.dtype

        if num_scales is None:
            num_scales = self.num_train_scales
        if num_scales < 2:
            raise ValueError("num_scales must be at least 2")

        sigmas = get_karras_sigmas(
            num_scales=num_scales,
            sigma_min=self.sigma_min,
            sigma_max=self.sigma_max,
            rho=self.rho,
            device=device,
            dtype=dtype)
        step_idx = sample_training_indices(batch_size, num_scales, device=device)
        sigma_1 = sigmas[step_idx]
        sigma_2 = sigmas[step_idx + 1]

        sigma_1_expanded = self._sigma_dims(sigma_1, trajectory)
        sigma_2_expanded = self._sigma_dims(sigma_2, trajectory)
        z = torch.randn_like(trajectory)
        noisy_1 = trajectory + z * sigma_1_expanded
        noisy_2 = trajectory + z * sigma_2_expanded

        pred_2 = self.predict_consistency(noisy_2, sigma_2, global_cond)
        teacher = ema_model if ema_model is not None else self
        with torch.no_grad():
            pred_1 = teacher.predict_consistency(noisy_1, sigma_1, global_cond)

        consistency_per_elem = consistency_error(pred_2, pred_1, loss_type=self.loss_type)
        consistency_per_sample = reduce(consistency_per_elem, 'b ... -> b', 'mean')
        if self.scale_consistency_loss:
            delta_sigma = torch.clamp(sigma_2 - sigma_1, min=1e-6)
            consistency_per_sample = consistency_per_sample * (100.0 / delta_sigma)
        consistency_loss = consistency_per_sample.mean()

        reconstruction_per_elem = consistency_error(
            pred_2, trajectory, loss_type=self.loss_type)
        reconstruction_per_sample = reduce(reconstruction_per_elem, 'b ... -> b', 'mean')
        reconstruction_loss = reconstruction_per_sample.mean()

        loss = consistency_loss + self.reconstruction_loss_weight * reconstruction_loss

        metrics = {
            'consistency_loss': consistency_loss.detach(),
            'reconstruction_loss': reconstruction_loss.detach(),
            'reconstruction_loss_weight': float(self.reconstruction_loss_weight),
            'total_loss': loss.detach(),
            'num_scales': float(num_scales),
            'sigma_1_mean': sigma_1.mean().detach(),
            'sigma_2_mean': sigma_2.mean().detach()
        }
        return loss, metrics
