# Copyright (c) 2025 Hansheng Chen

import inspect
import numpy as np
import torch
import diffusers

from dataclasses import dataclass
from typing import Optional, Tuple, Union
from diffusers.configuration_utils import register_to_config
from diffusers.utils import BaseOutput
from diffusers.schedulers import SchedulerMixin
from diffusers.configuration_utils import ConfigMixin


@dataclass
class FlowWrapperSchedulerOutput(BaseOutput):
    prev_sample: torch.FloatTensor


class FlowAdapterScheduler(SchedulerMixin, ConfigMixin):

    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: float = 1.0,
            base_scheduler='DPMSolverMultistep',
            eps=1e-6,
            **kwargs):

        sigmas = torch.from_numpy(1 - np.linspace(
            0, 1, num_train_timesteps, dtype=np.float32, endpoint=False))
        self.sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
        self.timesteps = self.sigmas * num_train_timesteps
        alphas = 1 - self.sigmas

        base_scheduler_class = getattr(diffusers.schedulers, base_scheduler + 'Scheduler', None)

        if base_scheduler_class is None:
            raise AttributeError(f'Cannot find base_scheduler [{base_scheduler}].')
        if base_scheduler in ['EulerDiscrete', 'EulerAncestralDiscrete']:
            self.scales = ((alphas ** 2 + self.sigmas ** 2) / (
                    1 + (self.sigmas / alphas.clamp(min=self.config.eps)) ** 2)).sqrt()
        elif base_scheduler in [
                'DPMSolverSinglestep', 'DPMSolverMultistep', 'DEISMultistep', 'SASolver', 'UniPCMultistep']:
            self.scales = (alphas ** 2 + self.sigmas ** 2).sqrt()
        else:
            raise AttributeError(f'Unsupported base_scheduler [{base_scheduler}].')

        signatures = inspect.signature(base_scheduler_class).parameters.keys()
        if 'final_sigmas_type' in signatures:
            kwargs['final_sigmas_type'] = 'zero'
        if 'lower_order_final' in signatures:
            kwargs['lower_order_final'] = True

        self.base_scheduler = base_scheduler_class(
            num_train_timesteps=num_train_timesteps,
            prediction_type='epsilon',
            **kwargs)
        self.base_scheduler.timesteps = self.timesteps
        self.base_scheduler.sigmas = self.sigmas / alphas.clamp(min=self.config.eps)

        self._step_index = None
        self._begin_index = None

    @property
    def step_index(self):
        return self._step_index

    @property
    def begin_index(self):
        return self._begin_index

    def set_timesteps(self, num_inference_steps: int, device=None):
        self.num_inference_steps = num_inference_steps

        sigmas = torch.from_numpy(1 - np.linspace(
            0, 1, num_inference_steps, dtype=np.float32, endpoint=False))
        sigmas = self.config.shift * sigmas / (1 + (self.config.shift - 1) * sigmas)
        self.timesteps = (sigmas * self.config.num_train_timesteps).to(device)
        if self.config.base_scheduler in ['DEISMultistep', 'SASolver']:
            self.sigmas = torch.cat(
                [sigmas, torch.tensor([self.config.eps], dtype=torch.float32, device=sigmas.device)])
        else:
            self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        alphas = 1 - self.sigmas

        self.base_scheduler.set_timesteps(num_inference_steps, device=device)

        if self.config.base_scheduler in ['EulerDiscrete', 'EulerAncestralDiscrete']:
            self.base_scheduler.timesteps = self.timesteps
            self.base_scheduler.sigmas = self.sigmas / alphas.clamp(min=self.config.eps)
            self.scales = ((alphas ** 2 + self.sigmas ** 2) / (
                    1 + (self.sigmas / alphas.clamp(min=self.config.eps)) ** 2)).sqrt()
        elif self.config.base_scheduler in [
                'DPMSolverSinglestep', 'DPMSolverMultistep', 'DEISMultistep', 'SASolver', 'UniPCMultistep']:
            self.base_scheduler.timesteps = self.timesteps
            self.base_scheduler.sigmas = self.sigmas / alphas.clamp(min=self.config.eps)
            self.scales = (alphas**2 + self.sigmas**2).sqrt()
        else:
            raise AttributeError(f'Unsupported base_scheduler [{self.config.base_scheduler}].')

        self._step_index = None
        self._begin_index = None

    def index_for_timestep(self, timestep, schedule_timesteps=None):
        if schedule_timesteps is None:
            schedule_timesteps = self.timesteps

        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def _init_step_index(self, timestep):
        if self.begin_index is None:
            if isinstance(timestep, torch.Tensor):
                timestep = timestep.to(self.timesteps.device)
            self._step_index = self.index_for_timestep(timestep)
        else:
            self._step_index = self._begin_index

    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Union[float, torch.FloatTensor],
            sample: torch.FloatTensor,
            generator: Optional[torch.Generator] = None,
            return_dict: bool = True,
            prediction_type='u',
            eps=1e-6) -> Union[FlowWrapperSchedulerOutput, Tuple]:
        assert prediction_type in ['u', 'x0']

        if self.step_index is None:
            self._init_step_index(timestep)

        # Upcast to avoid precision issues when computing prev_sample
        ori_dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)

        sigma = self.sigmas[self.step_index]
        alpha = 1 - sigma
        scale = self.scales[self.step_index]
        next_scale = self.scales[self.step_index + 1]

        # norm_factor = (alpha ** 2 + sigma ** 2).sqrt()
        # alpha_vp = alpha / norm_factor
        # sigma_vp = sigma / norm_factor
        if prediction_type == 'u':
            epsilon = sample + alpha * model_output
            # x_0 = sample - sigma * model_output
        else:
            epsilon = (sample - alpha * model_output) / sigma.clamp(min=eps)
            # x_0 = model_output
        # v = alpha_vp * epsilon - sigma_vp * x_0

        if hasattr(self.base_scheduler, 'is_scale_input_called'):
            self.base_scheduler.is_scale_input_called = True
        kwargs = dict(return_dict=False)
        if generator is not None:
            kwargs.update(generator=generator)
        prev_sample = self.base_scheduler.step(
            epsilon,
            timestep,
            sample / scale,
            **kwargs
        )[0] * next_scale

        prev_sample = prev_sample.to(ori_dtype)

        # upon completion increase step index by one
        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return FlowWrapperSchedulerOutput(prev_sample=prev_sample)
