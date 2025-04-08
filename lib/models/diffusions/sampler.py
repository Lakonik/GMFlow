# Copyright (c) 2025 Hansheng Chen

import torch

from mmgen.models.builder import MODULES


@MODULES.register_module()
class ContinuousTimeStepSampler:
    def __init__(
            self,
            num_timesteps,
            shift=1.0,
            logit_normal_enable=False,
            logit_normal_mean=0.0,
            logit_normal_std=1.0):
        self.num_timesteps = num_timesteps
        self.shift = shift
        self.logit_normal_enable = logit_normal_enable
        self.logit_normal_mean = logit_normal_mean
        self.logit_normal_std = logit_normal_std

    def sample(self, batch_size):
        if self.logit_normal_enable:
            t_linear = torch.sigmoid(
                self.logit_normal_mean + self.logit_normal_std * torch.randn(
                    (batch_size, ), dtype=torch.float))
        else:
            t_linear = 1 - torch.rand((batch_size, ), dtype=torch.float)
        t = self.shift * t_linear / (1 + (self.shift - 1) * t_linear)
        return t * self.num_timesteps

    def __call__(self, batch_size):
        return self.sample(batch_size)
