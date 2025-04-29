# Copyright (c) 2025 Hansheng Chen

import torch
import torch.nn as nn

from copy import deepcopy
from mmgen.models.builder import MODELS, build_module
from mmgen.utils import get_root_logger

from ...core import link_untrained_params


@MODELS.register_module()
class Diffusion2D(nn.Module):

    def __init__(self,
                 diffusion=dict(type='GMFlow'),
                 diffusion_use_ema=False,
                 autocast_dtype=None,
                 pretrained=None,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        diffusion.update(train_cfg=train_cfg, test_cfg=test_cfg)
        self.diffusion = build_module(diffusion)
        self.diffusion_use_ema = diffusion_use_ema
        if self.diffusion_use_ema:
            if inference_only:
                self.diffusion_ema = self.diffusion
            else:
                self.diffusion_ema = build_module(diffusion)  # deepcopy doesn't work due to the monkey patch lora
                link_untrained_params(self.diffusion_ema, self.diffusion)

        self.autocast_dtype = autocast_dtype
        self.pretrained = pretrained

        if self.pretrained is not None:
            self.load_checkpoint(self.pretrained, map_location='cpu', strict=False, logger=get_root_logger())

        self.train_cfg = dict() if train_cfg is None else deepcopy(train_cfg)
        self.test_cfg = dict() if test_cfg is None else deepcopy(test_cfg)

    def train_step(self, data, optimizer, loss_scaler=None, running_status=None):
        bs = data['x'].size(0)

        for v in optimizer.values():
            v.zero_grad()

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            loss, log_vars = self.diffusion(
                data['x'].reshape(bs, 2, 1, 1),
                return_loss=True)

        loss.backward() if loss_scaler is None else loss_scaler.scale(loss).backward()

        for k, v in optimizer.items():
            grad_clip = self.train_cfg.get(k + '_grad_clip', 0.0)
            grad_clip_begin_iter = self.train_cfg.get(k + '_grad_clip_begin_iter', 0)
            grad_clip_skip_ratio = self.train_cfg.get(k + '_grad_clip_skip_ratio', 0.0)
            if grad_clip > 0.0 and running_status['iteration'] >= grad_clip_begin_iter:
                try:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        getattr(self, k).parameters(), self.train_cfg[k + '_grad_clip'],
                        error_if_nonfinite=True)
                    if grad_clip_skip_ratio > 0 and grad_norm > grad_clip * grad_clip_skip_ratio:
                        grad_norm = float('nan')
                        v.zero_grad()
                except RuntimeError:
                    grad_norm = float('nan')
                    v.zero_grad()
                log_vars.update({k + '_grad_norm': grad_norm})
            if loss_scaler is None:
                v.step()
            else:
                loss_scaler.unscale_(v)
                loss_scaler.step(v)

        log_vars = {k: float(v) for k, v in log_vars.items()}

        outputs_dict = dict(log_vars=log_vars, num_samples=bs)

        return outputs_dict

    def val_step(self, data, test_cfg_override=dict(), **kwargs):
        bs = data['x'].size(0)
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                if 'noise' in data:
                    noise = data['noise'].reshape(bs, 2, 1, 1)
                else:
                    noise = torch.randn((bs, 2, 1, 1), device=data['x'].device)
                x_out = diffusion(
                    noise=noise,
                    test_cfg_override=test_cfg_override)

            return dict(num_samples=bs, pred_x=x_out)
