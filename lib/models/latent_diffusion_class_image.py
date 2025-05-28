# Copyright (c) 2025 Hansheng Chen

import os
import torch

from threading import Thread
from PIL import Image
from copy import deepcopy
from mmgen.models.builder import MODELS, build_module
from mmgen.utils import get_root_logger

from .base import BaseModel
from ..core import link_untrained_params


@MODELS.register_module()
class LatentDiffusionClassImage(BaseModel):

    def __init__(self,
                 vae=dict(type='PretrainedVAE'),
                 diffusion=dict(type='GaussianFlow'),
                 diffusion_use_ema=False,
                 autocast_dtype=None,
                 pretrained=None,
                 inference_only=False,
                 train_cfg=None,
                 test_cfg=None):
        super().__init__()
        self.vae = build_module(vae) if vae is not None else None
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
        bs = data['latents'].size(0)
        labels = data['labels']

        prob_class = self.train_cfg.get('prob_class', 1.0)
        if prob_class < 1.0:
            labels = torch.where(
                torch.rand_like(labels, dtype=torch.float32) < prob_class,
                labels, data['negative_labels'])

        for v in optimizer.values():
            v.zero_grad()

        with torch.autocast(
                device_type='cuda',
                enabled=self.autocast_dtype is not None,
                dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
            loss, log_vars = self.diffusion(
                data['latents'],
                return_loss=True,
                class_labels=labels)

        loss.backward() if loss_scaler is None else loss_scaler.scale(loss).backward()

        log_vars = self.step_optimizer(optimizer, loss_scaler, running_status, log_vars)
        log_vars = {k: float(v) for k, v in log_vars.items()}
        outputs_dict = dict(log_vars=log_vars, num_samples=bs)

        return outputs_dict

    def eval_and_viz(self, out_images, data, viz_dir=None, cfg=dict()):
        if viz_dir is None:
            viz_dir = cfg.get('viz_dir', None)

        if viz_dir is not None:
            images_viz = (out_images * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            for i, image in enumerate(images_viz):
                name = f'{data["ids"][i]:09d}_' + data['name'][i] + '.png'
                image = Image.fromarray(image)
                t = Thread(target=image.save, args=(os.path.join(viz_dir, name),))
                t.start()

        return dict()

    def val_step(self, data, viz_dir=None, test_cfg_override=dict(), **kwargs):
        bs = len(data['labels'])
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)
        c, h, w = cfg['latent_size']
        guidance_scale = cfg.get('guidance_scale', 1.0)
        diffusion = self.diffusion_ema if self.diffusion_use_ema else self.diffusion

        with torch.no_grad():
            class_labels = data['labels']

            if guidance_scale != 0.0 and guidance_scale != 1.0:
                class_labels = torch.cat([data['negative_labels'], class_labels], dim=0)

            with torch.autocast(
                    device_type='cuda',
                    enabled=self.autocast_dtype is not None,
                    dtype=getattr(torch, self.autocast_dtype) if self.autocast_dtype is not None else None):
                if 'noise' in data:
                    noise = data['noise']
                else:
                    noise = torch.randn((bs, c, h, w), device=data['labels'].device)
                latents_out = diffusion(
                    noise=noise,
                    class_labels=class_labels,
                    guidance_scale=guidance_scale,
                    test_cfg_override=test_cfg_override)

            if hasattr(self.vae, 'dtype'):
                vae_dtype = self.vae.dtype
            else:
                vae_dtype = next(self.vae.parameters()).dtype
            latents_out = latents_out.to(vae_dtype)

            out_images = (self.vae.decode(latents_out).float() / 2 + 0.5).clamp(min=0, max=1)

            log_vars = self.eval_and_viz(out_images, data, viz_dir=viz_dir, cfg=cfg)

            return dict(log_vars=log_vars, num_samples=bs, pred_imgs=out_images)
