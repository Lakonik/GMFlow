# Copyright (c) 2025 Hansheng Chen

import sys
import inspect
import torch
import diffusers
import mmcv

from copy import deepcopy
from mmgen.models.architectures.common import get_module_device
from mmgen.models.builder import MODULES, build_module
from mmgen.models.diffusions.utils import _get_noise_batch

from . import GaussianFlow, schedulers
from lib.ops.gmflow_ops.gmflow_ops import (
    gm_to_sample, gm_to_mean, gaussian_samples_to_gm_samples, gm_samples_to_gaussian_samples,
    iso_gaussian_mul_iso_gaussian, gm_mul_iso_gaussian, gm_to_iso_gaussian, gm_spectral_logprobs)


@torch.jit.script
def probabilistic_guidance_jit(cond_mean, total_var, uncond_mean, guidance_scale: float, orthogonal: float = 1.0):
    bias = cond_mean - uncond_mean
    if orthogonal > 0.0:
        bias = bias - ((bias * cond_mean).mean(
            dim=(-3, -2, -1), keepdim=True
        ) / (cond_mean * cond_mean).mean(dim=(-3, -2, -1), keepdim=True).clamp(min=1e-6) * cond_mean).mul(orthogonal)
    bias_power = (bias * bias).mean(dim=(-3, -2, -1), keepdim=True)
    avg_var = total_var.mean(dim=(-3, -2, -1), keepdim=True)
    bias = bias * ((avg_var / bias_power.clamp(min=1e-6)).sqrt() * guidance_scale)
    gaussian_output = dict(
        mean=cond_mean + bias,
        var=total_var * (1 - (guidance_scale * guidance_scale)))
    return gaussian_output, bias, avg_var


class GMFlowMixin:

    def sample_forward_transition(self, x_t_low, t_low, t_high, noise):
        bs = x_t_low.size(0)
        if t_low.dim() == 0:
            t_low = t_low.expand(bs)
        if t_high.dim() == 0:
            t_high = t_high.expand(bs)

        std_low = t_low.reshape(*t_low.size(), *((x_t_low.dim() - t_low.dim()) * [1])) / self.num_timesteps
        mean_low = 1 - std_low
        std_high = t_high.reshape(*t_high.size(), *((x_t_low.dim() - t_high.dim()) * [1])) / self.num_timesteps
        mean_high = 1 - std_high

        mean_trans = mean_high / mean_low
        std_trans = (std_high ** 2 - (mean_trans * std_low) ** 2).sqrt()
        return x_t_low * mean_trans + noise * std_trans

    def u_to_x_0(self, denoising_output, x_t, t, eps=1e-6):
        if isinstance(denoising_output, dict) and 'logweights' in denoising_output:
            x_t = x_t.unsqueeze(-4)

        bs = x_t.size(0)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=x_t.device)
        if t.dim() == 0:
            t = t.expand(bs)
        t = t.reshape(*t.size(), *((x_t.dim() - t.dim()) * [1]))

        sigma = t / self.num_timesteps

        if isinstance(denoising_output, dict):
            if 'logweights' in denoising_output:
                means_x_0 = x_t - sigma * denoising_output['means']
                logstds_x_0 = denoising_output['logstds'] + torch.log(sigma.clamp(min=eps))
                return dict(
                    means=means_x_0,
                    logstds=logstds_x_0,
                    logweights=denoising_output['logweights'])
            elif 'var' in denoising_output:
                mean = x_t - sigma * denoising_output['mean']
                var = denoising_output['var'] * sigma.square()
                return dict(mean=mean, var=var)
            else:
                raise ValueError('Invalid denoising_output.')

        else:  # sample mode
            x_0 = x_t - sigma * denoising_output
            return x_0

    def denoising_gm_convert(
            self, gm_src, x_t_tgt, x_t_src, t_tgt, t_src, eps=1e-6, prediction_type='x0'):
        assert isinstance(gm_src, dict)

        bs = x_t_src.size(0)
        if not isinstance(t_tgt, torch.Tensor):
            t_tgt = torch.tensor(t_tgt, device=x_t_src.device)
        if not isinstance(t_src, torch.Tensor):
            t_src = torch.tensor(t_src, device=x_t_src.device)
        if t_tgt.dim() == 0:
            t_tgt = t_tgt.expand(bs)
        if t_src.dim() == 0:
            t_src = t_src.expand(bs)
        t_tgt = t_tgt.reshape(*t_tgt.size(), *((x_t_src.dim() - t_tgt.dim()) * [1]))
        t_src = t_src.reshape(*t_src.size(), *((x_t_src.dim() - t_src.dim()) * [1]))

        sigma_src = t_src / self.num_timesteps
        sigma_tgt = t_tgt / self.num_timesteps
        alpha_src = (1 - sigma_src).clamp(min=eps)
        alpha_tgt = (1 - sigma_tgt).clamp(min=eps)

        if prediction_type == 'u':
            gm_src = self.u_to_x_0(gm_src, x_t_src, t_src)
        else:
            assert prediction_type == 'x0'

        gaussian_mask = iso_gaussian_mul_iso_gaussian(
            dict(
                mean=x_t_tgt / alpha_tgt,
                var=(sigma_tgt / alpha_tgt).square()),
            dict(
                mean=x_t_src / alpha_src,
                var=(sigma_src / alpha_src).square()),
            1, -1)
        gm_output_tgt = gm_mul_iso_gaussian(gm_src, gaussian_mask, 1, 1)[0]
        return gm_output_tgt

    def reverse_transition(self, denoising_output, x_t_high, t_low, t_high, eps=1e-6, prediction_type='u'):
        if isinstance(denoising_output, dict):
            x_t_high = x_t_high.unsqueeze(-4)

        bs = x_t_high.size(0)
        if not isinstance(t_low, torch.Tensor):
            t_low = torch.tensor(t_low, device=x_t_high.device)
        if not isinstance(t_high, torch.Tensor):
            t_high = torch.tensor(t_high, device=x_t_high.device)
        if t_low.dim() == 0:
            t_low = t_low.expand(bs)
        if t_high.dim() == 0:
            t_high = t_high.expand(bs)
        t_low = t_low.reshape(*t_low.size(), *((x_t_high.dim() - t_low.dim()) * [1]))
        t_high = t_high.reshape(*t_high.size(), *((x_t_high.dim() - t_high.dim()) * [1]))

        sigma = t_high / self.num_timesteps
        sigma_to = t_low / self.num_timesteps
        alpha = 1 - sigma
        alpha_to = 1 - sigma_to

        sigma_to_over_sigma = sigma_to / sigma.clamp(min=eps)
        alpha_over_alpha_to = alpha / alpha_to.clamp(min=eps)
        beta_over_sigma_sq = 1 - (sigma_to_over_sigma * alpha_over_alpha_to) ** 2

        c1 = sigma_to_over_sigma ** 2 * alpha_over_alpha_to
        c2 = beta_over_sigma_sq * alpha_to

        if isinstance(denoising_output, dict):
            c3 = beta_over_sigma_sq * sigma_to ** 2
            if prediction_type == 'u':
                means_x_0 = x_t_high - sigma * denoising_output['means']
                logstds_x_t_low = torch.logaddexp(
                    (denoising_output['logstds'] + torch.log((sigma * c2).clamp(min=eps))) * 2,
                    torch.log(c3.clamp(min=eps))
                ) / 2
            elif prediction_type == 'x0':
                means_x_0 = denoising_output['means']
                logstds_x_t_low = torch.logaddexp(
                    (denoising_output['logstds'] + torch.log(c2.clamp(min=eps))) * 2,
                    torch.log(c3.clamp(min=eps))
                ) / 2
            else:
                raise ValueError('Invalid prediction_type.')
            means_x_t_low = c1 * x_t_high + c2 * means_x_0
            return dict(
                means=means_x_t_low,
                logstds=logstds_x_t_low,
                logweights=denoising_output['logweights'])

        else:  # sample mode
            c3_sqrt = beta_over_sigma_sq ** 0.5 * sigma_to
            noise = torch.randn_like(denoising_output)
            if prediction_type == 'u':
                x_0 = x_t_high - sigma * denoising_output
            elif prediction_type == 'x0':
                x_0 = denoising_output
            else:
                raise ValueError('Invalid prediction_type.')
            x_t_low = c1 * x_t_high + c2 * x_0 + c3_sqrt * noise
            return x_t_low

    @staticmethod
    def gm_sample(gm, power_spectrum=None, n_samples=1, generator=None):
        device = gm['means'].device
        if power_spectrum is not None:
            power_spectrum = power_spectrum.to(dtype=torch.float32).unsqueeze(-4)
            shape = list(gm['means'].size())
            shape[-4] = n_samples
            half_size = shape[-1] // 2 + 1
            spectral_samples = torch.randn(
                shape, dtype=torch.float32, device=device, generator=generator) * (power_spectrum / 2).exp()
            z_1 = spectral_samples.roll((-1, -1), dims=(-2, -1)).flip((-2, -1))[..., :half_size]
            z_0 = spectral_samples[..., :half_size]
            z_kr = torch.complex(z_0 + z_1, z_0 - z_1) / 2
            gaussian_samples = torch.fft.irfft2(z_kr, norm='ortho')
            samples = gaussian_samples_to_gm_samples(gm, gaussian_samples, axis_aligned=True)
        else:
            samples = gm_to_sample(gm, n_samples=n_samples)
            spectral_samples = None
        return samples, spectral_samples

    def gm_to_model_output(self, gm, output_mode, power_spectrum=None, generator=None):
        assert output_mode in ['mean', 'sample']
        if output_mode == 'mean':
            output = gm_to_mean(gm)
        else:  # sample
            output = self.gm_sample(gm, power_spectrum, generator=generator)[0].squeeze(-4)
        return output

    def gm_2nd_order(
            self, gm_output, gaussian_output, x_t, t, h,
            guidance_scale=0.0, gm_cond=None, gaussian_cond=None, avg_var=None, cfg_bias=None, ca=0.005, cb=1.0,
            gm2_correction_steps=0):
        if self.prev_gm is not None:
            if cfg_bias is not None:
                gm_mean = gm_to_mean(gm_output)
                base_gaussian = gaussian_cond
                base_gm = gm_cond
            else:
                gm_mean = gaussian_output['mean']
                base_gaussian = gaussian_output
                base_gaussian['var'] = base_gaussian['var'].mean(dim=(-1, -2), keepdim=True)
                avg_var = base_gaussian['var'].mean(dim=(-3, -2, -1), keepdim=True)
                base_gm = gm_output

            gm_from_prev = self.denoising_gm_convert(
                self.prev_gm, x_t, self.prev_x_t, t, self.prev_t, prediction_type='x0')
            mean_from_prev = gm_to_mean(gm_from_prev)
            self.prev_gm = gm_output

            # Compute rescaled 2nd-order mean difference
            k = 0.5 * h / self.prev_h
            prev_h_norm = self.prev_h / self.num_timesteps
            _guidance_scale = guidance_scale * cb
            err_power = avg_var * (_guidance_scale * _guidance_scale + ca)
            mean_diff = (gm_mean - mean_from_prev) * (
                (1 - err_power / (prev_h_norm * prev_h_norm)).clamp(min=0).sqrt() * k)

            bias = mean_diff
            # Here we fuse probabilistic guidance bias and 2nd-order mean difference to perform one single
            # update to the base GM, which avoids cumulative errors.
            if cfg_bias is not None:
                bias = mean_diff + cfg_bias
            bias_power = bias.square().mean(dim=(-3, -2, -1), keepdim=True)
            bias = bias * (avg_var / bias_power.clamp(min=1e-6)).clamp(max=1).sqrt()

            gaussian_output = dict(
                mean=base_gaussian['mean'] + bias,
                var=base_gaussian['var'] * (1 - bias_power / avg_var.clamp(min=1e-6)).clamp(min=1e-6))
            gm_output = gm_mul_iso_gaussian(
                base_gm, iso_gaussian_mul_iso_gaussian(gaussian_output, base_gaussian, 1, -1),
                1, 1)[0]

            # Additional correction steps for strictly matching the 2nd order mean difference
            if gm2_correction_steps > 0:
                adjusted_bias = bias
                tgt_bias = mean_diff + gm_mean - base_gaussian['mean']
                for _ in range(gm2_correction_steps):
                    out_bias = gm_to_mean(gm_output) - base_gaussian['mean']
                    err = out_bias - tgt_bias
                    adjusted_bias = adjusted_bias - err * (
                        adjusted_bias.norm(dim=-3, keepdim=True) / out_bias.norm(dim=-3, keepdim=True).clamp(min=1e-6)
                    ).clamp(max=1)
                    adjusted_bias_power = adjusted_bias.square().mean(dim=(-3, -2, -1), keepdim=True)
                    adjusted_bias = adjusted_bias * (avg_var / adjusted_bias_power.clamp(min=1e-6)).clamp(max=1).sqrt()
                    adjusted_gaussian_output = dict(
                        mean=base_gaussian['mean'] + adjusted_bias,
                        var=base_gaussian['var'] * (1 - adjusted_bias_power / avg_var.clamp(min=1e-6)).clamp(min=1e-6))
                    gm_output = gm_mul_iso_gaussian(
                        base_gm, iso_gaussian_mul_iso_gaussian(adjusted_gaussian_output, base_gaussian, 1, -1),
                        1, 1)[0]

        else:
            self.prev_gm = gm_output

        self.prev_x_t = x_t
        self.prev_t = t
        self.prev_h = h

        return gm_output, gaussian_output

    def init_gm_cache(self):
        self.prev_gm = None
        self.prev_x_t = None
        self.prev_t = None
        self.prev_h = None


@MODULES.register_module()
class GMFlow(GaussianFlow, GMFlowMixin):

    def __init__(
            self,
            *args,
            spectrum_net=None,
            spectral_loss_weight=1.0,
            **kwargs):
        super().__init__(*args, **kwargs)
        self.spectrum_net = build_module(spectrum_net) if spectrum_net is not None else None
        self.spectral_loss_weight = spectral_loss_weight
        self.intermediate_x_t = []
        self.intermediate_x_0 = []

    def loss(self, denoising_output, x_t_low, x_t_high, t_low, t_high):
        """
        GMFlow transition loss.
        """
        with torch.autocast(device_type='cuda', enabled=False):
            denoising_output = {k: v.float() for k, v in denoising_output.items()}
            x_t_low = x_t_low.float()
            x_t_high = x_t_high.float()
            t_low = t_low.float()
            t_high = t_high.float()

            x_t_low_gm = self.reverse_transition(denoising_output, x_t_high, t_low, t_high)
            loss_kwargs = {k: v for k, v in x_t_low_gm.items()}

            loss_kwargs.update(x_t_low=x_t_low, timesteps=t_high)
            return self.flow_loss(loss_kwargs)

    def spectral_loss(self, denoising_output, x_0, x_t, t):
        with torch.autocast(device_type='cuda', enabled=False):
            x_0 = x_0.float()
            x_t = x_t.float()
            t = t.float()

            bs = x_0.size(0)
            num_channels = x_0.size(-3)
            if t.dim() == 0:
                t = t.expand(bs)
            else:
                assert t.numel() == bs
                t = t.flatten()

            with torch.no_grad():
                denoising_output = {k: v.float() for k, v in denoising_output.items()}
                output_g = self.u_to_x_0(gm_to_iso_gaussian(denoising_output)[0], x_t, t)
                t = t.reshape(bs, *((x_t.dim() - 1) * [1]))
                u = (x_t - x_0) * (self.num_timesteps / t.clamp(min=1e-6))
                z_kr = gm_samples_to_gaussian_samples(
                    denoising_output, u.unsqueeze(-4), axis_aligned=True).squeeze(-4)
                z_kr_fft = torch.fft.fft2(z_kr, norm='ortho')
                z_kr_fft = z_kr_fft.real + z_kr_fft.imag

            log_var = self.spectrum_net(output_g)

            loss = z_kr_fft.square() * (torch.exp(-log_var) - 1) + log_var
            loss = loss.mean() * (0.5 * num_channels * self.spectral_loss_weight)
            return loss

    def forward_train(self, x_0, *args, **kwargs):
        device = get_module_device(self)

        assert x_0.dim() == 4
        num_batches, num_channels, h, w = x_0.size()
        trans_ratio = self.train_cfg.get('trans_ratio', 1.0)
        eps = self.train_cfg.get('eps', 1e-4)

        with torch.autocast(device_type='cuda', enabled=False):
            t_high = self.timestep_sampler(num_batches).to(device).clamp(min=eps, max=self.num_timesteps)
            t_low = t_high * (1 - trans_ratio)
            t_low = torch.minimum(t_low, t_high - eps).clamp(min=0)

            noise = _get_noise_batch(
                None, x_0.shape[-3:],
                num_timesteps=self.num_timesteps,
                num_batches=2 * x_0.size(0),
                timesteps_noise=False
            ).to(x_0)  # (2 * num_batches, num_channels, h, w)
            noise_0, noise_1 = torch.chunk(noise, 2, dim=0)

            x_t_low, _, _ = self.sample_forward_diffusion(x_0, t_low, noise_0)
            x_t_high = self.sample_forward_transition(x_t_low, t_low, t_high, noise_1)

        denoising_output = self.pred(x_t_high, t_high, **kwargs)
        loss = self.loss(denoising_output, x_t_low, x_t_high, t_low, t_high)
        log_vars = self.flow_loss.log_vars
        log_vars.update(loss_transition=float(loss))

        if self.spectrum_net is not None:
            loss_spectral = self.spectral_loss(denoising_output, x_0, x_t_high, t_high)
            log_vars.update(loss_spectral=float(loss_spectral))
            loss = loss + loss_spectral

        return loss, log_vars

    def forward_test(
            self, x_0=None, noise=None, guidance_scale=0.0,
            test_cfg_override=dict(), show_pbar=False, **kwargs):
        x_t = torch.randn_like(x_0) if noise is None else noise
        cfg = deepcopy(self.test_cfg)
        cfg.update(test_cfg_override)

        output_mode = cfg.get('output_mode', 'mean')
        assert output_mode in ['mean', 'sample']

        sampler = cfg['sampler']
        sampler_class = getattr(diffusers.schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            sampler_class = getattr(schedulers, sampler + 'Scheduler', None)
        if sampler_class is None:
            raise AttributeError(f'Cannot find sampler [{sampler}].')

        sampler_kwargs = cfg.get('sampler_kwargs', {})
        signatures = inspect.signature(sampler_class).parameters.keys()
        if 'shift' in signatures and 'shift' not in sampler_kwargs:
            sampler_kwargs['shift'] = cfg.get('shift', self.timestep_sampler.shift)
        if 'output_mode' in signatures:
            sampler_kwargs['output_mode'] = output_mode
        sampler = sampler_class(self.num_timesteps, **sampler_kwargs)

        num_timesteps = cfg.get('num_timesteps', self.num_timesteps)
        num_substeps = cfg.get('num_substeps', 1)
        orthogonal_guidance = cfg.get('orthogonal_guidance', 1.0)
        save_intermediate = cfg.get('save_intermediate', False)
        order = cfg.get('order', 1)
        gm2_coefs = cfg.get('gm2_coefs', [0.005, 1.0])
        gm2_correction_steps = cfg.get('gm2_correction_steps', 0)

        sampler.set_timesteps(num_timesteps * num_substeps, device=x_t.device)
        timesteps = sampler.timesteps
        self.intermediate_x_t = []
        self.intermediate_x_0 = []
        self.intermediate_gm_x_0 = []
        self.intermediate_gm_trans = []
        self.intermediate_t = []
        use_guidance = guidance_scale > 0.0
        assert order in [1, 2]

        if show_pbar:
            pbar = mmcv.ProgressBar(num_timesteps)

        # ========== Main sampling loop ==========
        self.init_gm_cache()

        for timestep_id in range(num_timesteps):
            t = timesteps[timestep_id * num_substeps]

            if save_intermediate:
                self.intermediate_x_t.append(x_t)
                self.intermediate_t.append(t)

            model_input = x_t
            if use_guidance:
                model_input = torch.cat([x_t, model_input], dim=0)

            gm_output = self.pred(model_input, t, **kwargs)
            assert isinstance(gm_output, dict)

            ori_dtype = gm_output['means'].dtype
            gm_output = {k: v.to(torch.float32) for k, v in gm_output.items()}
            x_t = x_t.to(torch.float32)
            model_input = model_input.to(torch.float32)

            gm_output = self.u_to_x_0(gm_output, model_input, t)

            # ========== Probabilistic CFG ==========
            if use_guidance:
                num_batches = x_t.size(0)
                gm_cond = {k: v[num_batches:] for k, v in gm_output.items()}
                gm_uncond = {k: v[:num_batches] for k, v in gm_output.items()}
                uncond_mean = gm_to_mean(gm_uncond)
                gaussian_cond = gm_to_iso_gaussian(gm_cond)[0]
                gaussian_cond['var'] = gaussian_cond['var'].mean(dim=(-1, -2), keepdim=True)
                gaussian_output, cfg_bias, avg_var = probabilistic_guidance_jit(
                    gaussian_cond['mean'], gaussian_cond['var'], uncond_mean, guidance_scale,
                    orthogonal=orthogonal_guidance)
                gm_output = gm_mul_iso_gaussian(
                    gm_cond, iso_gaussian_mul_iso_gaussian(gaussian_output, gaussian_cond, 1, -1),
                    1, 1)[0]
            else:
                gaussian_output = gm_to_iso_gaussian(gm_output)[0]
                gm_cond = gaussian_cond = avg_var = cfg_bias = None

            # ========== 2nd order GM ==========
            if order == 2:
                if timestep_id < num_timesteps - 1:
                    h = t - timesteps[(timestep_id + 1) * num_substeps]
                else:
                    h = t
                gm_output, gaussian_output = self.gm_2nd_order(
                    gm_output, gaussian_output, x_t, t, h,
                    guidance_scale, gm_cond, gaussian_cond, avg_var, cfg_bias, ca=gm2_coefs[0], cb=gm2_coefs[1],
                    gm2_correction_steps=gm2_correction_steps)

            if save_intermediate:
                self.intermediate_gm_x_0.append(gm_output)
                if timestep_id < num_timesteps - 1:
                    t_next = timesteps[(timestep_id + 1) * num_substeps]
                else:
                    t_next = 0
                gm_trans = self.reverse_transition(gm_output, x_t, t_next, t, prediction_type='x0')
                self.intermediate_gm_trans.append(gm_trans)

            # ========== GM SDE step or GM ODE substeps ==========
            x_t_base = x_t
            t_base = t
            for substep_id in range(num_substeps):
                if substep_id == 0:
                    if self.spectrum_net is not None and output_mode == 'sample':
                        power_spectrum = self.spectrum_net(gaussian_output)
                    else:
                        power_spectrum = None
                    model_output = self.gm_to_model_output(gm_output, output_mode, power_spectrum=power_spectrum)
                else:
                    assert output_mode == 'mean'
                    t = timesteps[timestep_id * num_substeps + substep_id]
                    model_output = self.gm_to_model_output(
                        self.denoising_gm_convert(gm_output, x_t, x_t_base, t, t_base, prediction_type='x0'),
                        output_mode)
                x_t = sampler.step(model_output, t, x_t, return_dict=False, prediction_type='x0')[0]

            if save_intermediate:
                self.intermediate_x_0.append(model_output)

            x_t = x_t.to(ori_dtype)

            if show_pbar:
                pbar.update()

        if show_pbar:
            sys.stdout.write('\n')

        return x_t

    def forward_likelihood(self, x_0, *args, **kwargs):
        device = get_module_device(self)

        assert x_0.dim() == 4
        num_batches, num_channels, h, w = x_0.size()

        t = torch.rand(num_batches, device=device, dtype=x_0.dtype).clamp(min=1e-4) * self.num_timesteps

        noise = _get_noise_batch(
            None, x_0.shape[-3:],
            num_timesteps=self.num_timesteps,
            num_batches=x_0.size(0),
            timesteps_noise=False
        ).to(x_0)  # (2 * num_batches, num_channels, h, w)

        x_t, _, _ = self.sample_forward_diffusion(x_0, t, noise)

        denoising_output = self.pred(x_t, t, **kwargs)
        gm_output = self.u_to_x_0(denoising_output, x_t, t)

        if self.spectrum_net is not None:
            gaussian_output = gm_to_iso_gaussian(gm_output)[0]
            power_spectrum = self.spectrum_net(gaussian_output).float()
        else:
            power_spectrum = None

        # convert to float32
        gm_output = {k: v.float() for k, v in gm_output.items()}
        x_0 = x_0.float()
        logprobs = gm_spectral_logprobs(gm_output, x_0.unsqueeze(-4), power_spectrum).squeeze(-1)

        return logprobs
