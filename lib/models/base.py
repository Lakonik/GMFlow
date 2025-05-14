import torch
import torch.nn as nn


class BaseModel(nn.Module):

    def step_optimizer(self, optimizer, loss_scaler, running_status, log_vars):
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
        return log_vars
