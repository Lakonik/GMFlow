import sys
import os
import numpy as np
import torch
import torch.distributed as dist
import mmcv

from copy import deepcopy
from mmcv.runner import HOOKS, get_dist_info
from mmgen.models.architectures.common import get_module_device
from mmgen.core import GenerativeEvalHook as _GenerativeEvalHook


def evaluate(model, dataloader, metrics=None,
             feed_batch_size=32, viz_dir=None, viz_step=1, sample_kwargs=dict()):
    batch_size = dataloader.batch_size
    rank, ws = get_dist_info()
    total_batch_size = batch_size * ws

    max_num_scenes = len(dataloader.dataset)

    if rank == 0:
        mmcv.print_log(
            f'Sample {max_num_scenes} fake scenes for evaluation', 'mmgen')
        pbar = mmcv.ProgressBar(max_num_scenes)

        if viz_dir is not None and os.path.exists(viz_dir):
            # remove existing files
            os.system(f'rm -rf {viz_dir}/*')

    if ws > 1:
        dist.barrier()

    log_vars = dict()
    batch_size_list = []
    # sampling fake images and directly send them to metrics
    for i, data in enumerate(dataloader):
        sample_kwargs_ = deepcopy(sample_kwargs)
        if viz_dir is not None and i % viz_step == 0:
            sample_kwargs_.update(viz_dir=viz_dir)

        outputs_dict = model.val_step(data, show_pbar=rank == 0, **sample_kwargs_)

        for k, v in outputs_dict['log_vars'].items():
            if k in log_vars:
                log_vars[k].append(outputs_dict['log_vars'][k])
            else:
                log_vars[k] = [outputs_dict['log_vars'][k]]
        batch_size_list.append(outputs_dict['num_samples'])

        if metrics is not None and len(metrics) > 0:
            pred_imgs = outputs_dict['pred_imgs']
            if pred_imgs.dim() == 5:  # multiview
                pred_imgs = pred_imgs.reshape(-1, *outputs_dict['pred_imgs'].shape[2:])
            pred_imgs = pred_imgs.split(feed_batch_size, dim=0)
            real_imgs = None
            if 'test_imgs' in data:  # legacy SSDNeRF dataset, channel last
                real_imgs = data['test_imgs'].permute(0, 1, 4, 2, 3)
                real_imgs = real_imgs.reshape(-1, *real_imgs.shape[2:])
            elif 'images' in data:  # new dataset, (*, c, h, w)
                real_imgs = data['images']
                if real_imgs.dim() == 5:
                    real_imgs = real_imgs.reshape(-1, *real_imgs.shape[2:])
            elif 'target_imgs' in outputs_dict:
                real_imgs = outputs_dict['target_imgs']
                if real_imgs.dim() == 5:
                    real_imgs = real_imgs.reshape(-1, *real_imgs.shape[2:])
            if real_imgs is not None:
                real_imgs = real_imgs.split(feed_batch_size, dim=0)
            for metric in metrics:
                for batch_id, batch_imgs in enumerate(pred_imgs):
                    # feed in fake images
                    metric.feed(batch_imgs * 2 - 1, 'fakes')
                    if real_imgs is not None:
                        metric.feed(real_imgs[batch_id] * 2 - 1, 'reals')

        if rank == 0:
            pbar.update(total_batch_size)

    if dist.is_initialized():
        device = get_module_device(model)
        batch_size_list = torch.tensor(batch_size_list, dtype=torch.float, device=device)
        batch_size_sum = torch.sum(batch_size_list)
        if dist.is_initialized():
            dist.all_reduce(batch_size_sum, op=dist.ReduceOp.SUM)
        for k, v in log_vars.items():
            weigted_values = torch.tensor(log_vars[k], dtype=torch.float, device=device) * batch_size_list
            weigted_values_sum = torch.sum(weigted_values)
            if dist.is_initialized():
                dist.all_reduce(weigted_values_sum, op=dist.ReduceOp.SUM)
            log_vars[k] = float(weigted_values_sum / batch_size_sum)
    else:
        for k, v in log_vars.items():
            log_vars[k] = np.average(log_vars[k], weights=batch_size_list)

    return log_vars


@HOOKS.register_module(force=True)
class GenerativeEvalHook(_GenerativeEvalHook):

    greater_keys = ['acc', 'top', 'AR@', 'auc', 'precision', 'mAP', 'is', 'test_ssim', 'test_psnr']
    less_keys = ['loss', 'fid', 'kid', 'test_lpips']
    _supported_best_metrics = ['fid', 'kid', 'is', 'test_ssim', 'test_psnr', 'test_lpips']

    def __init__(self,
                 *args,
                 data='',
                 viz_dir=None,
                 feed_batch_size=32,
                 viz_step=32,
                 clear_reals=False,
                 prefix='',
                 **kwargs):
        super(GenerativeEvalHook, self).__init__(*args, **kwargs)
        self.data = data
        self.viz_dir = viz_dir
        os.makedirs(viz_dir, exist_ok=True)
        self.feed_batch_size = feed_batch_size
        self.viz_step = viz_step
        self.clear_reals = clear_reals
        self.prefix = prefix

    @torch.no_grad()
    def after_train_iter(self, runner):
        interval = self.get_current_interval(runner)
        if not self.every_n_iters(runner, interval):
            return

        runner.model.eval()

        log_vars = evaluate(
            runner.model, self.dataloader, self.metrics, self.feed_batch_size,
            self.viz_dir, self.viz_step, self.sample_kwargs)

        if len(runner.log_buffer.output) == 0:
            runner.log_buffer.clear()
        rank, ws = get_dist_info()
        # a dirty walkround to change the line at the end of pbar
        if rank == 0:
            sys.stdout.write('\n')
            for metric in self.metrics:
                metric.summary()
                for name, val in metric._result_dict.items():
                    prefix_name = self.prefix + '_' + name if len(self.prefix) > 0 else name
                    runner.log_buffer.output[self.data + '_' + prefix_name] = val
                    # record best metric and save the best ckpt
                    if self.save_best_ckpt and name in self.best_metric:
                        self._save_best_ckpt(runner, val, name)
            for name, val in log_vars.items():
                prefix_name = self.prefix + '_' + name if len(self.prefix) > 0 else name
                # print(self.data + '_' + prefix_name + ' = {}'.format(val))
                runner.log_buffer.output[self.data + '_' + prefix_name] = val
                # record best metric and save the best ckpt
                if self.save_best_ckpt and name in self.best_metric:
                    self._save_best_ckpt(runner, val, name)
            runner.log_buffer.ready = True

        runner.model.train()

        # clear all current states for next evaluation
        for metric in self.metrics:
            metric.clear(clear_reals=self.clear_reals)
