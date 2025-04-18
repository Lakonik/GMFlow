import os
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, '../../')))

import shutil
import argparse
import multiprocessing as mp
import platform
import warnings

import cv2
import mmcv
import torch
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint

from mmgen.apis import set_random_seed
from mmgen.core import build_metric
from mmgen.datasets import build_dataset
from mmgen.models import build_model
from mmgen.utils import get_root_logger
from lib.apis.test import evaluate
from lib.parallel import MMDistributedDataParallel
from lib.datasets import build_dataloader

_distributed_metrics = ['FID', 'IS', 'FIDKID', 'PR', 'ADM', 'InceptionMetrics', 'DinoMetrics', 'ColorStats']


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test and eval a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
             '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
             '(only applicable to non-distributed testing)')
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
    parser.add_argument(
        '--diff_seed',
        action='store_true',
        help='Whether or not set different seeds for different ranks')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--data',
        type=str,
        nargs='+')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file.')
    parser.add_argument(
        '--timer',
        action='store_true',
        help='whether to enable timers')
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args


def setup_multi_processes(cfg):
    # set multi-process start method as `fork` to speed up the training
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', 'fork')
        mp.set_start_method(mp_start_method)

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', 0)
    cv2.setNumThreads(opencv_num_threads)

    # setup OMP threads
    # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
    if ('OMP_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1):
        omp_num_threads = 1
        warnings.warn(
            f'Setting OMP_NUM_THREADS environment variable for each process '
            f'to be {omp_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)

    # setup MKL threads
    if 'MKL_NUM_THREADS' not in os.environ and cfg.data.workers_per_gpu > 1:
        mkl_num_threads = 1
        warnings.warn(
            f'Setting MKL_NUM_THREADS environment variable for each process '
            f'to be {mkl_num_threads} in default, to avoid your system being '
            f'overloaded, please further tune the variable for optimal '
            f'performance in your application as needed.')
        os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    setup_multi_processes(cfg)

    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
        rank = 0
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        rank, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)

    dirname = os.path.dirname(args.checkpoint)
    ckpt = os.path.basename(args.checkpoint)

    if 'http' in args.checkpoint:
        log_path = None
    else:
        log_name = os.path.splitext(ckpt)[0] + '_eval_log' + '.txt'
        log_path = os.path.join(dirname, log_name)

    logger = get_root_logger(
        log_file=log_path, log_level=cfg.log_level, file_mode='a')
    logger.info('evaluation')

    # set random seeds
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}, '
                    f'use_rank_shift: {args.diff_seed}')
        set_random_seed(
            args.seed,
            deterministic=args.deterministic,
            use_rank_shift=args.diff_seed)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)

    if args.checkpoint.lower() != 'none':
        load_checkpoint(model, args.checkpoint, map_location='cpu')

    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)

    model.eval()

    for eval_cfg in cfg.evaluation:
        if args.data is not None:
            if eval_cfg.data not in args.data:
                continue

        viz_dir = eval_cfg.get('viz_dir', None)
        if viz_dir is not None:
            if os.path.exists(viz_dir):
                # if rank == 0:
                #     logger.warning(f'Viz directory {viz_dir} already exists, skipping this evaluation.')
                continue
            viz_dir_parent = os.path.dirname(viz_dir)
            viz_dir_name = os.path.basename(viz_dir)
            viz_dir_tmp = os.path.join(viz_dir_parent, 'tmp_' + viz_dir_name)
            if rank == 0:
                if os.path.exists(viz_dir_tmp):
                    shutil.rmtree(viz_dir_tmp)
                os.makedirs(viz_dir_tmp)
            if distributed:
                dist.barrier()
        else:
            viz_dir_tmp = None

        metrics = eval_cfg['metrics']
        if isinstance(metrics, dict):
            metrics = [metrics]
        metrics = [build_metric(metric) for metric in metrics]
        for metric in metrics:
            metric.prepare()

        # check metrics for dist evaluation
        if distributed and metrics:
            for metric in metrics:
                assert metric.name in _distributed_metrics, (
                    f'We only support {_distributed_metrics} for multi gpu '
                    f'evaluation, but receive {metric.name}.')

        # build the dataloader
        dataset = build_dataset(cfg.data[eval_cfg.data])

        # The default loader config
        loader_cfg = dict(
            workers_per_gpu=cfg.data.get('val_workers_per_gpu', cfg.data.workers_per_gpu),
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            shuffle=False)
        # The overall dataloader settings
        loader_cfg.update({
            k: v
            for k, v in cfg.data.items() if k not in [
                'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
                'test_dataloader', 'val_uncond', 'val_cond', 'val_text'
            ]
        })

        # specific config for test loader
        test_loader_cfg = {**loader_cfg, **cfg.data.get('test_dataloader', {})}

        data_loader = build_dataloader(dataset, **test_loader_cfg)

        if args.seed is not None:
            logger.info(f'Set random seed to {args.seed}, '
                        f'deterministic: {args.deterministic}, '
                        f'use_rank_shift: {args.diff_seed}')
            set_random_seed(
                args.seed,
                deterministic=args.deterministic,
                use_rank_shift=args.diff_seed)

        log_vars = evaluate(
            model, data_loader, metrics=metrics,
            feed_batch_size=eval_cfg.get('feed_batch_size', 32),
            viz_dir=viz_dir_tmp,
            viz_step=eval_cfg.get('viz_step', 1),
            sample_kwargs=eval_cfg.get('sample_kwargs', dict()),
            enable_timers=args.timer)

        if rank == 0:
            sys.stdout.write('\n')
            prefix = eval_cfg.get('prefix', '')
            for metric in metrics:
                with torch.no_grad():
                    metric.summary()
                for name, val in metric._result_dict.items():
                    prefix_name = prefix + '_' + name if len(prefix) > 0 else name
                    mmcv.print_log(f'{prefix_name} = {val}', 'mmgen')
            for name, val in log_vars.items():
                prefix_name = prefix + '_' + name if len(prefix) > 0 else name
                mmcv.print_log(f'{prefix_name} = {val}', 'mmgen')

        if viz_dir is not None:
            if distributed:
                dist.barrier()
            if rank == 0:
                os.rename(viz_dir_tmp, viz_dir)

    return


if __name__ == '__main__':
    main()
