import os
import os.path as osp
import time

import logging
import re
import torch
import mmcv

from collections import OrderedDict
from typing import Callable, Optional, Union
from tempfile import TemporaryDirectory
from torch.optim import Optimizer
from mmcv.runner import CheckpointLoader, load_state_dict, _load_checkpoint, weights_to_cpu
from mmcv.parallel import is_module_wrapper

from safetensors.torch import load_file

from lib.core import rgetattr, download_from_huggingface


@CheckpointLoader.register_scheme(prefixes='huggingface://')
def load_from_huggingface(filename, map_location=None):
    cached_file = download_from_huggingface(filename)
    ext = os.path.splitext(cached_file)[-1].lower()
    if ext == '.safetensors':
        return load_file(cached_file, device=map_location)
    else:
        return torch.load(cached_file, map_location=map_location)


def load_checkpoint(
        model: torch.nn.Module,
        filename: str,
        map_location: Union[str, Callable, None] = None,
        strict: bool = False,
        logger: Optional[logging.Logger] = None,
        revise_keys: list = [(r'^module\.', '')],
        convert_dtype: bool = False) -> Union[dict, OrderedDict]:
    checkpoint = _load_checkpoint(filename, map_location, logger)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # strip prefix of state_dict
    metadata = getattr(state_dict, '_metadata', OrderedDict())
    for p, r in revise_keys:
        state_dict = OrderedDict(
            {re.sub(p, r, k): v
             for k, v in state_dict.items()})
    # Keep metadata in state_dict
    state_dict._metadata = metadata

    # load state_dict
    if convert_dtype:  # use state_dict dtype
        for key, value in state_dict.items():
            param = rgetattr(model, key, None)
            if param is not None:
                param.data = value.data.to(device=param.device)
    else:  # use model dtype
        load_state_dict(model, state_dict, strict, logger)
    return checkpoint


def _save_to_state_dict(module, destination, prefix, keep_vars, trainable_only=False):
    for name, param in module._parameters.items():
        if param is not None and (not trainable_only or param.requires_grad):
            destination[prefix + name] = param if keep_vars else param.detach()
    for name, buf in module._buffers.items():
        # remove check of _non_persistent_buffers_set to allow nn.BatchNorm2d
        if buf is not None:
            destination[prefix + name] = buf if keep_vars else buf.detach()


def get_state_dict(module,
                   destination=None,
                   prefix='',
                   keep_vars=False,
                   trainable_only=False):
    # recursively check parallel module in case that the model has a
    # complicated structure, e.g., nn.Module(nn.Module(DDP))
    if is_module_wrapper(module):
        module = module.module

    # below is the same as torch.nn.Module.state_dict()
    if destination is None:
        destination = OrderedDict()
        destination._metadata = OrderedDict()  # type: ignore
    destination._metadata[prefix[:-1]] = local_metadata = dict(  # type: ignore
        version=module._version)
    _save_to_state_dict(module, destination, prefix, keep_vars, trainable_only=trainable_only)  # type: ignore
    for name, child in module._modules.items():
        if child is not None:
            get_state_dict(
                child, destination, prefix + name + '.', keep_vars=keep_vars, trainable_only=trainable_only)
    for hook in module._state_dict_hooks.values():
        hook_result = hook(module, destination, prefix, local_metadata)
        if hook_result is not None:
            destination = hook_result
    return destination  # type: ignore


def save_checkpoint(model,
                    filename,
                    optimizer=None,
                    loss_scaler=None,
                    save_apex_amp=False,
                    meta=None,
                    trainable_only=False,
                    fp16_ema=True):
    if meta is None:
        meta = {}
    elif not isinstance(meta, dict):
        raise TypeError(f'meta must be a dict or None, but got {type(meta)}')
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())

    if is_module_wrapper(model):
        model = model.module

    if hasattr(model, 'CLASSES') and model.CLASSES is not None:
        # save class name to the meta
        meta.update(CLASSES=model.CLASSES)

    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(get_state_dict(model, trainable_only=trainable_only))}
    if fp16_ema:
        for k, v in checkpoint['state_dict'].items():
            if '_ema.' in k and v.dtype == torch.float32:
                checkpoint['state_dict'][k] = v.half()

    # save optimizer state dict in the checkpoint
    if isinstance(optimizer, Optimizer):
        checkpoint['optimizer'] = optimizer.state_dict()
    elif isinstance(optimizer, dict):
        checkpoint['optimizer'] = {}
        for name, optim in optimizer.items():
            checkpoint['optimizer'][name] = optim.state_dict()

    # save loss scaler for mixed-precision (FP16) training
    if loss_scaler is not None:
        checkpoint['loss_scaler'] = loss_scaler.state_dict()

    # save state_dict from apex.amp
    if save_apex_amp:
        from apex import amp
        checkpoint['amp'] = amp.state_dict()

    if filename.startswith('pavi://'):
        try:
            from pavi import modelcloud
            from pavi.exception import NodeNotFoundError
        except ImportError:
            raise ImportError(
                'Please install pavi to load checkpoint from modelcloud.')
        model_path = filename[7:]
        root = modelcloud.Folder()
        model_dir, model_name = osp.split(model_path)
        try:
            model = modelcloud.get(model_dir)
        except NodeNotFoundError:
            model = root.create_training_model(model_dir)
        with TemporaryDirectory() as tmp_dir:
            checkpoint_file = osp.join(tmp_dir, model_name)
            with open(checkpoint_file, 'wb') as f:
                torch.save(checkpoint, f)
                f.flush()
            model.create_file(checkpoint_file, name=model_name)
    else:
        mmcv.mkdir_or_exist(osp.dirname(filename))
        # immediately flush buffer
        with open(filename, 'wb') as f:
            torch.save(checkpoint, f)
            f.flush()
