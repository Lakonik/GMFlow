import os

import mmcv
import torch.distributed as dist
from torch.hub import download_url_to_file
from huggingface_hub import hf_hub_download
from mmcv.runner import get_dist_info
from mmgen.utils.io_utils import MMGEN_CACHE_DIR


def download_from_url(url,
                      dest_path=None,
                      dest_dir=MMGEN_CACHE_DIR,
                      hash_prefix=None):
    """Modified from MMGeneration.
    """
    # get the exact destination path
    if dest_path is None:
        filename = url.split('/')[-1]
        dest_path = os.path.join(dest_dir, filename)

    if dest_path.startswith('~'):
        dest_path = os.path.expanduser('~') + dest_path[1:]

    # advoid downloading existed file
    if os.path.exists(dest_path):
        return dest_path

    rank, ws = get_dist_info()

    # only download from the master process
    if rank == 0:
        # mkdir
        _dir = os.path.dirname(dest_path)
        mmcv.mkdir_or_exist(_dir)
        download_url_to_file(url, dest_path, hash_prefix, progress=True)

    # sync the other processes
    if ws > 1:
        dist.barrier()

    return dest_path


def download_from_huggingface(filename):
    filename = filename.replace('huggingface://', '').split('/')
    repo_id = '/'.join(filename[:2])
    repo_filename = '/'.join(filename[2:])
    rank, world_size = get_dist_info()
    if rank == 0:
        cached_file = hf_hub_download(
            repo_id=repo_id, filename=repo_filename)
    if world_size > 1:
        dist.barrier()
        if rank > 0:
            cached_file = hf_hub_download(
                repo_id=repo_id, filename=repo_filename)
    return cached_file
