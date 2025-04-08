# Copyright (c) 2025 Hansheng Chen

import os

import numpy as np
import torch
import mmcv

from PIL import Image
from torch.utils.data import Dataset
from mmcv.parallel import DataContainer as DC
from mmgen.datasets.builder import DATASETS
from mmgen.utils import get_root_logger


def image_preproc(pil_image, image_size, random_flip=False):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    arr = arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]

    if random_flip and np.random.rand() < 0.5:
        arr = np.ascontiguousarray(arr[:, ::-1])

    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    elif arr.ndim == 3:
        if arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        elif arr.shape[2] == 4:
            arr = arr[:, :, :3]
        else:
            assert arr.shape[2] == 3
    else:
        raise ValueError(f'Unexpected number of dimensions: {arr.ndim}')
    return arr


@DATASETS.register_module()
class ImageNet(Dataset):
    def __init__(
            self,
            data_root='data/imagenet/train',
            datalist_path='data/imagenet/train.txt',
            label2name_path='data/imagenet/imagenet1000_clsidx_to_labels.txt',
            random_flip=True,
            negative_label=1000,
            image_size=256,
            test_mode=False,
            num_test_images=50000):
        super().__init__()
        self.data_root = data_root
        self.datalist_path = datalist_path
        self.label2name_path = label2name_path
        self.random_flip = random_flip
        self.negative_label = negative_label
        self.image_size = image_size
        self.test_mode = test_mode
        self.num_test_images = num_test_images

        self.label2name = {}
        with open(label2name_path, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                idx, label = line.split(':')
                idx, label = idx.strip(), label.strip()
                if label[-1] == ',':
                    label = label[:-1]
                self.label2name[int(idx)] = label

        if not test_mode:
            self.all_paths = []
            self.all_labels = []
            with open(datalist_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if len(line) == 0:
                        continue
                    path_label = line.split(' ')
                    assert len(path_label) == 2
                    self.all_paths.append(path_label[0])
                    self.all_labels.append(int(path_label[1]))

            logger = get_root_logger()
            mmcv.print_log(f'Data root: {self.data_root}', logger=logger)
            mmcv.print_log(f'Data list path: {self.datalist_path}', logger=logger)
            mmcv.print_log(f'Number of images: {len(self.all_paths)}', logger=logger)

    def __len__(self):
        return self.num_test_images if self.test_mode else len(self.all_paths)

    def __getitem__(self, idx):
        data = dict(ids=DC(idx, cpu_only=True))

        if self.test_mode:
            generator = torch.Generator().manual_seed(idx)
            label = torch.randint(0, 1000, (), generator=generator).long()

        else:
            rel_data_path = self.all_paths[idx]
            data.update(paths=DC(rel_data_path, cpu_only=True))
            data_path = os.path.join(self.data_root, rel_data_path)
            ext = os.path.splitext(data_path)[-1]
            if ext.lower() in ('.pth', '.pt'):
                torch_data = torch.load(data_path)
                label = torch_data['y'].long()
                data.update(latents=torch_data['x'].float())
            elif ext.lower() in ('.jpg', '.jpeg', '.png'):
                label = torch.tensor(self.all_labels[idx], dtype=torch.long)
                img_data = Image.open(data_path)
                data.update(
                    images=torch.from_numpy(image_preproc(
                        img_data, self.image_size, random_flip=self.random_flip)).float().permute(2, 0, 1) / 255.0)
            else:
                raise ValueError(f'Unsupported file extension: {ext}')

        name = self.label2name[label.item()]
        data.update(labels=label, name=DC(name, cpu_only=True))

        if self.negative_label is not None:
            if isinstance(self.negative_label, int):
                data.update(negative_labels=torch.tensor(self.negative_label, dtype=torch.long))
            else:
                raise ValueError(f'Unsupported negative label: {self.negative_label}')

        return data
