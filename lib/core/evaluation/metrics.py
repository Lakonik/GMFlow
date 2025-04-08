import os
import pickle
import math
import numpy as np
import skimage
import torch
import torch.distributed as dist
import torch.nn.functional as F
import mmcv

from scipy import linalg
from scipy.stats import entropy
from torch.nn.functional import conv2d
from torchvision import models
from mmcv.runner import get_dist_info
from mmgen.utils import get_root_logger
from mmgen.core.registry import METRICS
from mmgen.core.evaluation.metrics import load_inception, Metric, TERO_INCEPTION_URL
from mmgen.core.evaluation.metrics import FID as _FID
from mmgen.core.evaluation.metrics import PR as _PR
from ..utils.io_utils import download_from_huggingface, download_from_url


def _get_gaussian_kernel1d(kernel_size: int, sigma: float, dtype=None, device=None):
    lim = (kernel_size - 1) / (2.0 * math.sqrt(2.0) * sigma)
    x = torch.linspace(-lim, lim, steps=kernel_size, dtype=dtype, device=device)
    kernel1d = torch.softmax(x.pow_(2).neg_(), dim=0)
    return kernel1d


def _get_gaussian_kernel2d(kernel_size, sigma, dtype=None, device=None):
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0], dtype, device)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1], dtype, device)
    kernel2d = kernel1d_y.unsqueeze(-1) * kernel1d_x
    return kernel2d


def filter_img_2d(img, kernel2d):
    batch_shape = img.shape[:-2]
    img = conv2d(img.reshape(batch_shape.numel(), 1, *img.shape[-2:]),
                 kernel2d[None, None])
    return img.reshape(*batch_shape, *img.shape[-2:])


def filter_img_2d_separate(img, kernel1d_x, kernel1d_y):
    batch_shape = img.shape[:-2]
    img = conv2d(conv2d(img.reshape(batch_shape.numel(), 1, *img.shape[-2:]),
                        kernel1d_x[None, None, None, :]),
                 kernel1d_y[None, None, :, None])
    return img.reshape(*batch_shape, *img.shape[-2:])


def gaussian_blur(img, kernel_size, sigma):
    if isinstance(kernel_size, int):
        kernel_size = [kernel_size, kernel_size]
    if isinstance(sigma, (int, float)):
        sigma = [float(sigma), float(sigma)]
    kernel = _get_gaussian_kernel2d(kernel_size, sigma, dtype=img.dtype, device=img.device)
    return filter_img_2d(img, kernel)


def eval_psnr(img1, img2, max_val=1.0, eps=1e-6):
    mse = (img1 - img2).square().flatten(1).mean(dim=-1)
    psnr = (10 * (2 * math.log10(max_val) - torch.log10(mse + eps)))
    return psnr


def eval_ssim_skimage(img1, img2, to_tensor=False, **kwargs):
    """Following pixelNeRF evaluation.

    Args:
        img1 (Tensor): Images with order "NCHW".
        img2 (Tensor): Images with order "NCHW".
    """
    _img1 = img1.cpu().numpy()
    _img2 = img2.cpu().numpy()
    ssims = []
    for _img1_single, _img2_single in zip(_img1, _img2):
        ssims.append(skimage.metrics.structural_similarity(
            _img1_single, _img2_single, channel_axis=0, **kwargs))
    return img1.new_tensor(ssims) if to_tensor else np.array(ssims, dtype=np.float32)


def eval_ssim(img1, img2, max_val=1.0, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03,
              separate_filter=True):
    """Calculate SSIM (structural similarity) and contrast sensitivity using Gaussian kernel.

    Args:
        img1 (Tensor): Images with order "NCHW".
        img2 (Tensor): Images with order "NCHW".

    Returns:
        tuple: Pair containing the mean SSIM and contrast sensitivity between
        `img1` and `img2`.
    """
    assert img1.shape == img2.shape
    _, _, h, w = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, h, w)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        if separate_filter:
            window = _get_gaussian_kernel1d(size, sigma, dtype=img1.dtype, device=img1.device)
            mu1 = filter_img_2d_separate(img1, window, window)
            mu2 = filter_img_2d_separate(img2, window, window)
            sigma11 = filter_img_2d_separate(img1.square(), window, window)
            sigma22 = filter_img_2d_separate(img2.square(), window, window)
            sigma12 = filter_img_2d_separate(img1 * img2, window, window)
        else:
            window = _get_gaussian_kernel2d([size, size], [sigma, sigma], dtype=img1.dtype, device=img1.device)
            mu1 = filter_img_2d(img1, window)
            mu2 = filter_img_2d(img2, window)
            sigma11 = filter_img_2d(img1.square(), window)
            sigma22 = filter_img_2d(img2.square(), window)
            sigma12 = filter_img_2d(img1 * img2, window)
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1.square()
        sigma22 = img2.square()
        sigma12 = img1 * img2

    mu11 = mu1.square()
    mu22 = mu2.square()
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val)**2
    c2 = (k2 * max_val)**2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = torch.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)),
                      dim=(1, 2, 3))  # Return for each image individually.
    cs = torch.mean(v1 / v2, dim=(1, 2, 3))
    return ssim, cs


def compute_pr_distances(row_features,
                         col_features,
                         col_batch_size=10000):
    dist_batches = []
    for col_batch in col_features.split(col_batch_size):
        dist_batch = torch.cdist(
            row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        dist_batches.append(dist_batch.cpu())
    return torch.cat(dist_batches, dim=1)


@METRICS.register_module(force=True)
class PR(_PR):

    def __init__(
            self,
            num_images=None,
            image_shape=None,
            feats_pkl=None,
            k=3,
            bgr2rgb=True,
            vgg16_script=None,
            inception_args=None,
            row_batch_size=10000,
            col_batch_size=10000):
        super(_PR, self).__init__(num_images, image_shape)

        self.feats_pkl = feats_pkl

        self.vgg16 = self.inception_net = None
        self.device = 'cpu'

        if vgg16_script is not None:
            mmcv.print_log('loading vgg16 for improved precision and recall...',
                           'mmgen')
            if os.path.isfile(vgg16_script):
                self.vgg16 = torch.jit.load('work_dirs/cache/vgg16.pt').eval()
                self.use_tero_scirpt = True
            else:
                mmcv.print_log(
                    'Cannot load Tero\'s script module. Use official '
                    'vgg16 instead', 'mmgen')
                self.vgg16 = models.vgg16(pretrained=True).eval()
                self.use_tero_scirpt = False
            if torch.cuda.is_available():
                self.vgg16 = self.vgg16.cuda()
                self.device = 'cuda'
        elif inception_args is not None:
            self.inception_net, self.inception_style = load_inception(
                inception_args, 'FID')
            if torch.cuda.is_available():
                self.inception_net = self.inception_net.cuda()
                self.device = 'cuda'
        else:
            raise ValueError('Please provide either vgg16_script or inception_args')

        self.k = k
        self.bgr2rgb = bgr2rgb
        self.row_batch_size = row_batch_size
        self.col_batch_size = col_batch_size

    def prepare(self):
        self.features_of_reals = []
        self.features_of_fakes = []
        if self.feats_pkl is not None and mmcv.is_filepath(
                self.feats_pkl):
            with open(self.feats_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.features_of_reals = [torch.from_numpy(feat) for feat in reference['features_of_reals']]
                self.num_real_feeded = reference['num_real_feeded']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.feats_pkl}',
                    'mmgen')

    def extract_features(self, batch):
        if self.vgg16 is not None:
            if self.use_tero_scirpt:
                batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                feat = self.vgg16(batch, return_features=True)
            else:
                batch = F.interpolate(batch, size=(224, 224))
                before_fc = self.vgg16.features(batch)
                before_fc = before_fc.view(-1, 7 * 7 * 512)
                feat = self.vgg16.classifier[:4](before_fc)
        else:
            if self.inception_style == 'StyleGAN':
                batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                feat = self.inception_net(batch, return_features=True)
            else:
                feat = self.inception_net(batch)[0].view(batch.shape[0], -1)
        return feat

    @torch.no_grad()
    def feed_op(self, batch, mode):
        batch = batch.to(self.device)
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0]]

        feat = self.extract_features(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(feat) for _ in range(ws)]
            dist.all_gather(placeholder, feat)
            feat = torch.cat(placeholder, dim=0)

        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            if mode == 'reals':
                self.features_of_reals.append(feat)
            elif mode == 'fakes':
                self.features_of_fakes.append(feat)
            else:
                raise ValueError(f'{mode} is not a implemented feed mode.')

    def feed(self, batch, mode):
        if self.num_images is not None:
            return super().feed(batch, mode)
        else:
            self.feed_op(batch, mode)

    @torch.no_grad()
    def summary(self):
        gen_features = torch.cat(self.features_of_fakes)
        real_features = torch.cat(self.features_of_reals).to(device=gen_features.device)

        self._result_dict = {}

        for name, manifold, probes in [
            ('precision', real_features, gen_features),
            ('recall', gen_features, real_features)
        ]:
            kth = []
            for manifold_batch in manifold.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=manifold_batch,
                    col_features=manifold,
                    col_batch_size=self.col_batch_size)
                kth.append(
                    distance.to(torch.float32).kthvalue(self.k + 1).values.to(torch.float16))
            kth = torch.cat(kth)
            pred = []
            for probes_batch in probes.split(self.row_batch_size):
                distance = compute_pr_distances(
                    row_features=probes_batch,
                    col_features=manifold,
                    col_batch_size=self.col_batch_size)
                pred.append((distance <= kth).any(dim=1))
            self._result_dict[name] = float(torch.cat(pred).to(torch.float32).mean())

        precision = self._result_dict['precision']
        recall = self._result_dict['recall']
        self._result_str = f'precision: {precision}, recall:{recall}'
        return self._result_dict

    def clear_fake_data(self):
        self.features_of_fakes = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()
        if clear_reals:
            self.features_of_reals = []
            self.num_real_feeded = 0


@METRICS.register_module(force=True)
class FID(_FID):

    def __init__(self,
                 num_images=None,
                 image_shape=None,
                 inception_pkl=None,
                 bgr2rgb=True,
                 inception_args=dict(normalize_input=False)):
        super().__init__(
            num_images,
            image_shape=image_shape,
            inception_pkl=inception_pkl,
            bgr2rgb=bgr2rgb,
            inception_args=inception_args)

    def prepare(self):
        if self.inception_pkl is not None and mmcv.is_filepath(self.inception_pkl):
            if self.inception_pkl.startswith('huggingface://'):
                self.inception_pkl = download_from_huggingface(self.inception_pkl)
            elif self.inception_pkl.startswith(('http://', 'https://')):
                self.inception_pkl = download_from_url(self.inception_pkl)
            with open(self.inception_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.inception_pkl}',
                    'mmgen')
            self.num_real_feeded = self.num_images

    @torch.no_grad()
    def summary(self):
        # calculate reference inception stat
        if self.real_mean is None:
            feats = torch.cat(self.real_feats, dim=0)
            if self.num_images is not None:
                assert feats.shape[0] >= self.num_images
                feats = feats[:self.num_images]
            feats_np = feats.numpy()
            self.real_mean = np.mean(feats_np, 0)
            self.real_cov = np.cov(feats_np, rowvar=False)

        # calculate fake inception stat
        fake_feats = torch.cat(self.fake_feats, dim=0)
        if self.num_images is not None:
            assert fake_feats.shape[0] >= self.num_images
            fake_feats = fake_feats[:self.num_images]
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        # calculate distance between real and fake statistics
        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean, self.real_cov)

        # results for print/table
        self._result_str = (f'{fid:.4f} ({mean:.5f}/{cov:.5f})')
        # results for log_buffer
        self._result_dict = dict(fid=fid, fid_mean=mean, fid_cov=cov)

        return fid, mean, cov

    def feed(self, batch, mode):
        if self.num_images is not None:
            return super().feed(batch, mode)
        else:
            self.feed_op(batch, mode)


@METRICS.register_module()
class FIDKID(FID):
    name = 'FIDKID'

    def __init__(self,
                 num_images=None,
                 num_subsets=100,
                 max_subset_size=1000,
                 **kwargs):
        super().__init__(num_images=num_images, **kwargs)
        self.num_subsets = num_subsets
        self.max_subset_size = max_subset_size
        self.real_feats_np = None

    def prepare(self):
        if self.inception_pkl is not None and mmcv.is_filepath(
                self.inception_pkl):
            with open(self.inception_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                self.real_feats_np = reference['feats_np']
                mmcv.print_log(
                    f'Load reference inception pkl from {self.inception_pkl}',
                    'mmgen')
            self.num_real_feeded = self.num_images

    @staticmethod
    def _calc_kid(real_feat, fake_feat, num_subsets, max_subset_size):
        """Refer to the implementation from:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py#L18  # noqa
        Args:
            real_feat (np.array): Features of the real samples.
            fake_feat (np.array): Features of the fake samples.
            num_subsets (int): Number of subsets to calculate KID.
            max_subset_size (int): The max size of each subset.
        Returns:
            float: The calculated kid metric.
        """
        n = real_feat.shape[1]
        m = min(min(real_feat.shape[0], fake_feat.shape[0]), max_subset_size)
        t = 0
        for _ in range(num_subsets):
            x = fake_feat[np.random.choice(
                fake_feat.shape[0], m, replace=False)]
            y = real_feat[np.random.choice(
                real_feat.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
            b = (x @ y.T / n + 1)**3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    @torch.no_grad()
    def summary(self):
        if self.real_feats_np is None:
            feats = torch.cat(self.real_feats, dim=0)
            if self.num_images is not None:
                assert feats.shape[0] >= self.num_images
                feats = feats[:self.num_images]
            feats_np = feats.numpy()
            self.real_feats_np = feats_np
            self.real_mean = np.mean(feats_np, 0)
            self.real_cov = np.cov(feats_np, rowvar=False)

        fake_feats = torch.cat(self.fake_feats, dim=0)
        if self.num_images is not None:
            assert fake_feats.shape[0] >= self.num_images
            fake_feats = fake_feats[:self.num_images]
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)

        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)
        kid = self._calc_kid(self.real_feats_np, fake_feats_np, self.num_subsets,
                             self.max_subset_size) * 1000

        self._result_str = f'{fid:.4f} ({mean:.5f}/{cov:.5f}), {kid:.4f}'
        self._result_dict = dict(fid=fid, fid_mean=mean, fid_cov=cov, kid=kid)

        return fid, mean, cov, kid


@METRICS.register_module()
class InceptionMetrics(Metric):
    name = 'InceptionMetrics'

    def __init__(self,
                 num_images=None,
                 reference_pkl=None,
                 bgr2rgb=False,
                 resize=True,
                 inception_args=dict(
                    type='StyleGAN',
                    inception_path=TERO_INCEPTION_URL),
                 use_kid=False,
                 kid_num_subsets=100,
                 kid_max_subset_size=1000,
                 pr_k=3,
                 pr_row_batch_size=10000,
                 pr_col_batch_size=10000,
                 is_splits=10):
        super().__init__(num_images)
        self.reference_pkl = reference_pkl
        self.real_feats = []
        self.fake_feats = []
        self.preds = []
        self.real_mean = None
        self.real_cov = None
        self.bgr2rgb = bgr2rgb
        self.resize = resize
        self.device = 'cpu'

        logger = get_root_logger()
        ori_level = logger.level
        logger.setLevel('ERROR')
        self.inception_net, self.inception_style = load_inception(
            inception_args, 'FID')
        logger.setLevel(ori_level)

        if torch.cuda.is_available():
            self.inception_net = self.inception_net.cuda()
            self.device = 'cuda'
        self.inception_net.eval()

        self.use_kid = use_kid
        self.kid_num_subsets = kid_num_subsets
        self.kid_max_subset_size = kid_max_subset_size
        self.real_feats_np = None

        self.pr_k = pr_k
        self.pr_row_batch_size = pr_row_batch_size
        self.pr_col_batch_size = pr_col_batch_size

        self.is_splits = is_splits

    def prepare(self):
        self.real_feats = []
        self.real_feats_np = None
        self.fake_feats = []
        self.preds = []
        if self.reference_pkl is not None and mmcv.is_filepath(self.reference_pkl):
            if self.reference_pkl.startswith('huggingface://'):
                self.reference_pkl = download_from_huggingface(self.reference_pkl)
            elif self.reference_pkl.startswith(('http://', 'https://')):
                self.reference_pkl = download_from_url(self.reference_pkl)
            with open(self.reference_pkl, 'rb') as f:
                reference = pickle.load(f)
                self.real_mean = reference['mean']
                self.real_cov = reference['cov']
                self.real_feats_np = reference['real_feats_np']
                self.real_feats = [torch.from_numpy(reference['real_feats_np'])]
                self.num_real_feeded = reference['num_real_feeded']

    @staticmethod
    def _calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
        """Refer to the implementation from:

        https://github.com/rosinality/stylegan2-pytorch/blob/master/fid.py#L34
        """
        cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

        if not np.isfinite(cov_sqrt).all():
            print('product of cov matrices is singular')
            offset = np.eye(sample_cov.shape[0]) * eps
            cov_sqrt = linalg.sqrtm(
                (sample_cov + offset) @ (real_cov + offset))

        if np.iscomplexobj(cov_sqrt):
            if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
                m = np.max(np.abs(cov_sqrt.imag))

                raise ValueError(f'Imaginary component {m}')

            cov_sqrt = cov_sqrt.real

        mean_diff = sample_mean - real_mean
        mean_norm = mean_diff @ mean_diff

        trace = np.trace(sample_cov) + np.trace(
            real_cov) - 2 * np.trace(cov_sqrt)

        fid = mean_norm + trace

        return fid, mean_norm, trace

    @staticmethod
    def _calc_kid(real_feat, fake_feat, num_subsets, max_subset_size):
        """Refer to the implementation from:
        https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/metrics/kernel_inception_distance.py#L18  # noqa
        Args:
            real_feat (np.array): Features of the real samples.
            fake_feat (np.array): Features of the fake samples.
            num_subsets (int): Number of subsets to calculate KID.
            max_subset_size (int): The max size of each subset.
        Returns:
            float: The calculated kid metric.
        """
        n = real_feat.shape[1]
        m = min(min(real_feat.shape[0], fake_feat.shape[0]), max_subset_size)
        t = 0
        for _ in range(num_subsets):
            x = fake_feat[np.random.choice(
                fake_feat.shape[0], m, replace=False)]
            y = real_feat[np.random.choice(
                real_feat.shape[0], m, replace=False)]
            a = (x @ x.T / n + 1)**3 + (y @ y.T / n + 1)**3
            b = (x @ y.T / n + 1)**3
            t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m

        kid = t / num_subsets / m
        return float(kid)

    def extract_features(self, batch):
        if self.resize:
            batch = F.interpolate(batch, size=(299, 299), mode='bicubic')
        assert self.inception_style == 'StyleGAN'
        batch = (batch * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        feat = self.inception_net(batch, return_features=True)
        pred = F.linear(feat, self.inception_net.output.weight).softmax(dim=1)
        return feat, pred

    @torch.no_grad()
    def feed_op(self, batch, mode):
        if self.bgr2rgb:
            batch = batch[:, [2, 1, 0]]
        batch = batch.to(self.device)

        feat, pred = self.extract_features(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(feat) for _ in range(ws)]
            dist.all_gather(placeholder, feat)
            feat = torch.cat(placeholder, dim=0)
            if mode == 'fakes':
                placeholder = [torch.zeros_like(pred) for _ in range(ws)]
                dist.all_gather(placeholder, pred)
                pred = torch.cat(placeholder, dim=0)

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            if mode == 'reals':
                self.real_feats.append(feat.cpu())
            elif mode == 'fakes':
                self.fake_feats.append(feat.cpu())
                self.preds.append(pred.cpu().numpy())
            else:
                raise ValueError(
                    f"The expected mode should be set to 'reals' or 'fakes,\
                    but got '{mode}'")

    def feed(self, batch, mode):
        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()
            if mode == 'reals':
                if self.num_real_feeded == self.num_real_need:
                    return 0

                if isinstance(batch, dict):
                    batch_size = [v for v in batch.values()][0].shape[0]
                    end = min(batch_size,
                              self.num_real_need - self.num_real_feeded)
                    batch_to_feed = {k: v[:end, ...] for k, v in batch.items()}
                else:
                    batch_size = batch.shape[0]
                    end = min(batch_size,
                              self.num_real_need - self.num_real_feeded)
                    batch_to_feed = batch[:end, ...]

                global_end = min(batch_size * ws,
                                 self.num_real_need - self.num_real_feeded)
                self.feed_op(batch_to_feed, mode)
                self.num_real_feeded += global_end
                return end

            elif mode == 'fakes':
                if self.num_fake_feeded == self.num_fake_need:
                    return 0

                batch_size = batch.shape[0]
                end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
                if isinstance(batch, dict):
                    batch_to_feed = {k: v[:end, ...] for k, v in batch.items()}
                else:
                    batch_to_feed = batch[:end, ...]

                global_end = min(batch_size * ws,
                                 self.num_fake_need - self.num_fake_feeded)
                self.feed_op(batch_to_feed, mode)
                self.num_fake_feeded += global_end
                return end
            else:
                raise ValueError(
                    'The expected mode should be set to \'reals\' or \'fakes\','
                    f'but got \'{mode}\'')

    @torch.no_grad()
    def summary(self):
        real_feats = torch.cat(self.real_feats, dim=0)
        fake_feats = torch.cat(self.fake_feats, dim=0)
        if self.num_images is not None:
            assert real_feats.shape[0] >= self.num_images
            assert fake_feats.shape[0] >= self.num_images
            real_feats = real_feats[:self.num_images]
            fake_feats = fake_feats[:self.num_images]

        if self.real_feats_np is None:
            real_feats_np = real_feats.numpy()
            self.real_feats_np = real_feats_np
            self.real_mean = np.mean(real_feats_np, 0)
            self.real_cov = np.cov(real_feats_np, rowvar=False)

        self._result_dict = dict()

        # FID
        fake_feats_np = fake_feats.numpy()
        fake_mean = np.mean(fake_feats_np, 0)
        fake_cov = np.cov(fake_feats_np, rowvar=False)
        fid, mean, cov = self._calc_fid(fake_mean, fake_cov, self.real_mean,
                                        self.real_cov)
        self._result_dict.update(fid=fid)
        _result_str = f'FID: {fid:.4f} ({mean:.4f}/{cov:.4f})'

        # KID
        if self.use_kid:
            kid = self._calc_kid(self.real_feats_np, fake_feats_np, self.kid_num_subsets,
                                 self.kid_max_subset_size) * 1000
            self._result_dict.update(kid=kid)
            _result_str += f', KID: {kid:.4f}'
        else:
            kid = None

        # PR
        for name, manifold, probes in [
            ('precision', real_feats, fake_feats),
            ('recall', fake_feats, real_feats)
        ]:
            kth = []
            for manifold_batch in manifold.split(self.pr_row_batch_size):
                distance = compute_pr_distances(
                    row_features=manifold_batch,
                    col_features=manifold,
                    col_batch_size=self.pr_col_batch_size)
                kth.append(
                    distance.to(torch.float32).kthvalue(self.pr_k + 1).values.to(torch.float16))
            kth = torch.cat(kth)
            pred = []
            for probes_batch in probes.split(self.pr_row_batch_size):
                distance = compute_pr_distances(
                    row_features=probes_batch,
                    col_features=manifold,
                    col_batch_size=self.pr_col_batch_size)
                pred.append((distance <= kth).any(dim=1))
            self._result_dict[name] = float(torch.cat(pred).to(torch.float32).mean())
        precision = self._result_dict['precision']
        recall = self._result_dict['recall']
        _result_str += f', Precision: {precision:.5f}, Recall:{recall:.5f}'

        # IS
        split_scores = []
        self.preds = np.concatenate(self.preds, axis=0)
        if self.num_images is not None:
            assert self.preds.shape[0] >= self.num_images
            self.preds = self.preds[:self.num_images]
        num_preds = self.preds.shape[0]
        for k in range(self.is_splits):
            part = self.preds[k * (num_preds // self.is_splits):(k + 1) * (num_preds // self.is_splits), :]
            py = np.mean(part, axis=0)
            scores = []
            for i in range(part.shape[0]):
                pyx = part[i, :]
                scores.append(entropy(pyx, py))
            split_scores.append(np.exp(np.mean(scores)))
        is_mean = np.mean(split_scores)
        self._result_dict.update({'is': is_mean})
        _result_str = f', IS: {is_mean:.2f}'

        self._result_str = _result_str

        return fid, kid, precision, recall, is_mean

    def clear_fake_data(self):
        self.fake_feats = []
        self.preds = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()
        if clear_reals:
            self.real_feats = []
            self.real_feats_np = None
            self.num_real_feeded = 0


@METRICS.register_module()
class ColorStats(Metric):
    name = 'ColorStats'

    def __init__(self,
                 num_images=None):
        super().__init__(num_images)

    def prepare(self):
        self.stats = []

    @staticmethod
    def srgb_to_linear(c):
        threshold = 0.04045
        below = c <= threshold
        out = torch.where(
            below, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
        return out

    @staticmethod
    def linear_to_srgb(c):
        threshold = 0.0031308
        below = c <= threshold
        out = torch.where(
            below, 12.92 * c, 1.055 * c ** (1.0 / 2.4) - 0.055)
        return out

    def rgb_to_grayscale_srgb(self, img_srgb):
        img_lin = self.srgb_to_linear(img_srgb)
        R_lin = img_lin[..., 0]
        G_lin = img_lin[..., 1]
        B_lin = img_lin[..., 2]
        Y_lin = 0.2126 * R_lin + 0.7152 * G_lin + 0.0722 * B_lin
        gray_srgb = self.linear_to_srgb(Y_lin)
        return gray_srgb

    @staticmethod
    def srgb_to_hsv_saturation(img_srgb):
        c_max = torch.amax(img_srgb, dim=-3)
        c_min = torch.amin(img_srgb, dim=-3)
        delta = c_max - c_min
        sat = delta / c_max.clamp(min=1e-5)
        return sat

    def compute_stats(self, batch):
        batch = (batch / 2 + 0.5).clamp(0, 1)
        gray = self.rgb_to_grayscale_srgb(batch).flatten(1)
        contrast, brightness = torch.std_mean(gray, dim=1)
        saturation = self.srgb_to_hsv_saturation(batch).flatten(1).mean(dim=1)
        return torch.stack([brightness, contrast, saturation], dim=-1)

    @torch.no_grad()
    def feed_op(self, batch, mode):
        stats = self.compute_stats(batch)

        if dist.is_initialized():
            ws = dist.get_world_size()
            placeholder = [torch.zeros_like(stats) for _ in range(ws)]
            dist.all_gather(placeholder, stats)
            stats = torch.cat(placeholder, dim=0)

        # in distributed training, we only collect features at rank-0.
        if (dist.is_initialized() and dist.get_rank() == 0) or not dist.is_initialized():
            self.stats.append(stats.cpu())

    def feed(self, batch, mode):
        if mode == 'reals':
            return 0

        if self.num_images is None:
            self.feed_op(batch, mode)

        else:
            _, ws = get_dist_info()

            if self.num_fake_feeded == self.num_fake_need:
                return 0

            batch_size = batch.shape[0]
            end = min(batch_size, self.num_fake_need - self.num_fake_feeded)
            if isinstance(batch, dict):
                batch_to_feed = {k: v[:end, ...] for k, v in batch.items()}
            else:
                batch_to_feed = batch[:end, ...]

            global_end = min(batch_size * ws,
                             self.num_fake_need - self.num_fake_feeded)
            self.feed_op(batch_to_feed, mode)
            self.num_fake_feeded += global_end
            return end

    @torch.no_grad()
    def summary(self):
        stats = torch.cat(self.stats, dim=0).mean(dim=0)
        brightness, contrast, saturation = stats.tolist()
        self._result_dict = dict(
            brightness=brightness, contrast=contrast, saturation=saturation)
        self._result_str = f'Brightness: {brightness:.4f}, Contrast: {contrast:.4f}, Saturation: {saturation:.4f}'
        return brightness, contrast, saturation

    def clear_fake_data(self):
        self.stats = []
        self.num_fake_feeded = 0

    def clear(self, clear_reals=False):
        self.clear_fake_data()
