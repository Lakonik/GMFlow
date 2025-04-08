# Gaussian Mixture Flow Matching Models (GMFlow)

Official PyTorch implementation of the paper:

**Gaussian Mixture Flow Matching Models [[arXiv](https://arxiv.org/abs/2504.05304)]**
<br>
[Hansheng Chen](https://lakonik.github.io/)<sup>1</sup>, 
[Kai Zhang](https://kai-46.github.io/website/)<sup>2</sup>,
[Hao Tan](https://research.adobe.com/person/hao-tan/)<sup>2</sup>,
[Zexiang Xu](https://zexiangxu.github.io/)<sup>3</sup>, 
[Fujun Luan](https://research.adobe.com/person/fujun/)<sup>2</sup>,
[Leonidas Guibas](https://geometry.stanford.edu/?member=guibas)<sup>1</sup>,
[Gordon Wetzstein](http://web.stanford.edu/~gordonwz/)<sup>1</sup>, 
[Sai Bi](https://sai-bi.github.io/)<sup>2</sup><br>
<sup>1</sup>Stanford University, <sup>2</sup>Adobe Research, <sup>3</sup>Hillbot
<br>

<img src="gmdit.png" width="600"  alt=""/>

<img src="gmdit_results.png" width="1000"  alt=""/>

## Highlights

GMFlow is an extension of diffusion/flow matching models.

- **Gaussian Mixture Output**: GMFlow expands the network's output layer to predict a Gaussian Mixture (GM) distribution of flow velocity. Standard diffusion/flow matching models are special cases of GMFlow with a single Gaussian component.

- **Precise Few-Step Sampling**: GMFlow introduces novel **GM-SDE** and **GM-ODE** solvers that leverage analytic denoising distributions and velocity fields for precise few-step sampling.

- **Improved Classifier-Free Guidance (CFG)**: GMFlow introduces a **probabilistic guidance** scheme that mitigates the over-saturation issues of CFG and improves image generation quality.

- **Efficiency**: GMFlow maintains similar training and inference costs to standard diffusion/flow matching models.

## Installation

The code has been tested in the environment described as follows:

- Linux (tested on Ubuntu 20 and above)
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) 11.8 and above
- [PyTorch](https://pytorch.org/get-started/previous-versions/) 2.1 and above

Other dependencies can be installed via `pip install -r requirements.txt`. 

An example of installation commands is shown below (assuming you have already installed CUDA Toolkit and configured the environment variables):

```bash
# Create conda environment
conda create -y -n gmflow python=3.10 numpy=1.26 ninja
conda activate gmflow

# Goto https://pytorch.org/ to select the appropriate version
pip install torch torchvision

# Clone this repo and install other dependencies
git clone https://github.com/Lakonik/GMFlow && cd gmflow
pip install -r requirements.txt
```

This codebase may work on Windows systems, but it has not been tested extensively.

## GM-DiT ImageNet 256x256

### Inference

We provide a [Diffusers pipeline](lib/pipelines/gmdit_pipeline.py) for easy inference. The following code demonstrates how to sample images from the pretrained GM-DiT model using the GM-ODE 2 solver and the GM-SDE 2 solver.

```python
import torch
from huggingface_hub import snapshot_download
from lib.models.diffusions.schedulers import FlowEulerODEScheduler, GMFlowSDEScheduler
from lib.pipelines.gmdit_pipeline import GMDiTPipeline

# Currently the pipeline can only load local checkpoints, so we need to download the checkpoint first
ckpt = snapshot_download(repo_id='Lakonik/gmflow_imagenet_k8_ema')
pipe = GMDiTPipeline.from_pretrained(ckpt, variant='bf16', torch_dtype=torch.bfloat16)
pipe = pipe.to('cuda')

# Pick words that exist in ImageNet
words = ['jay', 'magpie']
class_ids = pipe.get_label_ids(words)

# Sample using GM-ODE 2 solver
pipe.scheduler = FlowEulerODEScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(42)
output = pipe(
    class_labels=class_ids,
    guidance_scale=0.45,
    num_inference_steps=32,
    num_inference_substeps=4,
    output_mode='mean',
    order=2,
    generator=generator)
for i, (word, image) in enumerate(zip(words, output.images)):
    image.save(f'{i:03d}_{word}_gmode2_step32.png')

# Sample using GM-SDE 2 solver (the first run may be slow due to CUDA compilation)
pipe.scheduler = GMFlowSDEScheduler.from_config(pipe.scheduler.config)
generator = torch.manual_seed(42)
output = pipe(
    class_labels=class_ids,
    guidance_scale=0.45,
    num_inference_steps=32,
    num_inference_substeps=1,
    output_mode='sample',
    order=2,
    generator=generator)
for i, (word, image) in enumerate(zip(words, output.images)):
    image.save(f'{i:03d}_{word}_gmsde2_step32.png')
```

The results will be saved under the current directory.

<img src="example_results.png" width="800"  alt=""/>

### Before Training: Data Preparation

Download [ILSVRC2012_img_train.tar](https://www.image-net.org/challenges/LSVRC/2012/2012-downloads.php) and the [metadata](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz). Extract the downloaded archives according to the following folder tree (or use symlinks).
```
./
├── configs/
├── data/
│   └── imagenet/
│       ├── train/
│       │   ├── n01440764/
│       │   │   ├── n01440764_10026.JPEG
│       │   │   ├── n01440764_10027.JPEG
│       │   │   …
│       │   ├── n01443537/
│       │   …
│       ├── imagenet1000_clsidx_to_labels.txt
│       ├── train.txt
|       …
├── lib/
├── tools/
…
```

Run the following command to prepare the ImageNet dataset using DDP on 1 node with 8 GPUs

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/prepare_imagenet_dit.py
```

### Training

Run the following command to train the model using DDP on 1 node with 8 GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/train.py configs/gmflow_imagenet_k8_8gpus.py --launcher pytorch --diff_seed
```

Alternatively, you can start single-node DDP training from a Python script:

```bash
python train.py configs/gmflow_imagenet_k8_8gpus.py --gpu-ids 0 1 2 3 4 5 6 7
```

The config in [gmflow_imagenet_k8_8gpus.py](configs/gmflow_imagenet_k8_8gpus.py) specifies a training batch size of 512 images per GPU and an inference batch size of 125 images per GPU. Training requires 32GB of VRAM per GPU, and the validation step requires an additional 8GB of VRAM per GPU. If you are using 32GB GPUs, you can disable the validation step by adding the `--no-validate` flag to the training command. Alternatively, you can also edit the config file to adjust the batch sizes.

By default, checkpoints will be saved into [checkpoints/](checkpoints/), logs will be saved into [work_dirs/](work_dirs/), and sampled images will be saved into [viz/](viz/).

#### Resuming Training

If existing checkpoints are found, the training will automatically resume from the latest checkpoint.

#### Tensorboard

The logs can be plotted using Tensorboard. Run the following command to start Tensorboard:

```bash
tensorboard --logdir work_dirs/
```

### Evaluation

After training, to conduct a complete evaluation of the model under varying guidance scales, run the following command to start DDP evaluation on 1 node with 8 GPUs:

```bash
torchrun --nnodes=1 --nproc_per_node=8 tools/test.py configs/gmflow_imagenet_k8_test.py checkpoints/gmflow_imagenet_k8_8gpus/latest.pth --launcher pytorch --diff_seed
```

Alternatively, you can start single-node DDP evaluation from a Python script:
```bash
python test.py configs/gmflow_imagenet_k8_test.py checkpoints/gmflow_imagenet_k8_8gpus/latest.pth --gpu-ids 0 1 2 3 4 5 6 7
```

The config in [gmflow_imagenet_k8_test.py](configs/gmflow_imagenet_k8_test.py) specifies an inference batch size of 125 images per GPU, which requires 35GB of VRAM per GPU. You can edit the config file to adjust the batch size.

The evaluation results will be saved to where the checkpoint is located, and the sampled images will be saved into [viz/](viz/).

## Toy Model on 2D Checkerboard

We provide a minimal GMFlow trainer in [train_toymodel.py](train_toymodel.py) for the toy model on the 2D checkerboard dataset. Run the following command to train the model:

```bash
python train_toymodel.py -k 64
```

This minimal trainer does not support transition loss and EMA. To reproduce the results in the paper, you can use the following command to start the full trainer:

```bash
python train.py configs/gmflow_checkerboard_k64.py --gpu-ids 0
```

This full trainer is not optimized for the simple 2D checkerboard dataset, so GPU usage may be inefficient.

## Essential Code

- Training
    - [train_toymodel.py](train_toymodel.py): A simplified training script for the 2D checkerboard experiment.
    - [gmflow.py](lib/models/diffusions/gmflow.py): The `forward_train` method contains the full training loop.
- Inference
    - [gmdit_pipeline.py](lib/pipelines/gmdit_pipeline.py): Full sampling code in the style of Diffusers.
    - [gmflow.py](lib/models/diffusions/gmflow.py): The `forward_test` method contains the same full sampling loop.
- Network
    - [gmflow.py](lib/models/architecture/gmflow.py): GMDiT and SpectrumMLP
    - [toymodels.py](lib/models/architecture/toymodels.py): MLP toy model for the 2D checkerboard experiment.
- GM math operations
    - [gmflow_ops](lib/ops/gmflow_ops/): A complete library of analytical operations for GM and Gaussian distributions.

## Citation
```
@misc{gmflow,
      title={Gaussian Mixture Flow Matching Models}, 
      author={Hansheng Chen and Kai Zhang and Hao Tan and Zexiang Xu and Fujun Luan and Leonidas Guibas and Gordon Wetzstein and Sai Bi},
      year={2025},
      eprint={2504.05304},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2504.05304}, 
}
```
