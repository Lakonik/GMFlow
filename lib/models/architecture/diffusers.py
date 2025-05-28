import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict, Any, Optional, Union, Tuple
from collections import OrderedDict
from transformers import CLIPTextModel as _CLIPTextModel
from transformers import CLIPTextModelWithProjection, T5EncoderModel, CLIPTokenizer, T5TokenizerFast
from diffusers.models import UNet2DConditionModel as _UNet2DConditionModel
from diffusers.models import ControlNetModel, AutoencoderKL, DiTTransformer2DModel
from diffusers.models.attention_processor import Attention
from diffusers.models.autoencoders.vae import Decoder
from diffusers.models.resnet import ResnetBlock2D
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.embeddings import Timesteps

from mmcv.runner import load_checkpoint, _load_checkpoint, load_state_dict
from mmcv.cnn.utils import constant_init, kaiming_init
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger

from ...core import rgetattr


def ceildiv(a, b):
    return -(a // -b)


def autocast_patch(module, dtype=None, enabled=True):

    def make_new_forward(old_forward, dtype, enabled):
        def new_forward(*args, **kwargs):
            with torch.autocast(device_type='cuda', dtype=dtype, enabled=enabled):
                result = old_forward(*args, **kwargs)
            return result

        return new_forward

    module.forward = make_new_forward(module.forward, dtype, enabled)


def unet_enc(
        unet,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        added_cond_kwargs=None):
    # 0. center input if necessary
    if unet.config.center_input_sample:
        sample = 2 * sample - 1.0

    # 1. time
    t_emb = unet.get_time_embed(sample=sample, timestep=timestep)
    emb = unet.time_embedding(t_emb)
    aug_emb = unet.get_aug_embed(
        emb=emb, encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)
    emb = emb + aug_emb if aug_emb is not None else emb

    if unet.time_embed_act is not None:
        emb = unet.time_embed_act(emb)

    encoder_hidden_states = unet.process_encoder_hidden_states(
        encoder_hidden_states=encoder_hidden_states, added_cond_kwargs=added_cond_kwargs)

    # 2. pre-process
    sample = unet.conv_in(sample)

    # 3. down
    down_block_res_samples = (sample,)
    for downsample_block in unet.down_blocks:
        if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
            sample, res_samples = downsample_block(
                hidden_states=sample,
                temb=emb,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

        down_block_res_samples += res_samples

    return emb, down_block_res_samples, sample


def unet_dec(
        unet,
        emb,
        down_block_res_samples,
        sample,
        encoder_hidden_states: torch.Tensor,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        down_block_additional_residuals: Optional[Tuple[torch.Tensor]] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None):
    is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None

    if is_controlnet:
        new_down_block_res_samples = ()

        for down_block_res_sample, down_block_additional_residual in zip(
                down_block_res_samples, down_block_additional_residuals):
            down_block_res_sample = down_block_res_sample + down_block_additional_residual
            new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

        down_block_res_samples = new_down_block_res_samples

    # 4. mid
    if unet.mid_block is not None:
        if hasattr(unet.mid_block, "has_cross_attention") and unet.mid_block.has_cross_attention:
            sample = unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample = unet.mid_block(sample, emb)

    if is_controlnet:
        sample = sample + mid_block_additional_residual

    # 5. up
    for i, upsample_block in enumerate(unet.up_blocks):
        res_samples = down_block_res_samples[-len(upsample_block.resnets):]
        down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

        if hasattr(upsample_block, 'has_cross_attention') and upsample_block.has_cross_attention:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
                encoder_hidden_states=encoder_hidden_states,
                cross_attention_kwargs=cross_attention_kwargs,
            )
        else:
            sample = upsample_block(
                hidden_states=sample,
                temb=emb,
                res_hidden_states_tuple=res_samples,
            )

    # 6. post-process
    if unet.conv_norm_out:
        sample = unet.conv_norm_out(sample)
        sample = unet.conv_act(sample)
    sample = unet.conv_out(sample)

    return sample


@MODULES.register_module()
class UNet2DConditionModel(_UNet2DConditionModel):
    def __init__(self,
                 *args,
                 freeze=True,
                 freeze_exclude=[],
                 pretrained=None,
                 torch_dtype='float32',
                 freeze_exclude_fp32=True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
            for attr in freeze_exclude:
                rgetattr(self, attr).requires_grad_(True)

        self.init_weights(pretrained)
        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

        self.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))

        self.freeze_exclude_fp32 = freeze_exclude_fp32
        if self.freeze_exclude_fp32:
            for attr in freeze_exclude:
                m = rgetattr(self, attr)
                assert isinstance(m, nn.Module)
                m.to(torch.float32)
                autocast_patch(m, enabled=False)

    def init_weights(self, pretrained):
        if pretrained is not None:
            logger = get_root_logger()
            # load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            metadata = getattr(state_dict, '_metadata', OrderedDict())
            state_dict._metadata = metadata
            assert self.conv_in.weight.shape[1] == self.conv_out.weight.shape[0]
            if state_dict['conv_in.weight'].size() != self.conv_in.weight.size():
                assert state_dict['conv_in.weight'].shape[1] == state_dict['conv_out.weight'].shape[0]
                src_chn = state_dict['conv_in.weight'].shape[1]
                tgt_chn = self.conv_in.weight.shape[1]
                assert src_chn < tgt_chn
                convert_mat_out = torch.tile(torch.eye(src_chn), (ceildiv(tgt_chn, src_chn), 1))
                convert_mat_out = convert_mat_out[:tgt_chn]
                convert_mat_in = F.normalize(convert_mat_out.pinverse(), dim=-1)
                state_dict['conv_out.weight'] = torch.einsum(
                    'ts,scxy->tcxy', convert_mat_out, state_dict['conv_out.weight'])
                state_dict['conv_out.bias'] = torch.einsum(
                    'ts,s->t', convert_mat_out, state_dict['conv_out.bias'])
                state_dict['conv_in.weight'] = torch.einsum(
                    'st,csxy->ctxy', convert_mat_in, state_dict['conv_in.weight'])
            load_state_dict(self, state_dict, logger=logger)

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        dtype = sample.dtype
        return super().forward(
            sample, timestep, encoder_hidden_states, return_dict=False, **kwargs)[0].to(dtype)

    def forward_enc(self, sample, timestep, encoder_hidden_states, **kwargs):
        return unet_enc(self, sample, timestep, encoder_hidden_states, **kwargs)

    def forward_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs):
        return unet_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs)


@MODULES.register_module()
class CLIPTextModel(_CLIPTextModel):
    def __init__(
            self,
            *args,
            freeze=True,
            pretrained=None,
            attention_dropout=0.0,
            bos_token_id=0,
            dropout=0.0,
            eos_token_id=2,
            hidden_act='gelu',
            hidden_size=1024,
            initializer_factor=1.0,
            initializer_range=0.02,
            intermediate_size=4096,
            layer_norm_eps=1e-05,
            max_position_embeddings=77,
            model_type='clip_text_model',
            num_attention_heads=16,
            num_hidden_layers=23,
            pad_token_id=1,
            projection_dim=512,
            torch_dtype='float32',
            transformers_version='4.25.0.dev0',
            vocab_size=49408,
            **kwargs):
        cfg = _CLIPTextModel.config_class(
            *args,
            attention_dropout=attention_dropout,
            bos_token_id=bos_token_id,
            dropout=dropout,
            eos_token_id=eos_token_id,
            hidden_act=hidden_act,
            hidden_size=hidden_size,
            initializer_factor=initializer_factor,
            initializer_range=initializer_range,
            intermediate_size=intermediate_size,
            layer_norm_eps=layer_norm_eps,
            max_position_embeddings=max_position_embeddings,
            model_type=model_type,
            num_attention_heads=num_attention_heads,
            num_hidden_layers=num_hidden_layers,
            pad_token_id=pad_token_id,
            projection_dim=projection_dim,
            torch_dtype=torch_dtype,
            transformers_version=transformers_version,
            vocab_size=vocab_size,
            **kwargs)
        self.pretrained = pretrained
        super(CLIPTextModel, self).__init__(cfg)
        self.to(cfg.torch_dtype)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)

    def init_weights(self):
        if self.pretrained is None:
            super(CLIPTextModel, self).init_weights()
        else:
            logger = get_root_logger()
            load_checkpoint(self, self.pretrained, map_location='cpu', strict=False, logger=logger)


@MODULES.register_module()
class VAEDecoder(Decoder, ModelMixin):
    def __init__(
            self,
            in_channels=12,
            out_channels=24,
            up_block_types=('UpDecoderBlock2D',),
            block_out_channels=(64,),
            layers_per_block=2,
            norm_num_groups=32,
            act_fn='silu',
            norm_type='group',
            zero_init_residual=True):
        super(VAEDecoder, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            up_block_types=up_block_types,
            block_out_channels=block_out_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            act_fn=act_fn,
            norm_type=norm_type)
        self.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        self.zero_init_residual = zero_init_residual
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.GroupNorm):
                constant_init(m, 1)

        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, ResnetBlock2D):
                    constant_init(m.conv2, 0)
                elif isinstance(m, Attention):
                    constant_init(m.to_out[0], 0)


@MODULES.register_module()
class PretrainedUNet(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 freeze=True,
                 torch_dtype='float32',
                 checkpointing=True,
                 use_ref_attn=False,
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.unet = _UNet2DConditionModel.from_pretrained(
            from_pretrained, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
        self.unet.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        if use_ref_attn:
            unet_attn_procs = dict()
            for name, attn_proc in self.unet.attn_processors.items():
                unet_attn_procs[name] = ReferenceAttnProc(
                    attn_proc, enabled=name.endswith('attn1.processor'), name=name)
            self.unet.set_attn_processor(unet_attn_procs)
        if checkpointing:
            self.unet.enable_gradient_checkpointing()

    def forward(self, sample, timestep, encoder_hidden_states, **kwargs):
        return self.unet(
            sample, timestep, encoder_hidden_states, return_dict=False, **kwargs)[0]

    def forward_enc(self, sample, timestep, encoder_hidden_states, **kwargs):
        return unet_enc(self.unet, sample, timestep, encoder_hidden_states, **kwargs)

    def forward_dec(self, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs):
        return unet_dec(self.unet, emb, down_block_res_samples, sample, encoder_hidden_states, **kwargs)


@MODULES.register_module()
class PretrainedControlNet(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 freeze=True,
                 torch_dtype='float32',
                 checkpointing=True,
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.controlnet = ControlNetModel.from_pretrained(
            from_pretrained, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
        self.controlnet.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))
        if checkpointing:
            self.controlnet.enable_gradient_checkpointing()

    def forward(self, *args, **kwargs):
        return self.controlnet.forward(*args, **kwargs)


@MODULES.register_module()
class PretrainedCLIPTextModelWithProjection(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 freeze=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.model = CLIPTextModelWithProjection.from_pretrained(
            from_pretrained, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


@MODULES.register_module()
class PretrainedT5EncoderModel(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 freeze=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.model = T5EncoderModel.from_pretrained(
            from_pretrained, **kwargs)
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


@MODULES.register_module()
class PretrainedDiT(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 pos_embed_path=None,
                 freeze=True,
                 torch_dtype='float32',
                 checkpointing=True,
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.dit = DiTTransformer2DModel.from_pretrained(
            from_pretrained, **kwargs)
        if pos_embed_path is not None:
            pos_embed = torch.load(pos_embed_path)['pos_embed']
            self.dit.pos_embed.pos_embed.data.copy_(pos_embed)
        for module in self.dit.modules():
            if isinstance(module, Timesteps):
                module.downscale_freq_shift = 0  # fix diffusers implementation bug
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
        if checkpointing:
            self.dit.enable_gradient_checkpointing()

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None):
        return self.dit.forward(
            hidden_states=hidden_states,
            timestep=timestep,
            class_labels=class_labels,
            cross_attention_kwargs=cross_attention_kwargs,
            return_dict=False)[0]


@MODULES.register_module()
class PretrainedCLIPTokenizer(nn.Module):
    def __init__(self, from_pretrained=None, **kwargs):
        super().__init__()
        self.model = CLIPTokenizer.from_pretrained(from_pretrained, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@MODULES.register_module()
class PretrainedT5TokenizerFast(nn.Module):
    def __init__(self, from_pretrained=None, **kwargs):
        super().__init__()
        self.model = T5TokenizerFast.from_pretrained(from_pretrained, **kwargs)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)


@MODULES.register_module()
class PretrainedVAE(nn.Module):
    def __init__(self,
                 from_pretrained=None,
                 del_encoder=False,
                 del_decoder=False,
                 use_slicing=False,
                 freeze=True,
                 torch_dtype='float32',
                 **kwargs):
        super().__init__()
        if torch_dtype is not None:
            kwargs.update(torch_dtype=getattr(torch, torch_dtype))
        self.vae = AutoencoderKL.from_pretrained(
            from_pretrained, **kwargs)
        if del_encoder:
            del self.vae.encoder
        if del_decoder:
            del self.vae.decoder
        if use_slicing:
            self.vae.enable_slicing()
        self.freeze = freeze
        if self.freeze:
            self.requires_grad_(False)
        self.vae.set_use_memory_efficient_attention_xformers(
            not hasattr(torch.nn.functional, 'scaled_dot_product_attention'))

    def forward(self, *args, **kwargs):
        return self.vae.forward(*args, **kwargs)[0]

    def encode(self, img):
        return self.vae.encode(img).latent_dist.sample() * self.vae.config.scaling_factor

    def decode(self, code):
        return self.vae.decode(code / self.vae.config.scaling_factor, return_dict=False)[0]

    def visualize(self, code, scene_name, viz_dir):
        code_viz = (self(code).to(dtype=torch.float32, device='cpu').numpy().transpose(0, 2, 3, 1) / 2 + 0.5).clip(
            min=0, max=1)
        code_viz = np.round(code_viz * 255).astype(np.uint8)
        for code_viz_single, scene_name_single in zip(code_viz, scene_name):
            plt.imsave(os.path.join(viz_dir, 'scene_' + scene_name_single + '_dec.png'), code_viz_single)


@MODULES.register_module()
class PretrainedVAEDecoder(PretrainedVAE):
    def __init__(self, **kwargs):
        super().__init__(
            del_encoder=True,
            del_decoder=False,
            **kwargs)

    def forward(self, code):
        return super().decode(code)


@MODULES.register_module()
class PretrainedVAEEncoder(PretrainedVAE):
    def __init__(self, **kwargs):
        super().__init__(
            del_encoder=False,
            del_decoder=True,
            **kwargs)

    def forward(self, img):
        return super().encode(img)


class ReferenceAttnProc(torch.nn.Module):
    def __init__(
            self,
            chained_proc,
            enabled=False,
            name=None
    ) -> None:
        super().__init__()
        self.enabled = enabled
        self.chained_proc = chained_proc
        self.name = name

    def __call__(
            self, attn: Attention, hidden_states, encoder_hidden_states=None, attention_mask=None,
            mode='w', ref_dict: dict = None):
        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        if self.enabled:
            if mode == 'w':
                ref_dict[self.name] = encoder_hidden_states
            elif mode == 'r':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict.pop(self.name)], dim=1)
            elif mode == 'm':
                encoder_hidden_states = torch.cat([encoder_hidden_states, ref_dict[self.name]], dim=1)
            else:
                assert False, mode
        res = self.chained_proc(attn, hidden_states, encoder_hidden_states, attention_mask)
        return res
