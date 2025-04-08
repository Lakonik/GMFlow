# Copyright (c) 2025 Hansheng Chen

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional
from functools import partial
from diffusers.models import DiTTransformer2DModel
from diffusers.models.attention import BasicTransformerBlock, _chunked_feed_forward, Attention, FeedForward
from diffusers.models.embeddings import (
    PatchEmbed, Timesteps, CombinedTimestepLabelEmbeddings, TimestepEmbedding, LabelEmbedding)
from diffusers.models.normalization import AdaLayerNormZero
from diffusers.models.modeling_utils import ModelMixin
from diffusers.configuration_utils import register_to_config, ConfigMixin

from mmcv.runner import load_checkpoint
from mmcv.cnn import constant_init, xavier_init
from mmgen.models.builder import MODULES
from mmgen.utils import get_root_logger

from .diffusers import autocast_patch
from ...core import rgetattr


class LabelEmbeddingMod(LabelEmbedding):
    def __init__(self, num_classes, hidden_size, dropout_prob=0.0, use_cfg_embedding=True):
        super(LabelEmbedding, self).__init__()
        if dropout_prob > 0:
            assert use_cfg_embedding
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob


class CombinedTimestepLabelEmbeddingsMod(CombinedTimestepLabelEmbeddings):
    """
    Modified CombinedTimestepLabelEmbeddings for reproducing the original DiT (downscale_freq_shift=0).
    """
    def __init__(
            self, num_classes, embedding_dim, class_dropout_prob=0.1, downscale_freq_shift=0, use_cfg_embedding=True):
        super(CombinedTimestepLabelEmbeddings, self).__init__()
        self.time_proj = Timesteps(num_channels=256, flip_sin_to_cos=True, downscale_freq_shift=downscale_freq_shift)
        self.timestep_embedder = TimestepEmbedding(in_channels=256, time_embed_dim=embedding_dim)
        self.class_embedder = LabelEmbeddingMod(num_classes, embedding_dim, class_dropout_prob, use_cfg_embedding)


class BasicTransformerBlockMod(BasicTransformerBlock):
    """
    Modified BasicTransformerBlock for reproducing the original DiT with shared time and class
    embeddings across all layers.
    """
    def __init__(
            self,
            dim: int,
            num_attention_heads: int,
            attention_head_dim: int,
            dropout=0.0,
            cross_attention_dim: Optional[int] = None,
            activation_fn: str = 'geglu',
            num_embeds_ada_norm: Optional[int] = None,
            attention_bias: bool = False,
            only_cross_attention: bool = False,
            double_self_attention: bool = False,
            upcast_attention: bool = False,
            norm_elementwise_affine: bool = True,
            norm_type: str = 'layer_norm',
            norm_eps: float = 1e-5,
            final_dropout: bool = False,
            attention_type: str = 'default',
            ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
            ada_norm_bias: Optional[int] = None,
            ff_inner_dim: Optional[int] = None,
            ff_bias: bool = True,
            attention_out_bias: bool = True):
        super(BasicTransformerBlock, self).__init__()
        self.only_cross_attention = only_cross_attention
        self.norm_type = norm_type
        self.num_embeds_ada_norm = num_embeds_ada_norm

        assert self.norm_type == 'ada_norm_zero'
        self.norm1 = AdaLayerNormZero(dim, num_embeds_ada_norm)
        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        self.norm2 = None
        self.attn2 = None

        self.norm3 = nn.LayerNorm(dim, norm_eps, norm_elementwise_affine)
        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            class_labels: Optional[torch.LongTensor] = None,
            emb: Optional[torch.Tensor] = None,
            added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None) -> torch.Tensor:
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype, emb=emb)

        if cross_attention_kwargs is None:
            cross_attention_kwargs = dict()
        attn_output = self.attn1(
            norm_hidden_states,
            encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
            attention_mask=attention_mask,
            **cross_attention_kwargs)
        attn_output = gate_msa.unsqueeze(1) * attn_output

        hidden_states = attn_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        norm_hidden_states = self.norm3(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = ff_output + hidden_states
        if hidden_states.ndim == 4:
            hidden_states = hidden_states.squeeze(1)

        return hidden_states


class GMOutput2D(nn.Module):

    def __init__(self,
                 num_gaussians,
                 out_channels,
                 embed_dim,
                 constant_logstd=None,
                 logstd_inner_dim=1024,
                 num_logstd_layers=2,
                 activation_fn='silu'):
        super(GMOutput2D, self).__init__()
        self.num_gaussians = num_gaussians
        self.out_channels = out_channels
        self.embed_dim = embed_dim
        self.constant_logstd = constant_logstd

        if constant_logstd is None:
            if activation_fn == 'gelu-approximate':
                act = partial(nn.GELU, approximate='tanh')
            elif activation_fn == 'silu':
                act = nn.SiLU
            else:
                raise ValueError(f'Unsupported activation function: {activation_fn}')

            assert num_logstd_layers >= 1
            in_dim = self.embed_dim
            logstd_layers = []
            for _ in range(num_logstd_layers - 1):
                logstd_layers.extend([
                    act(),
                    nn.Linear(in_dim, logstd_inner_dim)])
                in_dim = logstd_inner_dim
            self.logstd_layers = nn.Sequential(
                *logstd_layers,
                act(),
                nn.Linear(in_dim, 1))

        self.init_weights()

    def init_weights(self):
        if self.constant_logstd is None:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    xavier_init(m, distribution='uniform')
            constant_init(self.logstd_layers[-1], val=0)

    def forward(self, x, emb):
        bs, c, h, w = x.size()
        means, logweights = x.split([self.num_gaussians * self.out_channels, self.num_gaussians], dim=1)
        means = means.view(bs, self.num_gaussians, self.out_channels, h, w)
        logweights = logweights.view(bs, self.num_gaussians, 1, h, w).log_softmax(dim=1)
        if self.constant_logstd is None:
            logstds = self.logstd_layers(emb).view(bs, 1, 1, 1, 1)
        else:
            logstds = torch.full(
                (bs, 1, 1, 1, 1), self.constant_logstd,
                dtype=x.dtype, device=x.device)
        return dict(
            means=means,
            logweights=logweights,
            logstds=logstds)


class _GMDiTTransformer2DModel(DiTTransformer2DModel):

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            class_dropout_prob=0.0,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: Optional[int] = None,
            num_layers: int = 28,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            attention_bias: bool = True,
            sample_size: int = 32,
            patch_size: int = 2,
            activation_fn: str = 'gelu-approximate',
            num_embeds_ada_norm: Optional[int] = 1000,
            upcast_attention: bool = False,
            norm_type: str = 'ada_norm_zero',
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5):

        super(DiTTransformer2DModel, self).__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gm_channels = num_gaussians * (self.out_channels + 1)
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        self.patch_size = self.config.patch_size
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim)
        self.emb = CombinedTimestepLabelEmbeddingsMod(
            num_embeds_ada_norm, self.inner_dim, class_dropout_prob=0.0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlockMod(
                self.inner_dim,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                num_embeds_ada_norm=None,
                attention_bias=self.config.attention_bias,
                upcast_attention=self.config.upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps)
            for _ in range(self.config.num_layers)])

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_2 = nn.Linear(
            self.inner_dim, self.config.patch_size * self.config.patch_size * self.gm_channels)

        self.gm_out = GMOutput2D(
            num_gaussians,
            self.out_channels,
            self.inner_dim,
            constant_logstd=constant_logstd,
            logstd_inner_dim=logstd_inner_dim,
            num_logstd_layers=gm_num_logstd_layers)

    # https://github.com/facebookresearch/DiT/blob/main/models.py
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks
        for m in self.modules():
            if isinstance(m, AdaLayerNormZero):
                constant_init(m.linear, val=0)

        # Zero-out output layers
        constant_init(self.proj_out_1, val=0)

        self.gm_out.init_weights()

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None):
        # 1. Input
        bs, _, h, w = hidden_states.size()
        height, width = h // self.patch_size, w // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        cond_emb = self.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype)
        dropout_enabled = self.config.class_dropout_prob > 0 and self.training
        if dropout_enabled:
            uncond_emb = self.emb(timestep, torch.full_like(
                class_labels, self.config.num_embeds_ada_norm), hidden_dtype=hidden_states.dtype)

        # 2. Blocks
        for block in self.transformer_blocks:
            if dropout_enabled:
                dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.class_dropout_prob
                emb = torch.where(dropout_mask, uncond_emb, cond_emb)
            else:
                emb = cond_emb

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    emb,
                    use_reentrant=False)

            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    emb=emb)

        # 3. Output
        if dropout_enabled:
            dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.class_dropout_prob
            emb = torch.where(dropout_mask, uncond_emb, cond_emb)
        else:
            emb = cond_emb
        shift, scale = self.proj_out_1(F.silu(emb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]
        hidden_states = self.proj_out_2(hidden_states).reshape(
                bs, height, width, self.patch_size, self.patch_size, self.gm_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.gm_channels, height * self.patch_size, width * self.patch_size)

        return self.gm_out(hidden_states, cond_emb.detach())


@MODULES.register_module()
class GMDiTTransformer2DModel(_GMDiTTransformer2DModel):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            checkpointing=True,
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

        self.freeze_exclude_fp32 = freeze_exclude_fp32
        if self.freeze_exclude_fp32:
            for attr in freeze_exclude:
                m = rgetattr(self, attr)
                assert isinstance(m, nn.Module)
                m.to(torch.float32)
                autocast_patch(m, enabled=False)

        if checkpointing:
            self.enable_gradient_checkpointing()

    def init_weights(self, pretrained=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)


class _GMDiTTransformer2DModelV2(DiTTransformer2DModel):

    @register_to_config
    def __init__(
            self,
            num_gaussians=16,
            constant_logstd=None,
            logstd_inner_dim=1024,
            gm_num_logstd_layers=2,
            class_dropout_prob=0.0,
            num_attention_heads: int = 16,
            attention_head_dim: int = 72,
            in_channels: int = 4,
            out_channels: Optional[int] = None,
            num_layers: int = 28,
            dropout: float = 0.0,
            norm_num_groups: int = 32,
            attention_bias: bool = True,
            sample_size: int = 32,
            patch_size: int = 2,
            activation_fn: str = 'gelu-approximate',
            num_embeds_ada_norm: Optional[int] = 1000,
            upcast_attention: bool = False,
            norm_type: str = 'ada_norm_zero',
            norm_elementwise_affine: bool = False,
            norm_eps: float = 1e-5):

        super(DiTTransformer2DModel, self).__init__()

        # Validate inputs.
        if norm_type != "ada_norm_zero":
            raise NotImplementedError(
                f"Forward pass is not implemented when `patch_size` is not None and `norm_type` is '{norm_type}'."
            )
        elif norm_type == "ada_norm_zero" and num_embeds_ada_norm is None:
            raise ValueError(
                f"When using a `patch_size` and this `norm_type` ({norm_type}), `num_embeds_ada_norm` cannot be None."
            )

        # Set some common variables used across the board.
        self.attention_head_dim = attention_head_dim
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim
        self.out_channels = in_channels if out_channels is None else out_channels
        self.gradient_checkpointing = False

        # 2. Initialize the position embedding and transformer blocks.
        self.height = self.config.sample_size
        self.width = self.config.sample_size

        self.patch_size = self.config.patch_size
        self.pos_embed = PatchEmbed(
            height=self.config.sample_size,
            width=self.config.sample_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            embed_dim=self.inner_dim)
        self.emb = CombinedTimestepLabelEmbeddingsMod(
            num_embeds_ada_norm, self.inner_dim, class_dropout_prob=0.0)

        self.transformer_blocks = nn.ModuleList([
            BasicTransformerBlockMod(
                self.inner_dim,
                self.config.num_attention_heads,
                self.config.attention_head_dim,
                dropout=self.config.dropout,
                activation_fn=self.config.activation_fn,
                num_embeds_ada_norm=None,
                attention_bias=self.config.attention_bias,
                upcast_attention=self.config.upcast_attention,
                norm_type=norm_type,
                norm_elementwise_affine=self.config.norm_elementwise_affine,
                norm_eps=self.config.norm_eps)
            for _ in range(self.config.num_layers)])

        # 3. Output blocks.
        self.norm_out = nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=1e-6)
        self.proj_out_1 = nn.Linear(self.inner_dim, 2 * self.inner_dim)
        self.proj_out_means = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.config.num_gaussians * self.out_channels)
        self.proj_out_logweights = nn.Linear(
            self.inner_dim,
            self.config.patch_size * self.config.patch_size * self.config.num_gaussians)

        if constant_logstd is None:
            assert gm_num_logstd_layers >= 1
            in_dim = self.inner_dim
            logstd_layers = []
            for _ in range(gm_num_logstd_layers - 1):
                logstd_layers.extend([
                    nn.SiLU(),
                    nn.Linear(in_dim, logstd_inner_dim)])
                in_dim = logstd_inner_dim
            self.proj_out_logstds = nn.Sequential(
                *logstd_layers,
                nn.SiLU(),
                nn.Linear(in_dim, 1))

        self.init_weights()

    # https://github.com/facebookresearch/DiT/blob/main/models.py
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.pos_embed.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.pos_embed.proj.bias, 0)

        # Zero-out adaLN modulation layers in DiT blocks
        for m in self.modules():
            if isinstance(m, AdaLayerNormZero):
                constant_init(m.linear, val=0)

        # Zero-out output layers
        constant_init(self.proj_out_1, val=0)
        constant_init(self.proj_out_logweights, val=0)
        constant_init(self.proj_out_logstds[-1], val=0)

    def forward(
            self,
            hidden_states: torch.Tensor,
            timestep: Optional[torch.LongTensor] = None,
            class_labels: Optional[torch.LongTensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None):
        # 1. Input
        bs, _, h, w = hidden_states.size()
        height, width = h // self.patch_size, w // self.patch_size
        hidden_states = self.pos_embed(hidden_states)

        cond_emb = self.emb(
            timestep, class_labels, hidden_dtype=hidden_states.dtype)
        dropout_enabled = self.config.class_dropout_prob > 0 and self.training
        if dropout_enabled:
            uncond_emb = self.emb(timestep, torch.full_like(
                class_labels, self.config.num_embeds_ada_norm), hidden_dtype=hidden_states.dtype)

        # 2. Blocks
        for block in self.transformer_blocks:
            if dropout_enabled:
                dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.class_dropout_prob
                emb = torch.where(dropout_mask, uncond_emb, cond_emb)
            else:
                emb = cond_emb

            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    None,
                    None,
                    None,
                    timestep,
                    cross_attention_kwargs,
                    class_labels,
                    emb,
                    use_reentrant=False)

            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=None,
                    encoder_hidden_states=None,
                    encoder_attention_mask=None,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    emb=emb)

        # 3. Output
        if dropout_enabled:
            dropout_mask = torch.rand((bs, 1), device=hidden_states.device) < self.config.class_dropout_prob
            emb = torch.where(dropout_mask, uncond_emb, cond_emb)
        else:
            emb = cond_emb
        shift, scale = self.proj_out_1(F.silu(emb)).chunk(2, dim=1)
        hidden_states = self.norm_out(hidden_states) * (1 + scale[:, None]) + shift[:, None]

        out_means = self.proj_out_means(hidden_states).reshape(
                bs, height, width, self.patch_size, self.patch_size, self.config.num_gaussians * self.out_channels
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.config.num_gaussians, self.out_channels, height * self.patch_size, width * self.patch_size)
        out_logweights = self.proj_out_logweights(hidden_states).reshape(
                bs, height, width, self.patch_size, self.patch_size, self.config.num_gaussians
            ).permute(0, 5, 1, 3, 2, 4).reshape(
                bs, self.config.num_gaussians, 1, height * self.patch_size, width * self.patch_size
            ).log_softmax(dim=1)
        out_logstds = self.proj_out_logstds(cond_emb.detach()).reshape(bs, 1, 1, 1, 1)

        out_dict = dict(
            means=out_means,
            logweights=out_logweights,
            logstds=out_logstds)
        return out_dict


@MODULES.register_module()
class GMDiTTransformer2DModelV2(_GMDiTTransformer2DModelV2):

    def __init__(
            self,
            *args,
            freeze=False,
            freeze_exclude=[],
            pretrained=None,
            torch_dtype='float32',
            freeze_exclude_fp32=True,
            checkpointing=True,
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

        self.freeze_exclude_fp32 = freeze_exclude_fp32
        if self.freeze_exclude_fp32:
            for attr in freeze_exclude:
                m = rgetattr(self, attr)
                assert isinstance(m, nn.Module)
                m.to(torch.float32)
                autocast_patch(m, enabled=False)

        if checkpointing:
            self.enable_gradient_checkpointing()

    def init_weights(self, pretrained=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)


class _SpectrumMLP(ModelMixin, ConfigMixin):

    @register_to_config
    def __init__(
            self,
            base_size=(4, 32, 32),  # (c, h, w)
            layers=[64, 8]):
        super().__init__()
        assert len(base_size) == 3
        mlp = []
        in_chn = 2
        for i, out_chn in enumerate(layers):
            mlp.append(nn.Linear(in_chn, out_chn))
            mlp.append(nn.SiLU())
            in_chn = out_chn
        mlp.append(nn.Linear(in_chn, base_size[0] * base_size[1] * base_size[2]))
        self.mlp = nn.Sequential(*mlp)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                xavier_init(m, distribution='uniform')
        constant_init(self.mlp[-1], val=0)

    def forward(self, gaussian_output):

        ori_dtype = gaussian_output['mean'].dtype
        spectral_mlp_dtype = next(self.parameters()).dtype

        output_stats = torch.stack(
            [gaussian_output['var'].mean(dim=(-3, -2, -1)),
             gaussian_output['mean'].var(dim=(-2, -1)).mean(dim=-1)],
            dim=-1).to(spectral_mlp_dtype)

        c, base_h, base_w = self.config.base_size
        h, w = gaussian_output['mean'].shape[-2:]
        batch_shape = output_stats.shape[:-1]
        bs = batch_shape.numel()

        output_stats = output_stats.reshape(bs, 2)
        spectrum = self.mlp(output_stats).reshape(bs, c, base_h, base_w)

        if h != base_h or w != base_w:
            assert h < base_h and w < base_w
            h1 = (h + 1) // 2
            h2 = h - h1
            w1 = (w + 1) // 2
            w2 = w - w1
            spectrum = torch.cat(
                [torch.cat([spectrum[..., :h1, :w1], spectrum[..., :h1, -w2:]], dim=-1),
                 torch.cat([spectrum[..., -h2:, :w1], spectrum[..., -h2:, -w2:]], dim=-1)], dim=-2)

        log_var = spectrum.flatten(-2).log_softmax(dim=-1) + math.log(h * w)
        log_var = log_var.reshape(*batch_shape, c, h, w)

        return log_var.to(ori_dtype)


@MODULES.register_module()
class SpectrumMLP(_SpectrumMLP):

    def __init__(
            self,
            *args,
            pretrained=None,
            torch_dtype='float32',
            **kwargs):

        super().__init__(*args, **kwargs)

        self.init_weights(pretrained)
        if torch_dtype is not None:
            self.to(getattr(torch, torch_dtype))

    def init_weights(self, pretrained=None):
        super().init_weights()
        if pretrained is not None:
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
