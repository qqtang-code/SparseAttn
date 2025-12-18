# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team

import torch

from typing import Any, Tuple
from torch import Tensor
from torch.nn import Module

import torch.distributed as dist

import math
from typing import Optional, Tuple, Union

from einops import rearrange, repeat

def rotate_half(x, interleaved=False):
    if not interleaved:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1, x2 = x[..., ::2], x[..., 1::2]
        return rearrange(torch.stack((-x2, x1), dim=-1), "... d two -> ... (d two)", two=2)

def apply_rotary_emb_torch(x, cos, sin, interleaved=False):
    """
    x: (batch_size, seqlen, nheads, headdim)
    cos, sin: (seqlen, rotary_dim / 2) or (batch_size, seqlen, rotary_dim / 2)
    """
    ro_dim = cos.shape[-1] * 2
    assert ro_dim <= x.shape[-1]
    cos = repeat(cos, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    sin = repeat(sin, "... d -> ... 1 (2 d)" if not interleaved else "... d -> ... 1 (d 2)")
    return torch.cat(
        [x[..., :ro_dim] * cos + rotate_half(x[..., :ro_dim], interleaved) * sin, x[..., ro_dim:]],
        dim=-1,
    ) 

# mode="max-autotune" 会尝试生成最快的 Triton 代码
# fullgraph=True 告诉编译器这里没有 Python 控制流，可以全图优化
# fast_apply_rotary = torch.compile(apply_rotary_emb_torch, mode="max-autotune") FIXME: 不能和 Gradient Checkpointing 混用
fast_apply_rotary = apply_rotary_emb_torch

class SeqAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any, input: Tensor, scatter_idx: int, gather_idx: int, group: Any
    ) -> Tensor:
        ctx.scatter_idx = scatter_idx
        ctx.gather_idx = gather_idx
        ctx.group = group

        world_size = dist.get_world_size(group)

        input_list = [
            t.contiguous() for t in torch.tensor_split(input, world_size, scatter_idx)
        ]
        output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
        # if(dist.get_rank() == 0):
            # 打印input_list中每个张量的shape
            # print("\n\n==============================================\n \
                # input_list 中各张量shape: ", [t.shape for t in input_list])
            # 打印output_list中每个张量的shape
            # print("output_list 中各张量shape: ", [t.shape for t in output_list])
            # （可选）打印列表第一个张量的shape（代表所有分片的shape，因为torch.tensor_split是均分）
            # print("input_list 第一个张量shape: ", input_list[0].shape)
            # print("output_list 第一个张量shape: ", output_list[0].shape)

            # print("scatter_idx:", scatter_idx)
            # print("gather_idx:", gather_idx)
        dist.all_to_all(output_list, input_list, group=group)

        return torch.cat(output_list, dim=gather_idx).contiguous()

    @staticmethod
    def backward(ctx: Any, *grad_output: Tensor) -> Tuple[Tensor, None, None, None]:
        return (
            SeqAllToAll.apply(*grad_output, ctx.gather_idx, ctx.scatter_idx, ctx.group),
            None,
            None,
            None,
        )


class DistributedAttention(torch.nn.Module):
    """Initialization.

    Arguments:
        local_attention (Module): local attention with q,k,v
        scatter_idx (int): scatter_idx for all2all comm
        gather_idx (int): gather_idx for all2all comm
    """

    def __init__(
        self,
        local_attention: Module,
    ) -> None:
        super(DistributedAttention, self).__init__()
        self.local_attn = local_attention

    def forward(
        self,
        query: Tensor,
        key_values: Tensor,
        *args,
        group: Any = None,
        scatter_idx: int = -2,
        gather_idx: int = 1,
        **kwargs,
    ) -> Tensor:
        """forward

        Arguments:
            query (Tensor): query input to the layer
            key (Tensor): key input to the layer
            value (Tensor): value input to the layer
            args: other args

        Returns:
            * output (Tensor): context output
        """
        # in shape : e.g.,  [s/p:h:]
        query_heads = SeqAllToAll.apply(query, scatter_idx, gather_idx, group)
        key_values_heads = SeqAllToAll.apply(key_values, scatter_idx, gather_idx, group)

        # out shape : e.g., [s:h/p:]
        
        output_heads = self.local_attn(query_heads, key_values_heads, *args, **kwargs)

        # out e.g., [s/p::h]
        return SeqAllToAll.apply(output_heads, gather_idx, scatter_idx, group)

"""PyTorch Qwen3 model."""

from typing import List, Optional, Tuple, Union, Any

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

import torch.distributed as dist

import os

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging, ModelOutput, LossKwargs
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.processing_utils import Unpack
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

from flash_attn import flash_attn_kvpacked_func, flash_attn_varlen_kvpacked_func
from flash_attn.bert_padding import unpad_input, pad_input
import math

try:
    from flash_attn.layers.rotary import apply_rotary_emb_func
except ImportError:
    raise ImportError(
        "Please install RoPE kernels: `pip install git+https://github.com/HazyResearch/flash-attention.git#subdirectory=csrc/rotary`"
    )

from block_sparse_attn import block_streaming_attn_func

from dataclasses import dataclass

from .distributed_attention import DistributedAttention
from .attention_mask import (
    deterministic_z_from_log_alpha,
    sample_z_from_log_alpha,
    cdf_stretched_concrete,
)
from sparseattn.src.Xattention import Xattention_prefill_dim3, Xattention_prefill_dim4

logger = logging.get_logger(__name__)


class PawQwen3Config(Qwen3Config):
    def __init__(self, *args, **kwargs):
        self.local_window_size = kwargs.pop("local_window_size", 1024)
        self.disable_linear_regularization_term = kwargs.pop(
            "disable_linear_regularization_term", False
        )
        self.suggested_sparsity = kwargs.pop("suggested_sparsity", None)

        # Streaming
        self.toggle_type = kwargs.pop("toggle_type", "streaming")
        self.sink_size = kwargs.pop("sink_size", 128)
        
        # retrieval_mode
        self.retrieval_mode = kwargs.pop("retrieval_mode", "full")
        
        # Head Router
        self.pooling_mode = kwargs.pop("pooling_mode", "first_token")
        
        self.use_task_emb_for_mask = kwargs.pop("use_task_emb_for_mask", False)

        # TriangleMix
        self.triangle_n_last = kwargs.pop("triangle_n_last", 128)
        
        # ada-sparsity
        self.enable_ada_sparsity = kwargs.pop("enable_ada_sparsity", False)

        # Layer-wise sparsity control
        self.enable_layerwise_sparsity = kwargs.pop("enable_layerwise_sparsity", False)

        self.layerwise_sparsity_schedule = kwargs.pop(
            "layerwise_sparsity_schedule", "high-low-high"
        )
        self.layerwise_sparsity_min_ratio = kwargs.pop(
            "layerwise_sparsity_min_ratio", 0.5
        )
        self.layerwise_sparsity_max_ratio = kwargs.pop(
            "layerwise_sparsity_max_ratio", 1.0
        )
        self.layerwise_sparsity_power = kwargs.pop("layerwise_sparsity_power", 1.0)
        self.layerwise_sparsity_weight = kwargs.pop("layerwise_sparsity_weight", 1.0)

        self.erank_analysis_path = kwargs.pop("erank_analysis_path", None)

        # 新增：top-k 注意力的超参（每个 query 仅保留前 k 个 key）
        self.topk_k = kwargs.pop("topk_k", 32)
        self.pooling_seq = kwargs.pop("pooling_seq", True)
        self.enable_lambda_task = kwargs.pop("enable_lambda_task", False)
        self.use_softmax = kwargs.pop("use_softmax", False)
        
        super().__init__(*args, **kwargs)


def get_mask(
    log_alpha, training=False, threshold_for_deterministic=None, apply_one=False
):
    if training:
        mask = sample_z_from_log_alpha(log_alpha)
    else:
        mask = deterministic_z_from_log_alpha(log_alpha, apply_one=apply_one)
        if threshold_for_deterministic is not None:
            mask = (mask > threshold_for_deterministic).to(mask.dtype)
    return mask


def generate_streaming_info_blocksparse_flash_attn(
    sink_block_num, local_block_num, n_query_heads, device
):
    streaming_info = torch.tensor(
        [sink_block_num, local_block_num] * n_query_heads,
        device=device,
        dtype=torch.int32,
    )
    return streaming_info


def streaming_attn_varlen_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    streaming_info_kwargs: dict,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    dropout_p: float = 0.0,
    causal: bool = True,
    return_attn_probs: bool = False,
    window_size: Tuple[int, int] = (0, 0),
) -> Optional[torch.Tensor]:
    # kv is of shape [total_seqlen, k_or_v, num_heads, head_dim]
    k, v = kv[:, 0, :, :], kv[:, 1, :, :]

    total_seqlen, query_heads, head_dim = q.size()
    key_value_heads = k.size(1)

    # Since all heads are streaming heads
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=q.device, dtype=torch.int32
    )

    streaming_info_kwargs["n_query_heads"] = query_heads
    streaming_info_kwargs["device"] = q.device
    streaming_info = generate_streaming_info_blocksparse_flash_attn(
        **streaming_info_kwargs
    )

    attn_output = block_streaming_attn_func(
        q,
        k,
        v,
        cu_seqlens,
        cu_seqlens,
        head_mask_type,
        streaming_info,
        max_seqlen,
        max_seqlen,
        p_dropout=dropout_p,
        is_causal=causal,
    )

    return attn_output


def streaming_attn_kvpacked_func(
    q: torch.Tensor,
    kv: torch.Tensor,
    streaming_info_kwargs: dict,
    dropout_p: float = 0.0,
    causal: bool = True,
    return_attn_probs: bool = False,
) -> Optional[torch.Tensor]:
    # kv is of shape [bsz, kv_seq_len, k_or_v, num_heads, head_dim]

    bsz, seqlen, query_heads, head_dim = q.size()
    k, v = kv[:, :, 0, :, :], kv[:, :, 1, :, :]

    key_value_heads = k.size(2)
    kv_seqlen = k.size(1)

    q_unpad = q.view(bsz * seqlen, query_heads, head_dim)
    k_unpad = k.view(bsz * kv_seqlen, key_value_heads, head_dim)
    v_unpad = v.view(bsz * kv_seqlen, key_value_heads, head_dim)

    cu_seqlens_q = torch.arange(
        0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q_unpad.device
    )
    cu_seqlens_kv = torch.arange(
        0,
        (bsz + 1) * kv_seqlen,
        step=kv_seqlen,
        dtype=torch.int32,
        device=k_unpad.device,
    )

    # Since all heads are streaming heads
    head_mask_type = torch.tensor(
        [-1] * query_heads, device=q.device, dtype=torch.int32
    )

    streaming_info_kwargs["n_query_heads"] = query_heads
    streaming_info_kwargs["device"] = q_unpad.device
    streaming_info = generate_streaming_info_blocksparse_flash_attn(
        **streaming_info_kwargs
    )

    attn_output = block_streaming_attn_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_kv,
        head_mask_type,
        streaming_info,
        seqlen,
        seqlen,
        p_dropout=dropout_p,
        is_causal=causal,
    )

    return attn_output.reshape(bsz, seqlen, query_heads, head_dim)


class Qwen3RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Qwen3RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class FlashRotaryEmbedding(torch.nn.Module):
    """
    The rotary position embeddings from RoFormer_ (Su et. al).
    A crucial insight from the method is that the query and keys are
    transformed by rotation matrices which depend on the relative positions.

    Other implementations are available in the Rotary Transformer repo_ and in
    GPT-NeoX_, GPT-NeoX was an inspiration

    .. _RoFormer: https://arxiv.org/abs/2104.09864
    .. _repo: https://github.com/ZhuiyiTechnology/roformer
    .. _GPT-NeoX: https://github.com/EleutherAI/gpt-neox

    If scale_base is not None, this implements XPos (Sun et al., https://arxiv.org/abs/2212.10554).
    A recommended value for scale_base is 512: https://github.com/HazyResearch/flash-attention/issues/96
    Reference: https://github.com/sunyt32/torchscale/blob/main/torchscale/component/xpos_relative_position.py
    """

    def __init__(
        self,
        dim: int,
        base=10000.0,
        interleaved=False,
        scale_base=None,
        scaling_factor=1.0,
        pos_idx_in_fp32=True,
        device=None,
    ):
        """
        interleaved: if True, rotate pairs of even and odd dimensions (GPT-J style) instead
            of 1st half and 2nd half (GPT-NeoX style).
        pos_idx_in_fp32: if True, the position indices [0.0, ..., seqlen - 1] are in fp32,
            otherwise they might be in lower precision.
            This option was added because previously (before 2023-07-02), when we construct
            the position indices, we use the dtype of self.inv_freq. In most cases this would
            be fp32, but if the model is trained in pure bf16 (not mixed precision), then
            self.inv_freq would be bf16, and the position indices are also in bf16.
            Because of the limited precision of bf16 (e.g. 1995.0 is rounded to 2000.0), the
            embeddings for some positions will coincide.
            To maintain compatibility with models previously trained in pure bf16,
            we add this option.
        scaling_factor: RotaryEmbedding extended with linear scaling.
        """
        super().__init__()
        self.dim = dim
        self.base = float(base)
        self.pos_idx_in_fp32 = pos_idx_in_fp32
        # Generate and save the inverse frequency buffer (non trainable)
        inv_freq = self._compute_inv_freq(device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.interleaved = interleaved
        self.scale_base = scale_base
        self.scaling_factor = scaling_factor
        scale = (
            (torch.arange(0, dim, 2, device=device, dtype=torch.float32) + 0.4 * dim)
            / (1.4 * dim)
            if scale_base is not None
            else None
        )
        self.register_buffer("scale", scale)

        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None
        self._cos_k_cached = None
        self._sin_k_cached = None

    def _compute_inv_freq(self, device=None):
        return 1 / (
            self.base
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=torch.float32)
                / self.dim
            )
        )

    def _update_cos_sin_cache(self, seqlen, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seqlen > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seqlen
            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seqlen, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seqlen, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq
            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            if self.scale is None:
                self._cos_cached = torch.cos(freqs).to(dtype)
                self._sin_cached = torch.sin(freqs).to(dtype)
            else:
                power = (
                    torch.arange(
                        seqlen, dtype=self.scale.dtype, device=self.scale.device
                    )
                    - seqlen // 2
                ) / self.scale_base
                scale = self.scale.to(device=power.device) ** power.unsqueeze(-1)
                # We want the multiplication by scale to happen in fp32
                self._cos_cached = (torch.cos(freqs) * scale).to(dtype)
                self._sin_cached = (torch.sin(freqs) * scale).to(dtype)
                self._cos_k_cached = (torch.cos(freqs) / scale).to(dtype)
                self._sin_k_cached = (torch.sin(freqs) / scale).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q: (batch, seqlen, nheads, headdim)
        k: (batch, seqlen, nheads, headdim)
        seqlen_offset: can be used in generation where the qkv being passed in is only the last
        token in the batch.
        """
        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
        else:
            cu_seqlens, max_seqlen = None, q.shape[1]
        self._update_cos_sin_cache(
            max_seqlen + seqlen_offset, device=q.device, dtype=q.dtype
        )

        if self.scale is None:
            return apply_rotary_emb_func(
                q,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            ), apply_rotary_emb_func(
                k,
                self._cos_cached[seqlen_offset:],
                self._sin_cached[seqlen_offset:],
                self.interleaved,
                True,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            assert False

# 不支持，弃用
class Qwen3RotaryEmbeddingOrigin(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        interleaved=False,
        config: Optional[PawQwen3Config] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.scaling_factor = scaling_factor
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = True

        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"

        self._seq_len_cached = 0

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _update_cos_sin_cache(self, seq_len, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seq_len

            if "dynamic" in self.rope_type:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.config, device, seq_len=seq_len, **self.rope_kwargs
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq

            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = (torch.cos(freqs) * self.attention_scaling).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.attention_scaling).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,  # Used in sequence parallelism where each device sees only a chunk of the full sequence
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
    ):
        if unpadded_lengths is not None:
            cu_seqlens, max_seqlen = unpadded_lengths
            if seqlen_offset > 0:
                raise ValueError("seqlen_offset is not supported with unpadded_lengths")
        else:
            cu_seqlens, max_seqlen = None, q.shape[1]

        self._update_cos_sin_cache(max_seqlen + seqlen_offset, q.device, q.dtype)

        rope_q = apply_rotary_emb_func(
            q,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        rope_k = apply_rotary_emb_func(
            k,
            self._cos_cached[seqlen_offset:],
            self._sin_cached[seqlen_offset:],
            self.interleaved,
            True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return rope_q, rope_k
    
    
    
class Qwen3RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim=None,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
        rope_type="default",
        interleaved=False,
        config: Optional[PawQwen3Config] = None,
    ):
        super().__init__()
        self.rope_kwargs = {}
        self.scaling_factor = scaling_factor
        self.interleaved = interleaved
        self.pos_idx_in_fp32 = True

        if config is None:
            self.rope_kwargs = {
                "rope_type": rope_type,
                "factor": scaling_factor,
                "dim": dim,
                "base": base,
                "max_position_embeddings": max_position_embeddings,
            }
            self.rope_type = rope_type
        else:
            # BC: "rope_type" was originally "type"
            if config.rope_scaling is not None:
                self.rope_type = config.rope_scaling.get(
                    "rope_type", config.rope_scaling.get("type")
                )
            else:
                self.rope_type = "default"

        self._seq_len_cached = 0

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(
            self.config, device, **self.rope_kwargs
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def _update_cos_sin_cache(self, seq_len, device=None, dtype=None):
        # Reset the tables if the sequence length has changed,
        # if we're on a new device (possibly due to tracing for instance),
        # or if we're switching from inference mode to training
        if (
            seq_len > self._seq_len_cached
            or self._cos_cached.device != device
            or self._cos_cached.dtype != dtype
            or (self.training and self._cos_cached.is_inference())
        ):
            self._seq_len_cached = seq_len

            if "dynamic" in self.rope_type:
                inv_freq, self.attention_scaling = self.rope_init_fn(
                    self.config, device, seq_len=seq_len, **self.rope_kwargs
                )
                self.register_buffer("inv_freq", inv_freq, persistent=False)

            # We want fp32 here, not self.inv_freq.dtype, since the model could be loaded in bf16
            # And the output of arange can be quite large, so bf16 would lose a lot of precision.
            # However, for compatibility reason, we add an option to use the dtype of self.inv_freq.
            if self.pos_idx_in_fp32:
                t = torch.arange(seq_len, device=device, dtype=torch.float32)
                t /= self.scaling_factor
                # We want fp32 here as well since inv_freq will be multiplied with t, and the output
                # will be large. Having it in bf16 will lose a lot of precision and cause the
                # cos & sin output to change significantly.
                # We want to recompute self.inv_freq if it was not loaded in fp32
                if self.inv_freq.dtype != torch.float32:
                    inv_freq = self.inv_freq.to(torch.float32)
                else:
                    inv_freq = self.inv_freq
            else:
                t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
                t /= self.scaling_factor
                inv_freq = self.inv_freq

            # Don't do einsum, it converts fp32 to fp16 under AMP
            # freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            freqs = torch.outer(t, inv_freq)
            self._cos_cached = (torch.cos(freqs) * self.attention_scaling).to(dtype)
            self._sin_cached = (torch.sin(freqs) * self.attention_scaling).to(dtype)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        seqlen_offset: int = 0,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        position_ids: Optional[torch.LongTensor] = None, # [CRITICAL] 必须传入
    ):
        """
        使用 apply_rotary_emb_torch 绕过 Triton Kernel 的 Shape 检查
        完美支持 [S, H, D] 格式的 Context Parallel
        """
        if position_ids is not None:
            max_seqlen_in_batch = position_ids.max().item() + 1
        elif unpadded_lengths is not None:
            _, max_seqlen = unpadded_lengths
            max_seqlen_in_batch = max_seqlen + seqlen_offset
        else:
            max_seqlen_in_batch = q.shape[-2] + seqlen_offset

        self._update_cos_sin_cache(max_seqlen_in_batch, q.device, q.dtype)

        if position_ids is not None:
            # [Case A] Context Parallel / Varlen (推荐)
            # 根据全局 position_ids 取出对应的 cos/sin
            # position_ids: [S_total]
            # cos/sin: [S_total, Dim/2]
            cos = self._cos_cached[position_ids]
            sin = self._sin_cached[position_ids]
            
        elif unpadded_lengths is not None:
            # [Case B] Varlen without IDs (Standard DP)
            # 如果没有 position_ids，我们无法处理 CP 切分的情况
            # 但如果是纯 DP，我们可以尝试让 apply_rotary_emb_torch 广播
            # 不过为了安全，建议强制要求 position_ids
            raise ValueError("Context Parallel / Varlen requires explicit `position_ids`.")
            
        else:
            # [Case C] Dense [B, S, H, D]
            # 切片取当前窗口的 cos/sin
            seq_len = q.shape[-2]
            cos = self._cos_cached[seqlen_offset : seqlen_offset + seq_len]
            sin = self._sin_cached[seqlen_offset : seqlen_offset + seq_len]
        
        q_embed = fast_apply_rotary(q, cos, sin, interleaved=self.interleaved)
        k_embed = fast_apply_rotary(k, cos, sin, interleaved=self.interleaved)
        
        return q_embed, k_embed


class Qwen3MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


# def rotate_half(x):
#     """Rotates half the hidden dims of the input."""
#     x1 = x[..., : x.shape[-1] // 2]
#     x2 = x[..., x.shape[-1] // 2 :]
#     return torch.cat((-x2, x1), dim=-1)


# def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
#     """Applies Rotary Position Embedding to the query and key tensors.

#     Args:
#         q (`torch.Tensor`): The query tensor.
#         k (`torch.Tensor`): The key tensor.
#         cos (`torch.Tensor`): The cosine part of the rotary embedding.
#         sin (`torch.Tensor`): The sine part of the rotary embedding.
#         position_ids (`torch.Tensor`, *optional*):
#             Deprecated and unused.
#         unsqueeze_dim (`int`, *optional*, defaults to 1):
#             The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
#             sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
#             that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
#             k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
#             cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
#             the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
#     Returns:
#         `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
#     """
#     cos = cos.unsqueeze(unsqueeze_dim)
#     sin = sin.unsqueeze(unsqueeze_dim)
#     q_embed = (q * cos) + (rotate_half(q) * sin)
#     k_embed = (k * cos) + (rotate_half(k) * sin)
#     return q_embed, k_embed


@torch.jit.script
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class AttentionRouter(nn.Module):
    def __init__(self, input_dim, num_key_value_heads, d_feature=128,
                 use_task_emb=False, temp=0.2, hard=False, 
                 router_type='mlp', use_gumbel=True, learnable_temp=False,
                 dropout=0.1, use_softmax=True, pooling_mode='ctx_q'):
        super().__init__()
        self.num_kv = num_key_value_heads
        self.use_task_emb = use_task_emb
        self.router_type = router_type
        self.use_gumbel = use_gumbel
        self.learnable_temp = learnable_temp
        self.pooling_mode = pooling_mode
        self.use_softmax = use_softmax

        self.cls_feat_extractor = nn.Sequential( 
            nn.Linear(d_feature, 2 * d_feature),
            nn.SiLU(),
            nn.Linear(2 * d_feature, d_feature),
        )
        
        if self.use_softmax:
            logger.info("using softmax for attention router")
            self.cls_router_head_agnostic = nn.Sequential( 
                nn.Linear(d_feature, 2 * d_feature),
                nn.SiLU(),
                nn.Linear(2 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 2),
            )
        else:
            logger.info("use sigmoid function for attention router")
            self.cls_router_head_agnostic = nn.Sequential( 
                nn.Linear(d_feature, 2 * d_feature),
                nn.SiLU(),
                nn.Linear(2 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 1),
                nn.LayerNorm([self.num_kv, 1], elementwise_affine=False)
            )
        
        if self.use_task_emb:
            self.task_embedding = nn.Embedding(4, d_feature)

        # ---- learnable temperature ----
        if learnable_temp:
            self.log_temp = nn.Parameter(torch.log(torch.tensor(temp)))
        else:
            self.register_buffer("log_temp", torch.log(torch.tensor(temp)))
            self.tau = torch.exp(self.log_temp).clamp(0.3, 1.0)
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.cls_router_head_agnostic[0].weight, a=math.sqrt(5))
        nn.init.zeros_(self.cls_router_head_agnostic[0].bias)
        
        nn.init.kaiming_uniform_(self.cls_router_head_agnostic[2].weight, a=math.sqrt(5))
        nn.init.zeros_(self.cls_router_head_agnostic[2].bias)

        nn.init.zeros_(self.cls_router_head_agnostic[4].weight)
        nn.init.constant_(self.cls_router_head_agnostic[4].bias, 1.0)

        
    def forward(
        self, 
        x, 
        cu_seq_len=None,
        range_ids: torch.Tensor = None, 
        task_ids: Optional[torch.Tensor] = None,
        current_tau: Optional[torch.Tensor] = None,
    ):
        """
        x: [cu_seq_len, H, D]
        cu_seq_len: [0, seq_len_1, seq_len_2 + seq_len_1, ...]
        range_ids: [B, 6]
        task_ids: [B]
        
        return:
            {
              'decisions': [B, H],
              'hard_decisions': [B, H, 2],
              'sparse_mask': [B, H],
              'logits': [B, H, 1]
            }
        """
        bsz = (cu_seq_len.shape[0] - 1) if cu_seq_len is not None else 1
        
        # 目前所有支持的pooling 方法
        if self.pooling_mode == 'first_token':
            if cu_seq_len is not None:
                pooled_latent = self._segment_pooling(
                    x, range_ids, ['first_token'], cu_seq_len)  # [B, H, D]
            else:
                pooled_latent = self._segment_pooling_single_batch(
                    x, range_ids, ['first_token'])
        elif self.pooling_mode == 'q':
            if cu_seq_len is not None:
                pooled_latent = self._segment_pooling(
                    x, range_ids, ['q'], cu_seq_len)  # [B, H, D]
            else:
                pooled_latent = self._segment_pooling_single_batch(
                    x, range_ids, ['q'])
        elif self.pooling_mode == 'ctx_q':
            if cu_seq_len is not None:
                pooled_latent = self._segment_pooling(
                    x, range_ids, ['ctx_q'], cu_seq_len)  # [B, H, D]
            else:
                pooled_latent = self._segment_pooling_single_batch(
                    x, range_ids, ['ctx_q'])

        else:
            raise ValueError(f"Unknown pooling_mode: {self.pooling_mode}")
        
        if self.use_task_emb:
            if self.training:
                task_emb = self.task_embedding(task_ids) # [B, D]
                task_emb_expanded = task_emb.unsqueeze(1) 
                pooled_latent = pooled_latent + task_emb_expanded
            else:
                pooled_latent = pooled_latent
                                
        pooled_hidden_states = self.cls_feat_extractor(pooled_latent)

        binary_logits = self.cls_router_head_agnostic(pooled_hidden_states)
        
        if self.learnable_temp:
            tau = torch.exp(self.log_temp).clamp(0.3, 1.0)
        else:
            tau = current_tau if current_tau is not None else self.tau

        # --- Gumbel or Softmax routing ---
        if self.training:
            u = torch.rand_like(binary_logits)
            eps = 1e-8
            g = -torch.log(-torch.log(u + eps) + eps)
            
            if not self.use_softmax:
                z_soft = torch.sigmoid((binary_logits + g) / tau)
                z_hard = (z_soft > 0.5).float()
                z = z_hard + (z_soft - z_soft.detach())  # [B, H, 1]
                entropy = -(z_soft * torch.log(z_soft + eps) + (1 - z_soft) * torch.log(1 - z_soft + eps))
            else:
                z_soft = F.softmax((binary_logits + g) / tau, dim=-1)
                z_hard = torch.zeros_like(z_soft).scatter_(-1, z_soft.argmax(-1, keepdim=True), 1.0)
                z = z_hard + (z_soft - z_soft.detach())  # [B, H, 2]
                z = z[..., 1]  # [B, H]
                z_soft = z_soft[..., 1]
                z_soft = z_soft.unsqueeze(-1)
                z = z.unsqueeze(-1)
                entropy = -(z_soft * torch.log(z_soft + eps)).sum(dim=-1).mean() 
        else:
            # 推理阶段：直接根据 Logit 确定 (相当于 tau -> 0)
            # 或者也可以用 sigmoid(logit/tau) > 0.5，但在 deterministic 模式下 logit > 0 即可
            # z_soft = torch.sigmoid(binary_logits / tau) 
            if not self.use_softmax:
                z_soft = torch.sigmoid(binary_logits / tau)
                z_hard = (z_soft > 0.5).float()
                z = z_hard
            else:
                z_soft = F.softmax(binary_logits / tau, dim=-1)
                z_hard = z_soft.argmax(-1)
                z = z_hard
        
        return {
            'pooled_hidden_states': pooled_hidden_states, # [B, H, D]
            'decisions': z_soft,
            'hard_decisions': z_hard,
            'sparse_mask': z, # [B, H], 这是一个 STE Tensor
            'logits': binary_logits,
            'entropy': entropy
        }
        
    def _segment_pooling_single_batch(self, pooled_input: torch.Tensor, range_ids: torch.Tensor, segments: list) -> torch.Tensor:
        B, S, H, D = pooled_input.shape
        pooled_features_list = []
        
        POOL_MAP = {'first_token': (0, 1),'ctx': (2, 3), 'q': (4, 5), 'a': (6, 7), 'ctx_q': (2, 5)} 
        for i in range(B):
            sample_features = []

            for seg in segments:
                start_idx, end_idx = POOL_MAP[seg]
                start, end = range_ids[i, start_idx:end_idx + 1].tolist()[0], range_ids[i, start_idx:end_idx + 1].tolist()[-1]
                if end >= start:
                    # seg_slice = pooled_input[i, start : end + 1, :, :]
                    start_slice = pooled_input[i, start : start + 100, :, :]
                    end_slice = pooled_input[i, end - 100 : end + 1, :, :]
                    combined_slice = torch.cat((start_slice, end_slice), dim=0)
                    seg_pooled = combined_slice.mean(dim=0)  # [H, D]
                else:
                    seg_pooled = torch.zeros(H, D, device=pooled_input.device)
                
                sample_features.append(seg_pooled)

            if sample_features:
                combined_feature = torch.stack(sample_features, dim=0).mean(dim=0) # [H, D]
            else:
                combined_feature = torch.zeros(H, D, device=pooled_input.device)
                
            pooled_features_list.append(combined_feature)

        return torch.stack(pooled_features_list, dim=0) # [B, H, D]
        
        
    def _segment_pooling(
        self, 
        x: torch.Tensor, 
        range_ids: torch.Tensor, 
        segments: list[str],
        cu_seq_len: torch.Tensor,
    ) -> torch.Tensor:
        """_summary_

        Args:
            x (torch.Tensor): [cu_seqlen, H, D]
            range_ids (torch.Tensor): _description_
            segments (list[str]): _description_
            cu_seq_len (torch.Tensor): _description_

        Returns:
            torch.Tensor: _description_
        """
        POOL_MAP = {'first_token': (0, 1),'ctx': (2, 3), 'q': (4, 5), 'a': (6, 7), 'ctx_q': (2, 5)} 

        B = cu_seq_len.shape[0] - 1
        H, D = x.shape[1:]
        pooled_features_list = []

        for i in range(B):
            sample_features = []
            x_s, x_e = cu_seq_len[i], cu_seq_len[i + 1]
            for seg in segments:
                start_idx, end_idx = POOL_MAP[seg]
                start, end = range_ids[i, start_idx:end_idx + 1].tolist()[0], range_ids[i, start_idx:end_idx + 1].tolist()[-1]
                if end >= start:
                    prefix_seg_slice = x[x_s + start: x_s + start + 100,  : , :]
                    suffix_seg_slice = x[x_s + end - 99: x_s + end + 1,  : , :]
                    combined_slice = torch.cat((prefix_seg_slice, suffix_seg_slice), dim=0)
                    seg_pooled = combined_slice.mean(dim=0)  # [H, D]
                else:
                    seg_pooled = torch.zeros(H, D, device=x.device)

                sample_features.append(seg_pooled)

            if sample_features:
                combined_feature = torch.stack(sample_features, dim=0).mean(dim=0) # [H, D]
            else:
                combined_feature = torch.zeros(H, D, device=x.device)

            pooled_features_list.append(combined_feature)

        return torch.stack(pooled_features_list, dim=0) # [B, H, D]

class Qwen3Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        config: PawQwen3Config,
        context_window_toggle: Optional[int] = 1024,
    ):
        """
        @context_window_toggle: if not None, the attention will be limited to a context window specified by this value
        """
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.num_key_value_heads = getattr(
            config, "num_key_value_heads", self.num_heads
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = config.max_position_embeddings
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.register_buffer(
            "norm_factor",
            torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(
                torch.get_default_dtype()
            ),
            persistent=False,
        )

        self.q_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(
            self.head_dim, eps=config.rms_norm_eps
        )  # thus post q_norm does not need reshape

        self.rotary_emb = Qwen3RotaryEmbedding(config=self.config)
        self.distributed_attn_func = DistributedAttention(self.interpolated_attention)

        self._dtype = self.q_proj.weight.dtype
        self.attn_mask_log_alphas = nn.Parameter(
            torch.empty(self.num_key_value_heads, dtype=self._dtype)
        )
        self.attn_mask_log_alphas.data.normal_(
            mean=4.5, std=0.01
        )  # sigmoid(4.5) ≈ 0.989
        self.threshold_for_deterministic = None

        self.mask_allocator = AttentionRouter(
            input_dim=self.hidden_size,
            num_key_value_heads=self.num_key_value_heads,
            # head_dim = self.head_dim,
            d_feature=self.head_dim,
            use_task_emb=getattr(config, "use_task_emb_for_mask", False),
            temp=getattr(config, "mask_temp", 3/2),
            hard=getattr(config, "mask_hard_sample", False),
            pooling_mode=getattr(config, "pooling_mode", "first_token"),
            use_softmax=getattr(config, "use_softmax", False)
        )

        self.context_window_toggle = context_window_toggle

        self.toggle_type = config.toggle_type
        self.sink_blocks = (config.sink_size + 127) // 128
        self.local_blocks = (config.local_window_size + 127) // 128
        
        self.retrieval_mode = config.retrieval_mode

        if self.retrieval_mode == "xattn":
            from sparseattn.utils.ops.xattention_fa import xattn_flash_attn_func
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            # self.head_indices = self.num_heads // self.num_key_value_heads
            self.head_indices = self.num_heads
            self.xattn_flash_attn_func = xattn_flash_attn_func
            self.granularity = int(getattr(config, "block_size", 64))
            self.xattn_params = {
                "stride": 16,
                "norm": 1,
                "softmax": True,
                "threshold": 0.9,
                "chunk_size": 16384,
                "select_mode": "inverse",
                "use_triton": True,
                "causal": True,
                "kdb": 1,
                "keep_sink": True,
                "keep_recent": True,
            }

        if self.toggle_type == "streaming":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            self.context_window_toggle = (self.sink_blocks + self.local_blocks) * 128
        elif self.toggle_type == "local":
            pass
        elif self.toggle_type == "triangle":
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            self.context_window_toggle = (self.sink_blocks + self.local_blocks) * 128
            self.triangle_n_last = config.triangle_n_last
        elif self.toggle_type == "topk":
            self.topk_k = int(getattr(config, "topk_k", 2048))
            self.topk_q_chunk = int(os.environ.get("TOPK_Q_CHUNK", 128))
            self.topk_k_chunk = int(os.environ.get("TOPK_K_CHUNK", 4096))
        elif self.toggle_type == "xattn" or self.retrieval_mode == "xattn":
            from sparseattn.utils.ops.xattention_fa import xattn_flash_attn_func
            self.streaming_info_kwargs = {
                "sink_block_num": self.sink_blocks,
                "local_block_num": self.local_blocks,
            }
            # self.head_indices = self.num_heads // self.num_key_value_heads
            self.head_indices = self.num_heads
            self.xattn_flash_attn_func = xattn_flash_attn_func
            self.granularity = int(getattr(config, "block_size", 64))
            self.xattn_params = {
                "stride": 16,
                "norm": 1,
                "softmax": True,
                "threshold": 0.9,
                "chunk_size": 16384,
                "select_mode": "inverse",
                "use_triton": True,
                "causal": True,
                "kdb": 1,
                "keep_sink": True,
                "keep_recent": True,
            }
        elif self.toggle_type == "none":
            pass
        else:
            raise ValueError(f"Unknown toggle type: {self.toggle_type}")

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.threshold_for_deterministic = threshold_for_deterministic

    @torch.no_grad()
    def get_masks(self):
        z = get_mask(
            self.attn_mask_log_alphas,
            training=self.training,
            threshold_for_deterministic=self.threshold_for_deterministic,
        )
        return z

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.attn_mask_log_alphas.data.normal_(mean=value, std=0.01)

    @torch.no_grad()
    def fill_masks_with_value(self, value):
        if (
            isinstance(value, float)
            or isinstance(value, int)
            or (isinstance(value, torch.Tensor) and value.numel() == 1)
        ):
            self.attn_mask_log_alphas.data.fill_(value)
        else:
            if isinstance(value, list):
                value = torch.tensor(
                    value, dtype=self._dtype, device=self.attn_mask_log_alphas.device
                )
            value = value.reshape(-1)
            assert value.shape[0] == self.attn_mask_log_alphas.numel(), (
                "Value shape does not match mask shape"
            )
            self.attn_mask_log_alphas.data.copy_(value)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def interpolated_attention(self, q, kv, k, v, unpadded_lengths, z):
        if self.retrieval_mode == "full":
            if unpadded_lengths is not None:
                # varlen, ignore padding tokens, efficient for large batch with many paddings
                cu_seqlens, max_seqlen = unpadded_lengths
                attn_output = flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    dropout_p=0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=True,
                    return_attn_probs=False,
                )
            else:
                attn_output = flash_attn_kvpacked_func(
                    q,
                    kv,
                    dropout_p=0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=True,
                    return_attn_probs=False,
                )
        elif self.retrieval_mode == "xattn":
            is_vlen_input = (q.dim() == 3) and (unpadded_lengths is not None)

            if is_vlen_input:
                k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                v = v.repeat_interleave(self.num_key_value_groups, dim=1)
                q, k, v = q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous() 
            else:
                k = k.repeat_interleave(self.num_key_value_groups, dim=2)
                v = v.repeat_interleave(self.num_key_value_groups, dim=2)
                q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous() 
                
            stride = self.xattn_params["stride"]
            threshold = self.xattn_params["threshold"]
            norm = self.xattn_params["norm"]

            if unpadded_lengths is not None:
                cu_seqlens, max_seqlen = unpadded_lengths
                attn_output = Xattention_prefill_dim3(
                    q,
                    k,
                    v,
                    stride,
                    cu_seqlens,
                    norm,
                    threshold,
                    use_triton=True,
                )

            else:
                bsz,_,seqlen,_ = q.size()
                if not torch.is_tensor(seqlen):
                    seqlen = torch.tensor(seqlen, dtype=torch.int32, device=q.device)
                max_seqlen = torch.max(seqlen).item()

                cu_seqlens = torch.arange(
                    0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device
                )
                unpadded_lengths_xattn = (cu_seqlens, max_seqlen)

                cu_seqlens, max_seqlen = unpadded_lengths_xattn
                attn_output = Xattention_prefill_dim4(
                    q,
                    k,
                    v,
                    stride,
                    cu_seqlens,
                    norm,
                    threshold,
                    use_triton=True,
                ).transpose(1, 2)  # B, T, H, D
            if is_vlen_input:
                q = q.transpose(0, 1).contiguous()
            else:
                q = q.transpose(1, 2).contiguous()
        else:
            raise ValueError(f"Unknown retrieval mode: {self.retrieval_mode}")

        if self.toggle_type == "streaming" or self.toggle_type == "triangle":
            # breakpoint()
            if unpadded_lengths is not None:
                cu_seqlens, max_seqlen = unpadded_lengths
                cw_attn_output = streaming_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    self.streaming_info_kwargs,
                    cu_seqlens,
                    max_seqlen,
                    dropout_p=0.0,
                    causal=True,
                    return_attn_probs=False,
                )
            else:
                cw_attn_output = streaming_attn_kvpacked_func(
                    q,
                    kv,
                    self.streaming_info_kwargs,
                    dropout_p=0.0,
                    causal=True,
                    return_attn_probs=False,
                )
            if self.toggle_type == "triangle":
                if unpadded_lengths is not None:
                    cu_seqlens, _ = unpadded_lengths
                    total = q.size(0)
                    mask = torch.zeros(total, dtype=torch.bool, device=q.device)
                    B = cu_seqlens.numel() - 1
                    n_last = self.triangle_n_last
                    for b in range(B):
                        start = int(cu_seqlens[b].item())
                        end = int(cu_seqlens[b + 1].item())
                        seg_len = end - start
                        take = min(n_last, seg_len)
                        if take > 0:
                            mask[end - take : end] = True
                    cw_attn_output[mask] = attn_output[mask]
                else:
                    seq_len = q.size(1)
                    take = min(getattr(self, "triangle_n_last", 0), seq_len)
                    if take > 0:
                        cw_attn_output[:, -take:] = attn_output[:, -take:]

        elif self.toggle_type == "local":
            if unpadded_lengths is not None:
                # varlen, ignore padding tokens, efficient for large batch with many paddings
                cu_seqlens, max_seqlen = unpadded_lengths

                cw_attn_output = flash_attn_varlen_kvpacked_func(
                    q,
                    kv,
                    cu_seqlens,
                    cu_seqlens,
                    max_seqlen,
                    max_seqlen,
                    dropout_p=0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=True,
                    return_attn_probs=False,
                    window_size=(self.context_window_toggle - 1, 0),
                )
            else:
                cw_attn_output = flash_attn_kvpacked_func(
                    q,
                    kv,
                    dropout_p=0.0,
                    softmax_scale=1.0 / self.norm_factor,
                    causal=True,
                    return_attn_probs=False,
                    window_size=(self.context_window_toggle - 1, 0),
                )
        elif self.toggle_type == "xattn":  

            if not self.training :
                _, seq_len, _, _ = q.size()       
            if self.training or seq_len != 1:

                is_vlen_input = (q.dim() == 3) and (unpadded_lengths is not None)

                if is_vlen_input:
                    k = k.repeat_interleave(self.num_key_value_groups, dim=1)
                    v = v.repeat_interleave(self.num_key_value_groups, dim=1)
                    q, k, v = q.transpose(0, 1).contiguous(), k.transpose(0, 1).contiguous(), v.transpose(0, 1).contiguous() 
                else:
                    k = k.repeat_interleave(self.num_key_value_groups, dim=2)
                    v = v.repeat_interleave(self.num_key_value_groups, dim=2)
                    q, k, v = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous() 
                    
                stride = self.xattn_params["stride"]
                threshold = self.xattn_params["threshold"]
                norm = self.xattn_params["norm"]

                if unpadded_lengths is not None:
                    cu_seqlens, max_seqlen = unpadded_lengths
                    cw_attn_output = Xattention_prefill_dim3(
                        q,
                        k,
                        v,
                        stride,
                        cu_seqlens,
                        norm,
                        threshold,
                        use_triton=True,
                    )

                else:
                    bsz,_,seqlen,_ = q.size()
                    if not torch.is_tensor(seqlen):
                        seqlen = torch.tensor(seqlen, dtype=torch.int32, device=q.device)
                    max_seqlen = torch.max(seqlen).item()

                    cu_seqlens = torch.arange(
                        0, (bsz + 1) * seqlen, step=seqlen, dtype=torch.int32, device=q.device
                    )
                    unpadded_lengths = (cu_seqlens, max_seqlen)

                    cu_seqlens, max_seqlen = unpadded_lengths
                    cw_attn_output = Xattention_prefill_dim4(
                        q,
                        k,
                        v,
                        stride,
                        cu_seqlens,
                        norm,
                        threshold,
                        use_triton=True,
                    ).transpose(1, 2)  # B, T, H, D
            else:
                if unpadded_lengths is not None:
                    # varlen, ignore padding tokens, efficient for large batch with many paddings
                    cu_seqlens, max_seqlen = unpadded_lengths

                    cw_attn_output, _, attn_probs = flash_attn_varlen_kvpacked_func(
                        q,
                        kv,
                        cu_seqlens,
                        cu_seqlens,
                        max_seqlen,
                        max_seqlen,
                        dropout_p=0.0,
                        softmax_scale=1.0 / self.norm_factor,
                        causal=True,
                        return_attn_probs=True,
                    )
                else:
                    cw_attn_output, _, attn_probs = flash_attn_kvpacked_func(
                        q,
                        kv,
                        dropout_p=0.0,
                        softmax_scale=1.0 / self.norm_factor,
                        causal=True,
                        return_attn_probs=True,
                    )
        elif self.toggle_type == "none":
            cw_attn_output = torch.zeros_like(attn_output)
        else:
            raise ValueError(f"Unknown toggle type: {self.toggle_type}")

        if unpadded_lengths is not None:
            effective_attn_output = []
            cu_seqlens, max_seqlen = unpadded_lengths
            bsz = len(cu_seqlens) - 1
            for i in range(bsz):
                effective_attn_output.append(
                    attn_output[cu_seqlens[i]:cu_seqlens[i + 1], :, :] * z[i, None, ...] + cw_attn_output[cu_seqlens[i]:cu_seqlens[i + 1], :, :] * (1 - z)[i, None, ...]
                )
            effective_attn_output = torch.cat(effective_attn_output, dim=0)
        else:        
            effective_attn_output = attn_output * z[:,None,...] + cw_attn_output * (
                1 - z
            )[:,None,...]

        return effective_attn_output
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        current_tau: Optional[torch.Tensor] = None,
        position_embeddings: Optional[
            Tuple[torch.Tensor, torch.Tensor]
        ] = None,  # will become mandatory in v4.46
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        input_shape = hidden_states.shape[:-1] # [S_local, ]
        hidden_shape = (*input_shape, -1, self.head_dim) # [S_local, nhead, head_dim]
        q = self.q_norm(self.q_proj(hidden_states).view(hidden_shape))
        k = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        v = self.v_proj(hidden_states).view(hidden_shape)
 
        has_layer_past = past_key_value is not None
        
        if position_ids is None:
            
            past_len = past_key_value[1] if has_layer_past else 0
            seqlen_offset = past_len
        else:
            seqlen_offset = 0 # 有 position_ids 则 offset 无意义
            
        q, k = self.rotary_emb(
            q, 
            k, 
            seqlen_offset=seqlen_offset, 
            unpadded_lengths=unpadded_lengths,
            position_ids=position_ids 
        )
        
        kv = torch.stack([k, v], -3)
        if self.num_key_value_groups > 1:
            kv = kv.repeat_interleave(self.num_key_value_groups, dim=-2)

        # Cache QKV values
        if has_layer_past:
            new_len = past_len + q.size(1)
            if new_len > past_kv.size(1):
                past_kv = torch.cat(
                    [
                        past_kv,
                        torch.empty(
                            hidden_states.size(0),
                            256,
                            2,
                            kv.size(3),
                            kv.size(4),
                            dtype=kv.dtype,
                            device=kv.device,
                        ),
                    ],
                    1,
                )
            past_kv[:, past_len:new_len] = kv
            kv = past_kv[:, :new_len]
        else:
            past_kv = kv
        past_key_value = (past_kv, past_len + q.size(1)) if use_cache else None

        # Context Parallel Transform (Ulysses All2All)
        # 当前形状: [S_local, H_total, D]
        # 目标形状: [S_global, H_local, D]
        is_cp_enabled = (
            seq_parallel_group is not None
            and dist.is_initialized()
            and dist.get_world_size(seq_parallel_group) > 1
        )

        if is_cp_enabled:
            # Scatter dim 1 (Heads), Gather dim 0 (Seq)
            # 因为输入是 3D: [Seq, Head, Dim]
            if dist.get_rank() == 0:
                # print("------------- q SeqAllToAll ---------------")
                q = SeqAllToAll.apply(q, 1, 0, seq_parallel_group)
                # print("------------- k SeqAllToAll ---------------")
                k = SeqAllToAll.apply(k, 1, 0, seq_parallel_group)
                # print("------------- v SeqAllToAll ---------------")
                v = SeqAllToAll.apply(v, 1, 0, seq_parallel_group)
            else:
                q = SeqAllToAll.apply(q, 1, 0, seq_parallel_group)
                k = SeqAllToAll.apply(k, 1, 0, seq_parallel_group)
                v = SeqAllToAll.apply(v, 1, 0, seq_parallel_group)
            
            # 此时 q, k, v 变成了 [S_global, H_local, D]
            # unpadded_lengths (cu_seqlens) 是全局的，现在正好匹配 S_global
            
        # Attention Router
        # Router 需要全局序列信息来做 Pooling (first_token / ctx)
        # k: [S_global, H_local, D] -> Router -> z: [B, H_local]
        
        if not self.config.enable_ada_sparsity: # 暂时不用
            # 静态 Mask 逻辑
            z_kv = get_mask(
                self.attn_mask_log_alphas,
                training=self.training,
                threshold_for_deterministic=self.threshold_for_deterministic,
            )
            # 如果开启 CP，需要手动切分静态 mask 给当前 rank 的 heads
            if is_cp_enabled:
                rank = dist.get_rank(seq_parallel_group)
                world_size = dist.get_world_size(seq_parallel_group)
                # z_kv 形状是 [H_total_kv]，在 dim 0 切分
                z_kv = torch.tensor_split(z_kv, world_size, dim=0)[rank]
            
            # Expand to Group & Flatten
            z = z_kv.unsqueeze(-1).expand(-1, self.num_key_value_groups).reshape(-1) #[H_local_q]
        else:
            # 动态 Router
            # 注意：range_ids, task_ids 通常是 [B, ...] 的，B 与 Global Seq 是一致的
            if unpadded_lengths is not None:
                # unpadded_lengths[0] 是 cu_seqlens
                res = self.mask_allocator(k, unpadded_lengths[0], range_ids, task_ids, current_tau)
            else:
                # 如果没有 varlen info，Router 可能无法工作，或者退化为 single batch
                res = self.mask_allocator(k, unpadded_lengths[0], range_ids, task_ids, current_tau)
            
            # z_kv_batch: [B, H_local_kv, ]
            # entropy:  标量
            # pooled_hidden_states: [B, H_local_kv, D]
            # z_constrast: [B, H_local_kv, 1]
            z_kv_batch, entropy, pooled_hidden_states = res['sparse_mask'], res['entropy'], res['pooled_hidden_states']
            z_constrast = res['decisions']
            

            # GQA 适配: [B, H_local_kv, 1] -> [B, H_local, 1]
            # 注意：这里的 self.num_key_value_heads 在初始化时是 Total 的
            # 我们需要判断 z_kv_batch 是否已经是 local 大小
            local_kv_heads = self.num_key_value_heads // (dist.get_world_size(seq_parallel_group) if is_cp_enabled else 1)
            
            if z_kv_batch.shape[1] == local_kv_heads:
                # Expand GQA groups
                z_kv_batch = z_kv_batch.repeat_interleave(self.num_key_value_groups, dim=1)
            # breakpoint()
        # Attention Computation (Flash Attn)
        # 输入: [S_global, H_local, D]
        # Mask: z_kv_batch [B, H_local, 1] -> interpolated_attention 内部会广播
        
        kv_packed = torch.stack([k, v], dim=1)
        # breakpoint()
        attn_output = self.interpolated_attention(q, kv_packed, k, v, unpadded_lengths, z_kv_batch)
        # attn_output: [S_global, H_local, D]

        # Context Parallel Reverse Transform
        if is_cp_enabled:
            # 这里的 q 经历了 SeqAllToAll，已经是 S_global 长度 
            # 而 attn_output 长度可能小于 S_global (被 FlashAttn Unpad 了)
            expected_global_len = q.shape[0]
            actual_len = attn_output.shape[0]
            
            if actual_len < expected_global_len:
                pad_len = expected_global_len - actual_len
                # 在 Dim 0 (Seq) 的末尾补 pad_len 个 0
                attn_output = torch.nn.functional.pad(attn_output, (0, 0, 0, 0, 0, pad_len))
            # Scatter dim 0 (Seq), Gather dim 1 (Heads)
            # 变回: [S_local, H_total, D]
            attn_output = SeqAllToAll.apply(attn_output, 0, 1, seq_parallel_group)

        # Output Projection
        # [S_local, H_total, D] -> [S_local, Hidden]
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output.to(self.o_proj.weight.dtype))

        attn_weights = None
        
        # z_kv_batch: [B, H_local, 1] (Dynamic) -> [B, ]
        # entropy: [B, H_local_kv, 1] -> 标量
        # pooled_hidden_states: [B, H_local_kv, D]
        # z_constrast: [B, H_local, ]
        # 在 Model 层需要 reduce
        return z_kv_batch.squeeze(-1).sum(dim=-1), entropy.mean(), pooled_hidden_states, z_constrast.squeeze(-1), attn_output, attn_weights, None

class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: PawQwen3Config,
        context_window_toggle: Optional[int] = 4096,
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3Attention(
            config=config, context_window_toggle=context_window_toggle
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self._fsdp_wrap = True

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.self_attn.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        return self.self_attn.get_masks()

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.self_attn.reset_masks(value)

    @torch.no_grad()
    def fill_masks_with_value(self, value):
        self.self_attn.fill_masks_with_value(value)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        seq_parallel_group: Optional[Any] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        current_tau: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        z_sum, entropy, pooled_hidden_states, z_constrast, hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
            segment_ids=segment_ids,
            range_ids=range_ids,
            task_ids=task_ids,
            current_tau=current_tau,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (z_sum, entropy, pooled_hidden_states, z_constrast, hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class Qwen3PreTrainedModel(PreTrainedModel):
    config_class = PawQwen3Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


@dataclass
class BaseModelOutputWithPastAndSparsity(ModelOutput):
    last_hidden_state: torch.FloatTensor
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None
    target_sparsity: Optional[torch.FloatTensor] = None
    sparsity_loss: Optional[torch.FloatTensor] = None
    # Diagnostics
    expected_model_sparsity: Optional[torch.FloatTensor] = None
    lambda1: Optional[torch.FloatTensor] = None
    lambda2: Optional[torch.FloatTensor] = None
    expected_z_mean: Optional[torch.FloatTensor] = None
    expected_z_std: Optional[torch.FloatTensor] = None
    log_alpha_mean: Optional[torch.FloatTensor] = None
    log_alpha_std: Optional[torch.FloatTensor] = None
    # Layer-wise sparsity diagnostics
    layerwise_model_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_target_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_sparsity_loss: Optional[torch.FloatTensor] = None  # scalar
    # contrastive_loss
    log_z_loss: Optional[torch.FloatTensor] = None
    head_entropy: Optional[torch.FloatTensor] = None
    

class Qwen3Model(Qwen3PreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen3DecoderLayer`]

    Args:
        config: PawQwen3Config
    """

    def __init__(
        self,
        config: PawQwen3Config,
    ):
        super().__init__(config)
        context_window_toggle = config.local_window_size
        disable_linear_regularization_term = config.disable_linear_regularization_term

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, context_window_toggle=context_window_toggle)
                for _ in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config=config)
        self.gradient_checkpointing = False

        self.total_num_heads = config.num_attention_heads * config.num_hidden_layers
        self.total_num_kv_heads = config.num_key_value_heads * config.num_hidden_layers

        self._dtype = self.norm.weight.dtype
        if disable_linear_regularization_term:
            self.sparsity_lambda_1 = torch.tensor([0.0], dtype=self._dtype)
        else:
            self.sparsity_lambda_1 = nn.Parameter(
                torch.tensor([0.0], dtype=self._dtype)
            )
        self.sparsity_lambda_2 = nn.Parameter(torch.tensor([0.0], dtype=self._dtype))
        
        if self.config.enable_lambda_task:
            self.num_tasks = 4
            self.sparsity_lambda1_task = nn.Parameter(
                torch.zeros(self.num_tasks, dtype=self._dtype)
            )
            self.sparsity_lambda2_task = nn.Parameter(
                torch.zeros(self.num_tasks, dtype=self._dtype)
            )
        else:
            self.sparsity_lambda1_task = None
            self.sparsity_lambda2_task = None


        self.threshold_for_deterministic = None
        if config.suggested_sparsity is not None:
            self.round_masks_for_sparsity(config.suggested_sparsity)

        self._erank_cache = {}
        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def reset_parameters(self):
        if self.config.enable_lambda_task:
            self.sparsity_lambda1_task.data.copy_(torch.rand_like(self.sparsity_lambda1_task) * 0.5)
            self.sparsity_lambda2_task.data.copy_(torch.rand_like(self.sparsity_lambda2_task) * 0.5)

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        for layer in self.layers:
            layer.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        masks = []
        for layer in self.layers:
            masks.append(layer.get_masks())
        return masks

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        for layer in self.layers:
            layer.reset_masks(value)
        self.sparsity_lambda_1.data.zero_()
        self.sparsity_lambda_2.data.zero_()

    @torch.no_grad()
    def get_sparsity(self):
        masks = self.get_masks()
        total_sum = 0
        for mask in masks:
            total_sum += mask.sum()
        return 1 - (total_sum / self.total_num_kv_heads)

    @torch.no_grad()
    def _pre_save_get_threshold(self):
        orig_threshold = self.threshold_for_deterministic

        sparsity_target = self.get_sparsity()
        l = 0
        r = 1
        while r - l > 1e-8:
            m = (l + r) / 2
            self.set_threshold_for_deterministic(m)
            if self.get_sparsity() > sparsity_target:
                r = m
            else:
                l = m
        m = (l + r) / 2

        self.config.suggested_threshold = m

    @torch.no_grad()
    def _get_avg_erank(self, path: str) -> torch.Tensor:
        key = os.path.abspath(path)
        if key in self._erank_cache:
            return self._erank_cache[key]
        erank_res = torch.load(key, map_location="cpu")
        # print(f"Loaded e-rank results from {key}: {erank_res}")
        avg_erank = erank_res["avg_erank"]
        self._erank_cache[key] = avg_erank
        return avg_erank

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        if start_with_keep:
            value_1 = 10.0  # Some high value
            value_2 = -10.0  # Some low value
        else:
            value_1 = -10.0
            value_2 = 10.0
        for l, layer in enumerate(self.layers):
            value = value_1 if l % (width_1 + width_2) < width_1 else value_2
            layer.fill_masks_with_value(value)

    @torch.no_grad()
    def load_masks(self, masks):
        for l in range(len(masks)):
            self.layers[l].fill_masks_with_value(masks[l])

    @torch.no_grad()
    def round_masks_for_sparsity(self, target_sparsity):
        masks = self.get_masks()
        # masks is a list of tensors, each tensor is of shape (num_key_value_heads,)
        # First find the number of high values
        num_high = int(sum([mask.shape[0] for mask in masks]) * (1 - target_sparsity))

        # Find the top-num_high values
        # Break ties randomly
        rng = torch.Generator()
        rng.manual_seed(42)
        value_list = [
            (i, j, masks[i][j], torch.rand(1, generator=rng).item())
            for i in range(len(masks))
            for j in range(masks[i].shape[0])
        ]
        # Sort by the random variable then resort by the value
        value_list.sort(key=lambda x: x[3])
        value_list.sort(key=lambda x: x[2], reverse=True)
        for i, j, _, _ in value_list[:num_high]:
            masks[i][j] = 10.0
        for i, j, _, _ in value_list[num_high:]:
            masks[i][j] = -10.0

        self.load_masks(masks)

        return self.get_sparsity()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        unpadded_lengths: Optional[Tuple[torch.Tensor]] = None,
        seq_parallel_group: Optional[Any] = None,
        target_sparsity: Optional[float] = None,
        current_tau: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        erank_analysis_path: Optional[str] = None,
        enable_contrastive_loss: bool = False,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        compute_sparsity = self.training
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time"
            )
        elif input_ids is None and inputs_embeds is None:
            raise ValueError(
                "You have to specify either decoder_input_ids or decoder_inputs_embeds"
            )

        # position_ids = None
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        # if dist.get_rank() == 0:
        #     breakpoint()
        # else:
        #     import time
        #     time.sleep()
            
        hidden_states = inputs_embeds

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = () if use_cache else None

        z_sum = 0 if compute_sparsity else None

        layer_z_sums = []  # 收集每层 z_sum 以计算逐层稀疏度
        # all_pooled_hidden_states = []
        head_entropy = 0 if compute_sparsity else None
        layer_z_constrast = []

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if past_key_values is not None and len(past_key_values) > idx:
                past_key_value = past_key_values[idx]
            else:
                past_key_value = None

            if self.gradient_checkpointing and self.training:
                layer_outputs = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    None,
                    unpadded_lengths,
                    output_attentions,
                    False,
                    seq_parallel_group,
                    use_reentrant=False,
                    segment_ids=segment_ids,
                    range_ids=range_ids,
                    task_ids=task_ids,
                    current_tau=current_tau,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_value,
                    unpadded_lengths=unpadded_lengths,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    seq_parallel_group=seq_parallel_group,
                    segment_ids=segment_ids,
                    range_ids=range_ids,
                    task_ids=task_ids,
                    current_tau=current_tau,
                )
            # z_layer_sum: [B, ]
            # entropy: 标量
            # pooled_hidden_states: [B, H_local_kv, D]
            # z_constrast: [B, H_local, ]
            z_layer_sum, entropy, pooled_hidden_states, z_constrast, hidden_states = layer_outputs[0], layer_outputs[1], layer_outputs[2], layer_outputs[3], layer_outputs[4]

            if compute_sparsity:
                z_sum += z_layer_sum # [B, ]
                head_entropy = (head_entropy + entropy) / 2 # 标量 为什么这样平均？
            layer_z_sums.append(z_layer_sum)
            layer_z_constrast.append(z_constrast)

            if use_cache:
                next_decoder_cache += (layer_outputs[5 if output_attentions else 4],)

            if output_attentions:
                all_self_attns += (layer_outputs[4],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None

        if compute_sparsity:
            if (
                seq_parallel_group is not None
                and dist.is_initialized()
                and dist.get_world_size(seq_parallel_group) > 1
            ):
                # z_sum: [B, ]
                # # 因为不同 Rank 负责不同 Heads，所有 Rank 的 z_sum 相加 = 全局 Active Heads 数量
                dist.all_reduce(z_sum, op=dist.ReduceOp.SUM, group=seq_parallel_group)
                # head_entropy 标量
                dist.all_reduce(head_entropy, op=dist.ReduceOp.SUM, group=seq_parallel_group)
                head_entropy = head_entropy / dist.get_world_size(seq_parallel_group)
                
                # layer_z_sums 是 list of tensors
                # 堆叠 -> AllReduce -> 解开
                if layer_z_sums:
                    stacked_layer_z = torch.stack(layer_z_sums) # [L, B]
                    dist.all_reduce(stacked_layer_z, op=dist.ReduceOp.SUM, group=seq_parallel_group)
                    # 重新拆回 list，如果不拆也可以直接用 stacked_layer_z 计算 layerwise_model_sparsity
                    layer_z_sums = list(stacked_layer_z)
                
                # total_num_heads 是全局的总 Head 数，z_sum 现在也是全局的 Active Head 数
            model_sparsity = 1 - (z_sum / self.total_num_heads)
                # breakpoint()
        else:
            model_sparsity = None
            z_loss = None
        
        if compute_sparsity:
            layerwise_model_sparsity = None
            layerwise_target = None

            if len(layer_z_sums) > 0:
                per_layer_heads = self.config.num_attention_heads
                layerwise_model_sparsity = (
                    1.0 - torch.stack(layer_z_sums) / per_layer_heads
                )  # (num_layers,)

            if target_sparsity is None:
                z_loss = None
            else:
                if self.config.enable_lambda_task:
                    diff = (model_sparsity - target_sparsity)

                    # per-sample lambda
                    lambda1_per_sample = self.sparsity_lambda1_task[task_ids]   # [B]
                    lambda2_per_sample = self.sparsity_lambda2_task[task_ids]   # [B]

                    # per-sample loss
                    per_sample_loss = (
                        lambda1_per_sample * diff
                        + lambda2_per_sample * diff.pow(2)
                    )

                    log_z_loss = per_sample_loss.detach()

                    task_losses = []
                    for task_id in range(self.num_tasks):
                        mask = (task_ids == task_id)
                        if mask.sum() > 0:
                            task_losses.append(per_sample_loss[mask].mean())

                    z_loss = torch.stack(task_losses).mean()
                else:
                    z_loss = (model_sparsity - target_sparsity).abs()
                    log_z_loss = z_loss.detach()
                    z_loss = z_loss.mean() 
        else:
            layerwise_model_sparsity = None
            layerwise_target = None
        
        if z_loss is not None:
            z_loss = z_loss.sum()
        
        if not return_dict:
            # return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, model_sparsity, target_sparsity, z_loss] if v is not None)
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_cache,
                    all_hidden_states,
                    all_self_attns,
                    model_sparsity,
                    target_sparsity,
                    z_loss,
                    self.sparsity_lambda1_task,
                    self.sparsity_lambda2_task,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndSparsity(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            model_sparsity=model_sparsity,
            target_sparsity=target_sparsity,
            sparsity_loss=z_loss,
            lambda1=self.sparsity_lambda1_task,
            lambda2=self.sparsity_lambda2_task,
            layerwise_model_sparsity=layerwise_model_sparsity,
            layerwise_target_sparsity=layerwise_target,
            log_z_loss=log_z_loss,
            head_entropy=head_entropy,
        )


@dataclass
class CausalLMOutputWithPastAndSparsity(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    model_sparsity: Optional[torch.FloatTensor] = None
    target_sparsity: Optional[torch.FloatTensor] = None
    sparsity_loss: Optional[torch.FloatTensor] = None
    # Diagnostics
    expected_model_sparsity: Optional[torch.FloatTensor] = None
    lambda1: Optional[torch.FloatTensor] = None
    lambda2: Optional[torch.FloatTensor] = None
    expected_z_mean: Optional[torch.FloatTensor] = None
    expected_z_std: Optional[torch.FloatTensor] = None
    log_alpha_mean: Optional[torch.FloatTensor] = None
    log_alpha_std: Optional[torch.FloatTensor] = None
    # Layer-wise sparsity diagnostics
    layerwise_model_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_target_sparsity: Optional[torch.FloatTensor] = None  # (num_layers,)
    layerwise_sparsity_loss: Optional[torch.FloatTensor] = None  # scalar
    task_ids: Optional[torch.FloatTensor] = None
    log_z_loss: Optional[torch.FloatTensor] = None
    head_entropy: Optional[torch.FloatTensor] = None

class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class PawQwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(
        self,
        config,
        enable_contrastive_loss=False,
    ):
        super().__init__(config)
        self.model = Qwen3Model(
            config,
        )
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.logit_block_size = int(os.environ.get("LOGIT_BLOCK_SIZE", 16384))
        self.enable_contrastive_loss = enable_contrastive_loss
        # Initialize weights and apply final processing
        self.post_init()

    @torch.no_grad()
    def set_threshold_for_deterministic(self, threshold_for_deterministic):
        self.model.set_threshold_for_deterministic(threshold_for_deterministic)

    @torch.no_grad()
    def get_masks(self):
        return self.model.get_masks()

    @torch.no_grad()
    def reset_masks(self, value=4.0):
        self.model.reset_masks(value)

    @torch.no_grad()
    def get_sparsity(self):
        return self.model.get_sparsity()

    @torch.no_grad()
    def reset_masks_with_stripe_pattern(self, width_1, width_2, start_with_keep=True):
        self.model.reset_masks_with_stripe_pattern(width_1, width_2, start_with_keep)

    @torch.no_grad()
    def load_masks(self, masks):
        self.model.load_masks(masks)

    @torch.no_grad()
    def round_masks_for_sparsity(self, target_sparsity):
        return self.model.round_masks_for_sparsity(target_sparsity)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def compute_loss(self, hidden_states, labels):
        if (labels != -100).sum() == 0:
            return torch.tensor(
                0.0, device=hidden_states.device, dtype=hidden_states.dtype
            )
        min_len = min(hidden_states.size(0), labels.size(0))
        hidden_states = hidden_states[:min_len]
        labels = labels[:min_len]

        logits = self.lm_head(hidden_states)
        if len(logits.shape) > 2:
            logits = logits.transpose(-1, -2)
        return F.cross_entropy(
            logits,
            labels,
            ignore_index=-100,
            reduction=("sum" if getattr(self, "token_scaled_loss", False) else "mean"),
        )

    def save_pretrained(self, *args, **kwargs):
        # First save the suggested threshold
        self.model._pre_save_get_threshold()
        return super().save_pretrained(*args, **kwargs)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        seq_lengths: Optional[torch.Tensor] = None,
        return_token_losses: bool = False,
        shifted_labels: Optional[torch.LongTensor] = None,
        seq_parallel_group: Optional[Any] = None,
        target_sparsity: Optional[float] = None,
        current_tau: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.LongTensor] = None,
        range_ids: Optional[torch.LongTensor] = None,
        task_ids: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, Qwen3ForCausalLM

        >>> model = Qwen3ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )
    
        if seq_lengths is not None:
            if inputs_embeds is not None:
                assert len(inputs_embeds.shape) == 2, (
                    "inputs_embeds should be a 2D tensor with `seq_lengths`"
                )
                # assert inputs_embeds.size(0) == seq_lengths.sum(), "inputs_embeds and seq_lengths should have the same batch size"
            else:
                assert len(input_ids.shape) == 1, (
                    "input_ids should be a 1D tensor with `seq_lengths`"
                )
                # assert input_ids.size(0) == seq_lengths.sum(), "input_ids and seq_lengths should have the same batch size"

            assert attention_mask is None or attention_mask.all().item(), (
                "attention_mask should be None or all ones for `seq_lengths`"
            )
            assert not use_cache, "use_cache is not supported with `seq_lengths`"
            max_seqlen = (seq_lengths[1:]-seq_lengths[:-1]).max().item()
            unpadded_lengths = (seq_lengths, max_seqlen)
        
        elif attention_mask is not None and not use_cache and attention_mask.size(0) != 1:
            # breakpoint()
            if inputs_embeds is not None:
                bsz = inputs_embeds.size(0)
                inputs_embeds, unpad_indices, cu_seqlens, max_seqlen = unpad_input(
                    inputs_embeds, attention_mask
                )
            else:
                bsz = input_ids.size(0)
                tmp = input_ids.unsqueeze(-1)
                input_ids, unpad_indices, cu_seqlens, max_seqlen = unpad_input(tmp, attention_mask)
                max_seqlen_for_pad_seq = attention_mask.size(-1)
                input_ids = input_ids.squeeze(-1)
            unpadded_lengths = (cu_seqlens, max_seqlen)
        else:
            unpadded_lengths = None

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            unpadded_lengths=unpadded_lengths,
            seq_parallel_group=seq_parallel_group,
            target_sparsity=target_sparsity,
            current_tau=current_tau,
            segment_ids=segment_ids,
            range_ids=range_ids,
            task_ids=task_ids,
            enable_contrastive_loss=self.enable_contrastive_loss,
        )
        
        hidden_states = outputs[0]
        if seq_lengths is None and unpadded_lengths is not None:
            hidden_states = pad_input(hidden_states, unpad_indices, bsz, max_seqlen_for_pad_seq)
        if labels is not None or shifted_labels is not None:
            if shifted_labels is not None:
                labels = shifted_labels.reshape(-1)
                hidden_states = hidden_states.reshape(-1, hidden_states.size(-1))
            else:
                labels = labels[..., 1:].reshape(-1).contiguous()
                hidden_states = hidden_states[..., :-1, :].reshape(
                    -1, hidden_states.size(-1)
                ).contiguous()
            if self.logit_block_size > 0:
                num_valid_labels = (labels != -100).sum()
                hidden_states = torch.split(hidden_states, self.logit_block_size, dim=0)
                labels = torch.split(labels, self.logit_block_size, dim=0)

                if getattr(self, "token_scaled_loss", False):
                    loss = sum(
                        torch.utils.checkpoint.checkpoint(
                            self.compute_loss,
                            hidden_state_block,
                            label_block,
                            use_reentrant=False,
                        )
                        for hidden_state_block, label_block in zip(
                            hidden_states, labels
                        )
                    )
                else:
                    loss = sum(
                        ((label_block != -100).sum() / max(num_valid_labels.item(), 1))
                        * torch.utils.checkpoint.checkpoint(
                            self.compute_loss,
                            hidden_state_block,
                            label_block,
                            use_reentrant=False,
                        )
                        for hidden_state_block, label_block in zip(
                            hidden_states, labels
                        )
                    )
            else:
                loss = self.compute_loss(hidden_states, labels)

            logits = None
        else:
            logits = self.lm_head(hidden_states)
            loss = None
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPastAndSparsity(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            model_sparsity=outputs.model_sparsity,
            target_sparsity=outputs.target_sparsity,
            sparsity_loss=outputs.sparsity_loss,
            lambda1=outputs.lambda1,
            lambda2=outputs.lambda2,
            layerwise_model_sparsity=outputs.layerwise_model_sparsity,
            layerwise_target_sparsity=outputs.layerwise_target_sparsity,
            task_ids=task_ids,
            log_z_loss=outputs.log_z_loss,
            head_entropy=outputs.head_entropy,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        **kwargs,
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(
                    past_state.index_select(0, beam_idx.to(past_state.device))
                    for past_state in layer_past
                ),
            )
        return reordered_past
