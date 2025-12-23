from typing import Optional, Tuple, List

import torch
import torch.functional as F

from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaModel,
    CausalLMOutputWithPast,
    Union,
    BaseModelOutputWithPast,
    apply_rotary_pos_emb,
)

from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3ForCausalLM,
    Qwen3Model,
)
from torch.nn import CrossEntropyLoss
import types

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0)
    )
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def old_flash_attention_2_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    # LlamaFlashAttention2 attention does not support output_attentions
    output_attentions = False

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Flash attention requires the input to have the shape
    # batch_size x seq_length x head_dime x hidden_dim
    # therefore we just need to keep the original shape
    query_states = query_states.view(
        bsz,
        q_len,
        self.num_heads
        if hasattr(self, "num_heads")
        else self.config.num_attention_heads
        if hasattr(self, "num_heads")
        else self.config.num_attention_heads,
        self.head_dim,
    ).transpose(1, 2)
    key_states = key_states.view(
        bsz,
        q_len,
        self.num_key_value_heads
        if hasattr(self, "num_key_value_heads")
        else self.config.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)
    value_states = value_states.view(
        bsz,
        q_len,
        self.num_key_value_heads
        if hasattr(self, "num_key_value_heads")
        else self.config.num_key_value_heads,
        self.head_dim,
    ).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[-2]

    cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_value is not None:
        # reuse k, v, self_attention
        key_states = torch.cat([past_key_value[0], key_states], dim=2)
        value_states = torch.cat([past_key_value[1], value_states], dim=2)

    past_key_value = (key_states, value_states) if use_cache else None

    query_states = query_states.transpose(1, 2)
    key_states = key_states.transpose(1, 2)
    value_states = value_states.transpose(1, 2)

    # TODO: llama does not have dropout in the config??
    # It is recommended to use dropout with FA according to the docs
    # when training.
    dropout_rate = 0.0  # if not self.training else self.attn_dropout

    # In PEFT, usually we cast the layer norms in float32 for training stability reasons
    # therefore the input hidden states gets silently casted in float32. Hence, we need
    # cast them back in float16 just to be sure everything works as expected.
    # This might slowdown training & inference so it is recommended to not cast the LayerNorms
    # in fp32. (LlamaRMSNorm handles it correctly)
    input_dtype = query_states.dtype
    if input_dtype == torch.float32:
        query_states = query_states.to(torch.float16)
        key_states = key_states.to(torch.float16)
        value_states = value_states.to(torch.float16)

    attn_output = self._flash_attention_forward(
        query_states,
        key_states,
        value_states,
        padding_mask,
        q_len,
        dropout=dropout_rate,
    )

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def _flash_attention_forward(
    self,
    query_states,
    key_states,
    value_states,
    padding_mask,
    query_length,
    dropout=0.0,
    softmax_scale=None,
):
    """
    Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
    first unpad the input, then computes the attention scores and pad the final attention scores.

    Args:
        query_states (`torch.Tensor`):
            Input query states to be passed to Flash Attention API
        key_states (`torch.Tensor`):
            Input key states to be passed to Flash Attention API
        value_states (`torch.Tensor`):
            Input value states to be passed to Flash Attention API
        padding_mask (`torch.Tensor`):
            The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
            position of padding tokens and 1 for the position of non-padding tokens.
        dropout (`int`, *optional*):
            Attention dropout
        softmax_scale (`float`, *optional*):
            The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
    """
    # Contains at least one padding token in the sequence
    if padding_mask is not None:
        batch_size = query_states.shape[0]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            cu_seq_lens,
            max_seq_lens,
        ) = self._upad_input(
            query_states, key_states, value_states, padding_mask, query_length
        )

        cu_seqlens_q, cu_seqlens_k = cu_seq_lens
        max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=True,
        )

        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, query_length)
    else:
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            dropout,
            softmax_scale=softmax_scale,
            causal=True,
        )

    return attn_output


def _upad_input(self, query_layer, key_layer, value_layer, padding_mask, query_length):
    indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(padding_mask)
    batch_size, kv_seq_len, num_key_value_heads, head_dim = key_layer.shape

    key_layer = index_first_axis(
        key_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    value_layer = index_first_axis(
        value_layer.reshape(batch_size * kv_seq_len, num_key_value_heads, head_dim),
        indices_k,
    )
    if query_length == kv_seq_len:
        query_layer = index_first_axis(
            query_layer.reshape(
                batch_size * kv_seq_len,
                self.num_heads
                if hasattr(self, "num_heads")
                else self.config.num_attention_heads
                if hasattr(self, "num_heads")
                else self.config.num_attention_heads,
                head_dim,
            ),
            indices_k,
        )
        cu_seqlens_q = cu_seqlens_k
        max_seqlen_in_batch_q = max_seqlen_in_batch_k
        indices_q = indices_k
    elif query_length == 1:
        max_seqlen_in_batch_q = 1
        cu_seqlens_q = torch.arange(
            batch_size + 1, dtype=torch.int32, device=query_layer.device
        )  # There is a memcpy here, that is very bad.
        indices_q = cu_seqlens_q[:-1]
        query_layer = query_layer.squeeze(1)
    else:
        # The -q_len: slice assumes left padding.
        padding_mask = padding_mask[:, -query_length:]
        query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
            query_layer, padding_mask
        )

    return (
        query_layer,
        key_layer,
        value_layer,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    )

def old_qwen_for_causal_lm_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, CausalLMOutputWithPast]:
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
    )

    hidden_states = outputs[0]

    if self.training:
        logits = self.lm_head(hidden_states)
    else:
        logits = self.lm_head(hidden_states[:, -1:, :])

    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )



def old_qwen_model_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Union[Tuple, BaseModelOutputWithPast]:
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
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def old_qwen_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    position_embeddings: Optional[torch.Tensor] = None,
    padding_mask: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        padding_mask=padding_mask,
        position_embeddings=position_embeddings,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs


# qwen
def enable_tuple_kv_cache_for_qwen(model: Qwen3ForCausalLM):
    print("Enabling tuple KV cache for Qwen")
    model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None
    model.model.forward = types.MethodType(old_qwen_model_forward, model.model)
    for idx in range(len(model.model.layers)):
        model.model.layers[idx].forward = types.MethodType(
            old_qwen_decoder_layer_forward, model.model.layers[idx]
        )
        model.model.layers[idx].self_attn.forward = types.MethodType(
            old_flash_attention_2_forward, model.model.layers[idx].self_attn
        )
        model.model.layers[idx].self_attn._upad_input = types.MethodType(
            _upad_input, model.model.layers[idx].self_attn
        )
        model.model.layers[idx].self_attn._flash_attention_forward = types.MethodType(
            _flash_attention_forward, model.model.layers[idx].self_attn
        )
    model.forward = types.MethodType(old_qwen_for_causal_lm_forward, model)


import os
from torch import nn

from .utils import (
    reorder_linear_weights,
    reorder_full_attn_heads,
)

from .utils import (
    DuoAttentionStaticKVCache,
    enable_duo_attention_static_kv_cache_for_llama,
)
from .utils import apply_rope_inplace, enable_flashinfer_rmsnorm

from tensor_parallel.pretrained_model import TensorParallelPreTrainedModel
from flash_attn import flash_attn_func, flash_attn_with_kvcache



def qwen_duo_attention_forward_one_way_reordered(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    position_embeddings: Optional[torch.Tensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(
        bsz,
        q_len,
        self.num_heads
        if hasattr(self, "num_heads")
        else self.config.num_attention_heads,
        self.head_dim,
    )
    key_states = key_states.view(
        bsz,
        q_len,
        self.num_key_value_heads
        if hasattr(self, "num_key_value_heads")
        else self.config.num_key_value_heads,
        self.head_dim,
    )
    value_states = value_states.view(
        bsz,
        q_len,
        self.num_key_value_heads
        if hasattr(self, "num_key_value_heads")
        else self.config.num_key_value_heads,
        self.head_dim,
    )

    # new data structure for past_key_value
    # past_key_value = (full_KV, streaming_KV)
    # full_KV: (2 x bsz, num_full_key_value_heads, full_kv_seq_len, head_dim)
    # streaming_KV: (2 x bsz, num_streaming_key_value_heads, cache_size, head_dim)

    kv_seq_len = key_states.shape[1]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )

    if not hasattr(self, "full_attn_head_mask") or self.full_attn_head_mask is None:
        self.full_attn_head_mask = self.full_attention_heads > 0.5
        self.num_full_attn_head = self.full_attn_head_mask.sum().item()
        self.num_streaming_attn_head = (
            self.num_key_value_heads
            if hasattr(self, "num_key_value_heads")
            else self.config.num_key_value_heads - self.num_full_attn_head
        )

        self.num_full_query_head = self.num_full_attn_head * self.num_key_value_groups
        self.num_streaming_query_head = (
            self.num_heads
            if hasattr(self, "num_heads")
            else self.config.num_attention_heads - self.num_full_query_head
        )

    full_key_states = key_states[:, :, : self.num_full_attn_head, :]
    full_value_states = value_states[:, :, : self.num_full_attn_head, :]

    streaming_key_states = key_states[:, :, self.num_full_attn_head :, :]
    streaming_value_states = value_states[:, :, self.num_full_attn_head :, :]

    if past_key_value is not None:
        # reuse k, v, self_attention
        past_full_KV = past_key_value[0].transpose(1, 2)
        past_streaming_KV = past_key_value[1].transpose(1, 2)

        past_full_key_states = past_full_KV[:bsz]
        past_full_value_states = past_full_KV[bsz:]

        full_key_states = torch.cat([past_full_key_states, full_key_states], dim=1)
        full_value_states = torch.cat(
            [past_full_value_states, full_value_states], dim=1
        )

        past_streaming_key_states = past_streaming_KV[:bsz]
        past_streaming_value_states = past_streaming_KV[bsz:]

        streaming_key_states = torch.cat(
            [past_streaming_key_states, streaming_key_states], dim=1
        )
        streaming_value_states = torch.cat(
            [past_streaming_value_states, streaming_value_states], dim=1
        )

    if q_len == kv_seq_len:
        # pre-filling: use flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )
    else:
        # decoding or continous filling
        if self.num_full_attn_head > 0:
            full_query_states = query_states[:, :, : self.num_full_query_head, :]

            full_attn_output = flash_attn_func(
                full_query_states,
                full_key_states,
                full_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            full_attn_output = None

        if self.num_streaming_attn_head > 0:
            streaming_query_states = query_states[:, :, self.num_full_query_head :, :]

            streaming_attn_output = flash_attn_func(
                streaming_query_states,
                streaming_key_states,
                streaming_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            streaming_attn_output = None

        if full_attn_output is None:
            attn_output = streaming_attn_output
        elif streaming_attn_output is None:
            attn_output = full_attn_output
        else:
            attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=2)

    attn_output = attn_output.reshape(
        bsz,
        q_len,
        self.num_heads if hasattr(self, "num_heads")
        else self.config.num_attention_heads * self.head_dim,
    )

    attn_output = self.o_proj(attn_output)

    if streaming_key_states.shape[1] > self.recent_size + self.sink_size:
        recent_key_states = streaming_key_states[:, -self.recent_size :, :, :].clone()
        streaming_key_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_key_states)
        streaming_key_states = streaming_key_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

        recent_value_states = streaming_value_states[
            :, -self.recent_size :, :, :
        ].clone()
        streaming_value_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_value_states)
        streaming_value_states = streaming_value_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

    past_key_value = (
        (
            torch.cat([full_key_states, full_value_states], dim=0).transpose(1, 2),
            torch.cat([streaming_key_states, streaming_value_states], dim=0).transpose(
                1, 2
            ),
        )
        if use_cache
        else None
    )

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value

# Qwen 
def enable_qwen_duo_attention_eval(
    model: Qwen3ForCausalLM,
    full_attention_heads,
    sink_size,
    recent_size,
):
    enable_tuple_kv_cache_for_qwen(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        module.forward = types.MethodType(
            qwen_duo_attention_forward_one_way_reordered, module
        )
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )
        layer_full_attention_heads = reorder_full_attn_heads(layer_full_attention_heads)

        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_buffer(
            "full_attention_heads",
            layer_full_attention_heads,
        )


def llama_duo_attention_forward_two_way(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz_x_2, q_len, _ = hidden_states.size()

    assert bsz_x_2 % 2 == 0

    bsz = bsz_x_2 // 2

    full_hidden_states = hidden_states[:bsz]
    streaming_hidden_states = hidden_states[bsz:]

    with torch.no_grad():
        full_query_states = self.q_proj(full_hidden_states)
        full_key_states = self.k_proj(full_hidden_states)
        full_value_states = self.v_proj(full_hidden_states)
        full_query_states = full_query_states.view(
            bsz, q_len, self.config.num_attention_heads, self.head_dim
        )
        full_key_states = full_key_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        )
        full_value_states = full_value_states.view(
            bsz, q_len, self.config.num_key_value_heads, self.head_dim
        )

    streaming_query_states = self.q_proj(streaming_hidden_states)
    streaming_key_states = self.k_proj(streaming_hidden_states)
    streaming_value_states = self.v_proj(streaming_hidden_states)
    streaming_query_states = streaming_query_states.view(
        bsz, q_len, self.config.num_attention_heads, self.head_dim
    )
    streaming_key_states = streaming_key_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )
    streaming_value_states = streaming_value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    cos, sin = position_embeddings

    with torch.no_grad():
        full_query_states, full_key_states = apply_rotary_pos_emb(
            full_query_states,
            full_key_states,
            cos,
            sin,
            unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
        )
        full_attn_output = self.full_attn_func(
            full_query_states,
            full_key_states,
            full_value_states,
            causal=True,
            dropout_p=0.0,
        )

    streaming_query_states, streaming_key_states = apply_rotary_pos_emb(
        streaming_query_states,
        streaming_key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )

    streaming_attn_output = self.streaming_attn_func(
        streaming_query_states,
        streaming_key_states,
        streaming_value_states,
        self.streaming_mask,
    )

    full_attention_heads = (
        self.full_attention_heads.clamp(0, 1)
        .view(1, 1, self.config.num_key_value_heads, 1, 1)
        .expand(1, 1, self.config.num_key_value_heads, self.num_key_value_groups, 1)
        .reshape(1, 1, self.config.num_attention_heads, 1)
    )

    streaming_attn_output = (
        1 - full_attention_heads
    ) * streaming_attn_output + full_attention_heads * full_attn_output

    with torch.no_grad():
        full_attn_output = full_attn_output.reshape(bsz, q_len, self.config.hidden_size)
        full_attn_output = self.o_proj(full_attn_output)

    streaming_attn_output = streaming_attn_output.reshape(bsz, q_len, self.config.hidden_size)
    streaming_attn_output = self.o_proj(streaming_attn_output)

    attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=0)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_duo_attention_forward_one_way_reordered(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)
    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    # new data structure for past_key_value
    # past_key_value = (full_KV, streaming_KV)
    # full_KV: (2 x bsz, num_full_key_value_heads, full_kv_seq_len, head_dim)
    # streaming_KV: (2 x bsz, num_streaming_key_value_heads, cache_size, head_dim)

    kv_seq_len = key_states.shape[1]
    if past_key_value is not None:
        kv_seq_len += past_key_value[0].shape[2]

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(
        query_states,
        key_states,
        cos,
        sin,
        unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    )

    if not hasattr(self, "full_attn_head_mask") or self.full_attn_head_mask is None:
        self.full_attn_head_mask = self.full_attention_heads > 0.5
        self.num_full_attn_head = self.full_attn_head_mask.sum().item()
        self.num_streaming_attn_head = (
            self.config.num_key_value_heads - self.num_full_attn_head
        )

        self.num_full_query_head = self.num_full_attn_head * self.num_key_value_groups
        self.num_streaming_query_head = self.config.num_attention_heads - self.num_full_query_head

    full_key_states = key_states[:, :, : self.num_full_attn_head, :]
    full_value_states = value_states[:, :, : self.num_full_attn_head, :]

    streaming_key_states = key_states[:, :, self.num_full_attn_head :, :]
    streaming_value_states = value_states[:, :, self.num_full_attn_head :, :]

    if past_key_value is not None:
        # reuse k, v, self_attention
        past_full_KV = past_key_value[0].transpose(1, 2)
        past_streaming_KV = past_key_value[1].transpose(1, 2)

        past_full_key_states = past_full_KV[:bsz]
        past_full_value_states = past_full_KV[bsz:]

        full_key_states = torch.cat([past_full_key_states, full_key_states], dim=1)
        full_value_states = torch.cat(
            [past_full_value_states, full_value_states], dim=1
        )

        past_streaming_key_states = past_streaming_KV[:bsz]
        past_streaming_value_states = past_streaming_KV[bsz:]

        streaming_key_states = torch.cat(
            [past_streaming_key_states, streaming_key_states], dim=1
        )
        streaming_value_states = torch.cat(
            [past_streaming_value_states, streaming_value_states], dim=1
        )

    if q_len == kv_seq_len:
        # pre-filling: use flash attention
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )
    else:
        # decoding or continous filling
        if self.num_full_attn_head > 0:
            full_query_states = query_states[:, :, : self.num_full_query_head, :]

            full_attn_output = flash_attn_func(
                full_query_states,
                full_key_states,
                full_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            full_attn_output = None

        if self.num_streaming_attn_head > 0:
            streaming_query_states = query_states[:, :, self.num_full_query_head :, :]

            streaming_attn_output = flash_attn_func(
                streaming_query_states,
                streaming_key_states,
                streaming_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            streaming_attn_output = None

        if full_attn_output is None:
            attn_output = streaming_attn_output
        elif streaming_attn_output is None:
            attn_output = full_attn_output
        else:
            attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=2)

    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

    attn_output = self.o_proj(attn_output)

    if streaming_key_states.shape[1] > self.recent_size + self.sink_size:
        recent_key_states = streaming_key_states[:, -self.recent_size :, :, :].clone()
        streaming_key_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_key_states)
        streaming_key_states = streaming_key_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

        recent_value_states = streaming_value_states[
            :, -self.recent_size :, :, :
        ].clone()
        streaming_value_states[
            :, self.sink_size : self.sink_size + self.recent_size, :, :
        ].copy_(recent_value_states)
        streaming_value_states = streaming_value_states[
            :, : self.sink_size + self.recent_size, :, :
        ]

    past_key_value = (
        (
            torch.cat([full_key_states, full_value_states], dim=0).transpose(1, 2),
            torch.cat([streaming_key_states, streaming_value_states], dim=0).transpose(
                1, 2
            ),
        )
        if use_cache
        else None
    )

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights, past_key_value


def llama_duo_attention_forward_one_way_reordered_static(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kv_cache: Optional[DuoAttentionStaticKVCache] = None,
    layer_idx: int = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    **kwargs,
):
    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.config.num_attention_heads, self.head_dim)
    key_states = key_states.view(bsz, q_len, self.config.num_key_value_heads, self.head_dim)
    value_states = value_states.view(
        bsz, q_len, self.config.num_key_value_heads, self.head_dim
    )

    kv_seq_len = q_len
    if kv_cache is not None:
        kv_seq_len += kv_cache.kv_seq_len

    # Replace the Huggingface's apply rotory pos emb with FlashInfer's rope

    # cos, sin = self.rotary_emb(value_states, position_ids)
    # query_states, key_states = apply_rotary_pos_emb(
    #     query_states,
    #     key_states,
    #     cos,
    #     sin,
    #     unsqueeze_dim=2,  # unsqueeze_dim=2 for the flash attention
    # )

    rope_scale = 1.0
    if self.config.rope_scaling is not None:
        rope_scale = self.config.rope_scaling.get("factor", 1.0)
    apply_rope_inplace(
        query_states, key_states, position_ids[:, 0], rope_scale, self.rope_theta
    )

    (
        full_key_states,
        full_value_states,
        streaming_key_states,
        streaming_value_states,
    ) = kv_cache.split_kv(layer_idx, key_states, value_states)
    full_key_states, full_value_states = kv_cache.put_full_kv(
        layer_idx, full_key_states, full_value_states
    )

    if q_len == kv_seq_len:
        # Initial pre-filling
        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            causal=True,
            dropout_p=0.0,
        )

    else:
        # Decoding or continous filling
        num_full_query_head = (
            kv_cache.num_full_kv_head_list[layer_idx] * self.num_key_value_groups
        )

        (
            cached_streaming_key_states,
            cached_streaming_value_states,
        ) = kv_cache.get_streaming_kv(layer_idx)

        streaming_key_states = torch.cat(
            [cached_streaming_key_states, streaming_key_states], dim=1
        )
        streaming_value_states = torch.cat(
            [cached_streaming_value_states, streaming_value_states], dim=1
        )

        if num_full_query_head > 0:
            full_query_states = query_states[:, :, :num_full_query_head, :]
            full_attn_output = flash_attn_func(
                full_query_states,
                full_key_states,
                full_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            full_attn_output = None

        if self.config.num_attention_heads - num_full_query_head > 0:
            streaming_query_states = query_states[:, :, num_full_query_head:, :]
            streaming_attn_output = flash_attn_func(
                streaming_query_states,
                streaming_key_states,
                streaming_value_states,
                causal=True,
                dropout_p=0.0,
            )
        else:
            streaming_attn_output = None

        if full_attn_output is None:
            attn_output = streaming_attn_output
        elif streaming_attn_output is None:
            attn_output = full_attn_output
        else:
            attn_output = torch.cat([full_attn_output, streaming_attn_output], dim=2)

    kv_cache.compress_and_replace_streaming_kv(
        layer_idx, streaming_key_states, streaming_value_states
    )

    attn_output = attn_output.reshape(bsz, q_len, self.config.hidden_size)

    attn_output = self.o_proj(attn_output)

    if not output_attentions:
        attn_weights = None

    return attn_output, attn_weights


# From Huggingface's Transformers v4.34.0. This is the forward method of LlamaModel using the tuple style KV cache.
def old_llama_model_forward(
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
) -> Union[Tuple, BaseModelOutputWithPast]:
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
            "You cannot specify both input_ids and inputs_embeds at the same time"
        )
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape
    elif inputs_embeds is not None:
        batch_size, seq_length, _ = inputs_embeds.shape
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    seq_length_with_past = seq_length
    past_key_values_length = 0

    if past_key_values is not None:
        past_key_values_length = past_key_values[0][0].shape[2]
        seq_length_with_past = seq_length_with_past + past_key_values_length

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length,
            seq_length + past_key_values_length,
            dtype=torch.long,
            device=device,
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
    else:
        position_ids = position_ids.view(-1, seq_length).long()

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # embed positions
    if attention_mask is None:
        attention_mask = torch.ones(
            (batch_size, seq_length_with_past),
            dtype=torch.bool,
            device=inputs_embeds.device,
        )
        padding_mask = None
    else:
        if 0 in attention_mask:
            padding_mask = attention_mask
        else:
            padding_mask = None

    attention_mask = self._prepare_decoder_attention_mask(
        attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
    )

    hidden_states = inputs_embeds
    
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = () if use_cache else None

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        past_key_value = past_key_values[idx] if past_key_values is not None else None

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
            position_embeddings=position_embeddings,
        )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if not return_dict:
        return tuple(
            v
            for v in [hidden_states, next_cache, all_hidden_states, all_self_attns]
            if v is not None
        )
    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=next_cache,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )
    


# From Huggingface's Transformers v4.34.0. This is the forward method of LlamaDecoderLayer using the tuple style KV cache.
def old_llama_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights, present_key_value = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
        use_cache=use_cache,
        position_embeddings=position_embeddings,
        padding_mask=padding_mask,
    )
    hidden_states = residual + hidden_states

    # Fully Connected
    residual = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    outputs = (hidden_states,)

    if output_attentions:
        outputs += (self_attn_weights,)

    if use_cache:
        outputs += (present_key_value,)

    return outputs

# From Huggingface's Transformers v4.34.0. This is the forward method of LlamaForCausalLM using the tuple style KV cache.
def old_llama_for_causal_lm_forward(
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
    cache_position: Optional[torch.LongTensor] = None,
) -> Union[Tuple, CausalLMOutputWithPast]:
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
    )

    hidden_states = outputs[0]

    if self.training:
        logits = self.lm_head(hidden_states)
    else:
        logits = self.lm_head(hidden_states[:, -1:, :])

    logits = logits.float()

    loss = None
    if labels is not None:
        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, self.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )

def enable_tuple_kv_cache_for_llama(model: LlamaForCausalLM):
    print("Enabling tuple KV cache for Llama")
    model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None
    model.model.forward = types.MethodType(old_llama_model_forward, model.model)
    for idx in range(len(model.model.layers)):
        model.model.layers[idx].forward = types.MethodType(
            old_llama_decoder_layer_forward, model.model.layers[idx]
        )
        model.model.layers[idx].self_attn.forward = types.MethodType(
            old_flash_attention_2_forward, model.model.layers[idx].self_attn
        )
        model.model.layers[idx].self_attn._upad_input = types.MethodType(
            _upad_input, model.model.layers[idx].self_attn
        )
        model.model.layers[idx].self_attn._flash_attention_forward = types.MethodType(
            _flash_attention_forward, model.model.layers[idx].self_attn
        )
    model.forward = types.MethodType(old_llama_for_causal_lm_forward, model)

def enable_llama_duo_attention_eval(
    model: LlamaForCausalLM,
    full_attention_heads,
    sink_size,
    recent_size,
):
    enable_tuple_kv_cache_for_llama(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        module.forward = types.MethodType(
            llama_duo_attention_forward_one_way_reordered, module
        )
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )
        layer_full_attention_heads = reorder_full_attn_heads(layer_full_attention_heads)

        module.sink_size = sink_size
        module.recent_size = recent_size
        module.register_buffer(
            "full_attention_heads",
            layer_full_attention_heads,
        )


def enable_llama_duo_attention_static_kv_cache_eval(
    model: LlamaForCausalLM,
    full_attention_heads,
):
    enable_duo_attention_static_kv_cache_for_llama(model)
    enable_flashinfer_rmsnorm(model)

    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    for idx, layer in enumerate(model.model.layers):
        module = layer.self_attn
        layer_full_attention_heads = torch.tensor(
            full_attention_heads[idx], device=device, dtype=dtype
        )

        module.forward = types.MethodType(
            llama_duo_attention_forward_one_way_reordered_static, module
        )
        module.q_proj = reorder_linear_weights(
            module.q_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "out",
        )
        module.k_proj = reorder_linear_weights(
            module.k_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.v_proj = reorder_linear_weights(
            module.v_proj,
            layer_full_attention_heads,
            module.head_dim,
            "out",
        )
        module.o_proj = reorder_linear_weights(
            module.o_proj,
            layer_full_attention_heads,
            module.num_key_value_groups * module.head_dim,
            "in",
        )


def get_llama_full_attention_heads(model):
    full_attention_heads = []
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            sharded_full_attention_heads = []
            for layer in shard.model.layers:
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                sharded_full_attention_heads.append(module.full_attention_heads)
            full_attention_heads.append(sharded_full_attention_heads)
        # concatenate the full_attention_heads from all shards, getting a list of tensors with len = num_layers
        device = full_attention_heads[0][0].device
        full_attention_heads = [
            torch.cat(
                [
                    sharded_heads[layer_idx].to(device)
                    for sharded_heads in full_attention_heads
                ]
            )
            for layer_idx in range(len(full_attention_heads[0]))
        ]
    elif isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            full_attention_heads.append(module.full_attention_heads)
    elif isinstance(model, LlamaModel):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            full_attention_heads.append(module.full_attention_heads)
    else:
        raise ValueError("Model type not supported")

    return full_attention_heads


def set_llama_full_attention_heads(model, full_attention_heads):
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            for layer_idx, layer in enumerate(shard.model.layers):
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                    module.full_attention_heads.device,
                    module.full_attention_heads.dtype,
                )
    elif isinstance(model, LlamaForCausalLM):
        for layer_idx, layer in enumerate(model.model.layers):
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                module.full_attention_heads.device, module.full_attention_heads.dtype
            )
    elif isinstance(model, LlamaModel):
        for layer_idx, layer in enumerate(model.layers):
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            module.full_attention_heads.data = full_attention_heads[layer_idx].to(
                module.full_attention_heads.device, module.full_attention_heads.dtype
            )
    else:
        raise ValueError("Model type not supported")


def map_llama_full_attention_heads(model, func):
    if isinstance(model, TensorParallelPreTrainedModel):
        for shard in model.wrapped_model.module_shards:
            for layer in shard.model.layers:
                module = layer.self_attn.tp_wrapped_module
                if not hasattr(module, "full_attention_heads"):
                    continue
                func(module.full_attention_heads)
    elif isinstance(model, LlamaForCausalLM):
        for layer in model.model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            func(module.full_attention_heads)
    elif isinstance(model, LlamaModel):
        for layer in model.layers:
            module = layer.self_attn
            if not hasattr(module, "full_attention_heads"):
                continue
            func(module.full_attention_heads)
    else:
        raise ValueError("Model type not supported")
