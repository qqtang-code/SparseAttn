import torch

def create_causal_mask(batch_size, head_num, block_size, block_num, divide_block_num):
    """
    Creates a causal attention mask used in transformer-based models.

    Parameters:
    - batch_size (int): The number of sequences in the batch.
    - head_num (int): The number of attention heads.
    - block_size (int): The size of each block in the sequence.
    - block_num (int): The total number of blocks in the sequence.
    - divide_block_num (int): The block index at which causality is applied.

    Returns:
    - torch.Tensor: A mask tensor of shape (batch_size, head_num, block_size, total_size)
    where total_size = block_size * block_num. The mask enforces causal attention by
    setting certain positions to `-inf` to prevent information leakage from future tokens.
    """
    divide_block_num += 1
    if divide_block_num < 1 or divide_block_num > block_num:
        raise ValueError(
            f"divide_block_num ({divide_block_num}) must be between 1 and block_num ({block_num})."
        )

    total_size = block_size * block_num
    device = "cuda"
    mask = torch.zeros(block_size, total_size, device=device)
    if divide_block_num < block_num:
        mask[:, divide_block_num * block_size :] = float("-inf")

    if divide_block_num - 1 < block_num:
        start_col = (divide_block_num - 1) * block_size
        end_col = start_col + block_size
        upper_tri_mask = torch.triu(
            torch.full((block_size, block_size), float("-inf"), device=device),
            diagonal=1,
        )
        mask[:, start_col:end_col] = upper_tri_mask

    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.expand(batch_size, head_num, block_size, total_size)
    return mask


def find_blocks_chunked(
    input_tensor,
    current_index,
    threshold,
    num_to_choose,
    decoding: bool,
    mode: str = "both",
    causal=True,
):
    """
    Finds and selects relevant blocks of attention for transformer-based models based on a
    threshold or a predefined number of blocks.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor of shape (batch_size, head_num, chunk_num, block_num).
    - current_index (int): The current index in the sequence processing.
    - threshold (float or None): A threshold value used to determine the minimum attention weight sum.
    - num_to_choose (int or None): The number of blocks to be selected, ensuring sufficient information retrieval.
    - decoding (bool): If True, operates in decoding mode; otherwise, it's in encoding mode.
    - mode (str): Defines the processing mode, either 'both', 'prefill', or 'decode'.
    - causal (bool): If True, applies causal masking to prevent future information leakage.

    Returns:
    - torch.Tensor: A boolean mask of shape (batch_size, head_num, chunk_num, block_num),
    indicating which blocks should be attended to.
    """
    assert threshold is None or num_to_choose is None
    batch_size, head_num, chunk_num, block_num = input_tensor.shape
    # 0 -- -- -- -- current_index
    # 0 -- -- -- -- -- current_index+1
    # 0 -- -- -- -- -- ----------- current_index + chunk_num - 1
    if mode == "prefill" and decoding:
        return torch.ones_like(input_tensor, dtype=torch.bool)
    if mode == "decode" and not decoding:
        mask = torch.ones_like(input_tensor, dtype=torch.bool)
        if causal:
            mask[:, :, :, current_index : current_index + chunk_num] = torch.tril(
                torch.ones(
                    1, head_num, chunk_num, chunk_num, device=input_tensor.device
                )
            )
            mask[:, :, current_index + chunk_num :, :] = 0
            return torch.cat(
                [
                    torch.ones_like(input_tensor, dtype=torch.bool)[
                        :, :, 0 : current_index + 1
                    ],
                    torch.zeros_like(input_tensor, dtype=torch.bool)[
                        :, :, current_index + 1 :
                    ],
                ],
                dim=-1,
            )
        else:
            return mask
    input_tensor = input_tensor.to(float)

    if threshold is not None:
        total_sum = input_tensor.sum(dim=-1, keepdim=True)
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.to(float)
            required_sum = total_sum * threshold.unsqueeze(0).unsqueeze(-1).unsqueeze(
                -1
            ).expand((batch_size, head_num, chunk_num, 1)).to(input_tensor.device)
        else:
            required_sum = total_sum * threshold
        if causal:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            mask[:, :, :, 0] = 1
            mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            other_values = input_tensor.masked_fill(mask, 0)
            sorted_values, _ = torch.sort(other_values, dim=-1, descending=True)
            sorted_values = sorted_values.to(input_tensor.device)

            sorted_values = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    torch.where(mask, input_tensor, 0).sum(dim=-1, keepdim=True),
                    sorted_values[:, :, :, :-2],
                ],
                dim=-1,
            )

            _, index = torch.sort(
                torch.where(mask, 100000 * (1 + input_tensor), input_tensor),
                dim=-1,
                descending=True,
            )
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)

            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
            # assert(bool((torch.where(mask,input_tensor,0).sum(dim=-1,keepdim=True) >= required_sum*0.99).all()))
        else:
            mask = torch.zeros_like(input_tensor, dtype=torch.bool)
            sorted_values, index = torch.sort(input_tensor, dim=-1, descending=True)
            sorted_values = sorted_values.to(input_tensor.device)
            cumulative_sum_without_self = torch.cat(
                [
                    torch.zeros(
                        (batch_size, head_num, chunk_num, 1), device=input_tensor.device
                    ),
                    sorted_values[:, :, :, 0:-1],
                ],
                dim=-1,
            ).cumsum(dim=-1)
            index_mask = cumulative_sum_without_self < required_sum
            index = torch.where(index_mask, index, 0)
            mask = mask.view(batch_size, head_num * chunk_num, block_num)
            index = index.view(batch_size, head_num * chunk_num, block_num)
            mask[
                :,
                torch.arange(mask.shape[1], device=mask.device).unsqueeze(dim=-1),
                index,
            ] = True
            mask = mask.view(batch_size, head_num, chunk_num, block_num)
    else:
        raise NotImplementedError("block num chunk prefill not impleted")

    # breakpoint()
    try:
        if causal:
            assert (~mask[:, :, :, current_index + chunk_num :]).all()
    except:
        mask[:, :, :, current_index + chunk_num :] = False

    if causal:
        if decoding:
            assert mask[:, :, :, 0].all() and mask[:, :, :, -1].all()
        else:
            lambda_mask = torch.zeros_like(
                input_tensor, dtype=bool, device=input_tensor.device
            )
            lambda_mask[:, :, :, 0] = 1
            lambda_mask[:, :, :, current_index : current_index + chunk_num] = (
                torch.eye(chunk_num, device=lambda_mask.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(1, head_num, chunk_num, chunk_num)
            )
            assert torch.where(lambda_mask, mask, True).all()

    return mask


from typing import List, Optional, Tuple, Union

import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import CrossEntropyLoss
from transformers.cache_utils import Cache
from transformers.modeling_outputs import (
    CausalLMOutputWithPast,
)
from transformers.models.llama.modeling_llama import (
    _CONFIG_FOR_DOC,
    LLAMA_INPUTS_DOCSTRING,
)
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)


@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
@replace_return_docstrings(
    output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC
)
def llama_causal_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
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
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

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
        cache_position=cache_position,
    )

    hidden_states = outputs[0]
    if self.config.pretraining_tp > 1:
        lm_head_slices = self.lm_head.weight.split(
            self.vocab_size // self.config.pretraining_tp, dim=0
        )
        logits = [
            F.linear(hidden_states, lm_head_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        logits = torch.cat(logits, dim=-1)
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


# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
# Copyright 2024 ByteDance and/or its affiliates.
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


def llama_mlp_forward(self, x):
    if self.config.pretraining_tp > 1:
        slice = self.intermediate_size // self.config.pretraining_tp
        gate_proj_slices = self.gate_proj.weight.split(slice, dim=0)
        up_proj_slices = self.up_proj.weight.split(slice, dim=0)
        down_proj_slices = self.down_proj.weight.split(slice, dim=1)

        gate_proj = torch.cat(
            [
                F.linear(x, gate_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ],
            dim=-1,
        )
        up_proj = torch.cat(
            [F.linear(x, up_proj_slices[i]) for i in range(self.config.pretraining_tp)],
            dim=-1,
        )

        intermediate_states = (self.act_fn(gate_proj) * up_proj).split(slice, dim=2)
        down_proj = [
            F.linear(intermediate_states[i], down_proj_slices[i])
            for i in range(self.config.pretraining_tp)
        ]
        down_proj = sum(down_proj)
    else:

        def inner_mlp_forward(xx):
            return self.down_proj(self.act_fn(self.gate_proj(xx)) * self.up_proj(xx))

        batch_size, seq_len, hidden_dim = x.shape
        chunk_size = 32768
        down_proj = torch.empty_like(x)
        for b in range(batch_size):
            for i in range(0, seq_len, chunk_size):
                down_proj[b : b + 1, i : i + chunk_size] = inner_mlp_forward(
                    x[b : b + 1, i : i + chunk_size]
                )
    return down_proj


# duoattention
import transformers
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
import matplotlib.pyplot as plt
import numpy as np
import os
import json


def parse_device(device: str):
    if "," in device:
        return [int(d) for d in device.split(",")]
    elif device in ["auto", "cpu"]:
        return device
    return f"cuda:{device}"


def get_model(model_name):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    if hasattr(model.config, "sliding_window") and model.config.sliding_window is None:
        model.config.sliding_window = model.config.max_position_embeddings

    return model


from transformers import (
    PretrainedConfig,
)
from typing import Sequence
from tensor_parallel.config import Config
from tensor_parallel.communications import CollectiveOperation
from tensor_parallel.aux_actions import (
    gather_kv,
    select_kv_for_rank,
    split_inner_dim,
    split_num_heads,
)
from tensor_parallel.state_actions import (
    Split,
    SplitInChunks,
)
from functools import partial
import tensor_parallel as tp
from tensor_parallel.pretrained_model import find_predefined_tensor_parallel_config
from tensor_parallel.autoconfig import get_default_config
from tensor_parallel.state_actions import Split
import re


def to_device(
    model,
    device,
    enable_tp=False,
    enable_pp=False,
    reverse_device_map=True,
    even_split_layers=True,
):
    if enable_tp and isinstance(device, list):
        if len(device) == 1:
            return model.to(f"cuda:{device[0]}")
        device_ids = [f"cuda:{idx}" for idx in device]
        world_size = len(device_ids)
        tensor_parallel_config = find_predefined_tensor_parallel_config(
            model.config, device_ids
        )
        if tensor_parallel_config is None:
            tensor_parallel_config = get_default_config(model, device_ids)
        tensor_parallel_config.state_rules[re.compile(r".*full_attention_heads$")] = (
            Split(world_size=world_size, dim=0)
        )
        return tp.tensor_parallel(
            model,
            device_ids,
            tensor_parallel_config=tensor_parallel_config,
            sharded=True,
        )
    elif enable_pp and isinstance(device, list):
        no_split_module_classes = [
            "LlamaDecoderLayer",
        ]
        max_memory = {
            device_id: torch.cuda.get_device_properties(device_id).total_memory
            for device_id in device
        }
        print("Max Memory:", max_memory)
        max_memory = get_balanced_memory(
            model,
            max_memory,
            no_split_module_classes=no_split_module_classes,
        )
        device_map = infer_auto_device_map(
            model, max_memory, no_split_module_classes=no_split_module_classes
        )
        modules = list(device_map.keys())
        num_devices = len(device)
        device_map = {}
        current_device_idx = 0

        if even_split_layers:
            num_layers_per_device = model.config.num_hidden_layers / num_devices
            current_layer_idx = 0
            current_other_idx = 0
            for idx, module in enumerate(modules):
                if "layer" in module:
                    device_map[module] = device[current_device_idx]
                    current_layer_idx += 1
                    if current_layer_idx >= num_layers_per_device:
                        current_device_idx += 1
                        current_layer_idx = 0
                elif "lm_head" in module:
                    device_map[module] = device[-1]
                elif "norm" in module:
                    device_map[module] = device[-1]
                elif "embed" in module:
                    device_map[module] = device[0]
                else:
                    device_map[module] = device[current_other_idx]
                    current_other_idx += 1
                    continue
        else:
            num_modules_per_device = len(modules) / num_devices
            for idx, module in enumerate(modules):
                device_map[module] = device[current_device_idx]
                if (idx + 1) >= num_modules_per_device * (current_device_idx + 1):
                    current_device_idx += 1

        if reverse_device_map:
            device_map = {k: num_devices - v - 1 for k, v in device_map.items()}
        print("Device Map:", device_map)
        dispatch_model(model, device_map)
        return model
    else:
        return model.to(device)


def get_tokenizer(tokenizer_name):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_name, use_fast=False, trust_remote_code=True
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        else:
            tokenizer.pad_token_id = 0

    return tokenizer


def full_attention_heads_to_list(full_attention_heads):
    num_pruned_layers = len(full_attention_heads)
    num_heads = full_attention_heads[0].shape[0]
    for idx in range(num_pruned_layers):
        full_attention_heads[idx] = (
            full_attention_heads[idx].detach().float().cpu().tolist()
        )
    return full_attention_heads


def visualize_pruned_attention_heads(full_attention_heads):
    img = np.array(full_attention_heads)
    fig = plt.figure(figsize=(10, 10))
    plt.imshow(img, cmap="coolwarm", interpolation="nearest")
    plt.xlabel("Attention Heads")
    plt.ylabel("Layers")
    plt.colorbar(fraction=0.046, pad=0.04)
    # scale the color to 0-1
    plt.clim(0, 1)
    plt.tight_layout()
    plt.title("Ratio of Full Attention Computations")
    return fig


def load_attn_pattern_new(attn_load_dir, sink_size=None, recent_size=None):
    if attn_load_dir.endswith(".tsv"):
        path = attn_load_dir
    else:
        path = os.path.join(attn_load_dir, "full_attention_heads.tsv")
    full_attention_heads = np.loadtxt(
        path,
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    if sink_size is None:
        config = json.load(open(os.path.join(attn_load_dir, "config.json")))
        sink_size = config["sink_size"]
        recent_size = config["recent_size"]
    return full_attention_heads, sink_size, recent_size


def seed_everything(seed):
    import random, os
    import numpy as np

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def sparsify_attention_heads(full_attention_heads, threshold=None, sparsity=None):
    # add a very small random noise to full_attention_heads to break ties
    full_attention_heads += np.random.uniform(0, 1e-6, full_attention_heads.shape)

    if sparsity is not None:
        # ignore the threshold and use the sparsity
        # set the sparsity small values to 0 and others to 1
        threshold = np.quantile(full_attention_heads, sparsity)
    else:
        assert threshold is not None, "Either threshold or sparsity must be provided"

    if sparsity >= 1:
        # all heads are pruned
        threshold = 2
    if sparsity <= 0:
        # no heads are pruned
        threshold = -1

    full_attention_heads = (full_attention_heads >= threshold).astype(float)
    sparsity = 1 - np.mean(full_attention_heads)
    return full_attention_heads, sparsity


def save_full_attention_heads(full_attention_heads, output_filename):
    np.savetxt(
        output_filename,
        np.array(full_attention_heads),
        delimiter="\t",
    )


def enable_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
):
    print(
        f"Enabling DuoAttention evaluation using sink size {sink_size} and recent size {recent_size}"
    )
    if "llama" in model.config.model_type:
        from .duoattention import enable_llama_duo_attention_eval

        enable_llama_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    elif "qwen" in model.config.model_type:
        from .duoattention import enable_qwen_duo_attention_eval

        enable_qwen_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def get_full_attention_heads(model):
    if "llama" in model.config.model_type:
        from .duoattention import get_llama_full_attention_heads

        return get_llama_full_attention_heads(model)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def set_full_attention_heads(model, full_attention_heads):
    if "llama" in model.config.model_type:
        from .duoattention import set_llama_full_attention_heads

        model = set_llama_full_attention_heads(model, full_attention_heads)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")
    return model


def map_full_attention_heads(model, func):
    if "llama" in model.config.model_type:
        from .duoattention import map_llama_full_attention_heads

        return map_llama_full_attention_heads(model, func)
    else:
        raise ValueError(f"Model type {model.config.model_type} not supported")


def load_full_attention_heads(load_dir, filename="full_attention_heads.tsv"):
    full_attention_heads = np.loadtxt(
        os.path.join(load_dir, filename),
        dtype=float,
        delimiter="\t",
    )
    full_attention_heads = np.clip(full_attention_heads, 0, 1)
    full_attention_heads = torch.tensor(full_attention_heads, dtype=torch.float32)
    return full_attention_heads


try:
    import xformers.ops as xops
except ImportError:
    xops = None

try:
    from block_sparse_attn import block_streaming_attn_func
except ImportError:
    block_streaming_attn_func = None


def streaming_attn_xformers(
    query_states, key_states, value_states, streaming_causal_mask
):
    # query_states: [bsz, seq_len, num_heads, head_dim]
    # key_states: [bsz, seq_len, num_heads, head_dim]
    # value_states: [bsz, seq_len, num_heads, head_dim]
    # Return: [bsz, seq_len, num_heads, head_dim]

    bsz, seq_len, num_heads, head_dim = query_states.size()
    attn_bias = streaming_causal_mask[:, :, :seq_len, :seq_len].expand(
        bsz, num_heads, seq_len, seq_len
    )

    streaming_attn_output = xops.memory_efficient_attention(
        query_states,
        key_states,
        value_states,
        attn_bias=attn_bias,
        p=0.0,
    )

    return streaming_attn_output


from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    CausalLMOutputWithPast,
    Union,
    BaseModelOutputWithPast,
)
import types


class DuoAttentionStaticKVCache:
    def __init__(
        self,
        model,
        full_attention_heads,
        batch_size,
        max_size,
        sink_size,
        recent_size,
    ):
        self.batch_size = batch_size
        self.max_size = max_size
        self.sink_size = sink_size
        self.recent_size = recent_size

        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.num_layers = model.config.num_hidden_layers
        self.num_heads = model.config.num_attention_heads
        self.num_kv_heads = model.config.num_key_value_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = model.config.hidden_size // self.num_heads

        self.num_full_kv_head_list = [0] * self.num_layers
        self.num_streaming_kv_head_list = [0] * self.num_layers

        self.kv_seq_len_list = [0] * self.num_layers
        self.streaming_kv_seq_len_list = [0] * self.num_layers

        self.streaming_key_states_list = []
        self.streaming_value_states_list = []
        self.full_key_states_list = []
        self.full_value_states_list = []

        for idx, layer_full_attention_heads in enumerate(full_attention_heads):
            layer_full_attention_heads = torch.tensor(layer_full_attention_heads) > 0.5
            num_full_kv_head = layer_full_attention_heads.sum().item()
            num_streaming_kv_head = self.num_kv_heads - num_full_kv_head

            self.num_full_kv_head_list[idx] = num_full_kv_head
            self.num_streaming_kv_head_list[idx] = num_streaming_kv_head

            streaming_key_states = torch.zeros(
                self.batch_size,
                self.sink_size + self.recent_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            streaming_value_states = torch.zeros(
                self.batch_size,
                self.sink_size + self.recent_size,
                num_streaming_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            full_key_states = torch.zeros(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            full_value_states = torch.zeros(
                self.batch_size,
                self.max_size,
                num_full_kv_head,
                self.head_dim,
                device=self.device,
                dtype=self.dtype,
            )

            self.streaming_key_states_list.append(streaming_key_states)
            self.streaming_value_states_list.append(streaming_value_states)
            self.full_key_states_list.append(full_key_states)
            self.full_value_states_list.append(full_value_states)

    @property
    def streaming_kv_seq_len(self):
        return self.streaming_kv_seq_len_list[-1]

    @property
    def kv_seq_len(self):
        return self.kv_seq_len_list[-1]

    def put_full_kv(self, layer_idx, full_key_states, full_value_states):
        incoming_kv_seq_len = full_key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )

        self.full_key_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_key_states)
        self.full_value_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_value_states)

        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        return self.get_full_kv(layer_idx)

    def compress_and_replace_streaming_kv(
        self, layer_idx, streaming_key_states, streaming_value_states
    ):
        incoming_kv_seq_len = streaming_key_states.shape[1]
        if incoming_kv_seq_len <= self.sink_size + self.recent_size:
            self.streaming_key_states_list[layer_idx][
                :,
                :incoming_kv_seq_len,
            ].copy_(streaming_key_states)
            self.streaming_value_states_list[layer_idx][
                :,
                :incoming_kv_seq_len,
            ].copy_(streaming_value_states)

            self.streaming_kv_seq_len_list[layer_idx] = incoming_kv_seq_len
        else:
            sink_key_states = streaming_key_states[:, : self.sink_size]
            recent_key_states = streaming_key_states[
                :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
            ]
            self.streaming_key_states_list[layer_idx][:, : self.sink_size].copy_(
                sink_key_states
            )
            self.streaming_key_states_list[layer_idx][
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_key_states)

            sink_value_states = streaming_value_states[:, : self.sink_size]
            recent_value_states = streaming_value_states[
                :, incoming_kv_seq_len - self.recent_size : incoming_kv_seq_len
            ]
            self.streaming_value_states_list[layer_idx][:, : self.sink_size].copy_(
                sink_value_states
            )
            self.streaming_value_states_list[layer_idx][
                :, self.sink_size : self.sink_size + self.recent_size
            ].copy_(recent_value_states)

            self.streaming_kv_seq_len_list[layer_idx] = (
                self.recent_size + self.sink_size
            )

    def put(self, layer_idx, key_states, value_states):
        incoming_kv_seq_len = key_states.shape[1]
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        if incoming_kv_seq_len + kv_seq_len > self.max_size:
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with max size {self.max_size}, current size: {kv_seq_len}."
            )
        if (
            incoming_kv_seq_len + streaming_kv_seq_len
            > self.sink_size + self.recent_size + self.prefilling_chunk_size
        ):
            raise ValueError(
                f"Trying to put {incoming_kv_seq_len} KVs into a cache with sink size {self.sink_size}, recent size {self.recent_size}, and prefilling chunk size {self.prefilling_chunk_size}, current size: {streaming_kv_seq_len}."
            )

        (
            full_key_states,
            full_value_states,
            streaming_key_states,
            streaming_value_states,
        ) = self.split_kv(layer_idx, key_states, value_states)

        self.full_key_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_key_states)
        self.full_value_states_list[layer_idx][
            :, kv_seq_len : kv_seq_len + incoming_kv_seq_len
        ].copy_(full_value_states)

        self.streaming_key_states_list[layer_idx][
            :,
            streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len,
        ].copy_(streaming_key_states)
        self.streaming_value_states_list[layer_idx][
            :,
            streaming_kv_seq_len : streaming_kv_seq_len + incoming_kv_seq_len,
        ].copy_(streaming_value_states)

        self.update_seq_len(layer_idx, incoming_kv_seq_len)

        return self.get(layer_idx)

    def update_seq_len(self, layer_idx, incoming_kv_seq_len):
        self.kv_seq_len_list[layer_idx] += incoming_kv_seq_len
        self.streaming_kv_seq_len_list[layer_idx] += incoming_kv_seq_len

    def get_full_kv(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        return (
            self.full_key_states_list[layer_idx][:, :kv_seq_len],
            self.full_value_states_list[layer_idx][:, :kv_seq_len],
        )

    def get_streaming_kv(self, layer_idx):
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            self.streaming_key_states_list[layer_idx][:, :streaming_kv_seq_len],
            self.streaming_value_states_list[layer_idx][:, :streaming_kv_seq_len],
        )

    def get(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            self.full_key_states_list[layer_idx][:, :kv_seq_len],
            self.full_value_states_list[layer_idx][:, :kv_seq_len],
            self.streaming_key_states_list[layer_idx][:, :streaming_kv_seq_len],
            self.streaming_value_states_list[layer_idx][:, :streaming_kv_seq_len],
        )

    def get_unsliced(self, layer_idx):
        kv_seq_len = self.kv_seq_len_list[layer_idx]
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        return (
            kv_seq_len,
            self.full_key_states_list[layer_idx],
            self.full_value_states_list[layer_idx],
            streaming_kv_seq_len,
            self.streaming_key_states_list[layer_idx],
            self.streaming_value_states_list[layer_idx],
        )

    def split_kv(self, layer_idx, key_states, value_states):
        num_full_kv_head = self.num_full_kv_head_list[layer_idx]
        full_key_states = key_states[:, :, :num_full_kv_head, :]
        full_value_states = value_states[:, :, :num_full_kv_head, :]
        streaming_key_states = key_states[:, :, num_full_kv_head:, :]
        streaming_value_states = value_states[:, :, num_full_kv_head:, :]
        return (
            full_key_states,
            full_value_states,
            streaming_key_states,
            streaming_value_states,
        )

    def compress(self, layer_idx):
        streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
        if streaming_kv_seq_len <= self.recent_size + self.sink_size:
            return
        recent_key_states = self.streaming_key_states_list[layer_idx][
            :, streaming_kv_seq_len - self.recent_size : streaming_kv_seq_len
        ].clone()
        self.streaming_key_states_list[layer_idx][
            :, self.sink_size : self.sink_size + self.recent_size
        ].copy_(recent_key_states)

        recent_value_states = self.streaming_value_states_list[layer_idx][
            :, streaming_kv_seq_len - self.recent_size : streaming_kv_seq_len
        ].clone()
        self.streaming_value_states_list[layer_idx][
            :, self.sink_size : self.sink_size + self.recent_size
        ].copy_(recent_value_states)

        self.streaming_kv_seq_len_list[layer_idx] = self.recent_size + self.sink_size

    def clear(self):
        for layer_idx in range(self.num_layers):
            self.kv_seq_len_list[layer_idx] = 0
            self.streaming_kv_seq_len_list[layer_idx] = 0

    def evict_last(self, num_tokens):
        for layer_idx in range(self.num_layers):
            kv_seq_len = self.kv_seq_len_list[layer_idx]
            streaming_kv_seq_len = self.streaming_kv_seq_len_list[layer_idx]
            self.kv_seq_len_list[layer_idx] = max(0, kv_seq_len - num_tokens)
            self.streaming_kv_seq_len_list[layer_idx] = max(
                0, streaming_kv_seq_len - num_tokens
            )

    @property
    def memory_usage(self):
        memory_usage = 0
        for layer_idx in range(self.num_layers):
            memory_usage += self.full_key_states_list[layer_idx].element_size() * (
                self.full_key_states_list[layer_idx].numel()
            )
            memory_usage += self.full_value_states_list[layer_idx].element_size() * (
                self.full_value_states_list[layer_idx].numel()
            )
            memory_usage += self.streaming_key_states_list[layer_idx].element_size() * (
                self.streaming_key_states_list[layer_idx].numel()
            )
            memory_usage += self.streaming_value_states_list[
                layer_idx
            ].element_size() * (self.streaming_value_states_list[layer_idx].numel())
        return memory_usage


def duo_attn_static_kv_cache_llama_for_causal_lm_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[DuoAttentionStaticKVCache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
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
        logits = logits.float()
    else:
        logits = self.lm_head(hidden_states[:, -1:, :])

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


def duo_attn_static_kv_cache_llama_model_forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[DuoAttentionStaticKVCache] = None,
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
        past_key_values_length = past_key_values.kv_seq_len
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

    for idx, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        layer_outputs = decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=past_key_values,
            layer_idx=idx,
            output_attentions=output_attentions,
            use_cache=use_cache,
            padding_mask=padding_mask,
        )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def duo_attn_static_kv_cache_llama_decoder_layer_forward(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    kv_cache: Optional[DuoAttentionStaticKVCache] = None,
    layer_idx: int = None,
    output_attentions: Optional[bool] = False,
    use_cache: Optional[bool] = False,
    padding_mask: Optional[torch.LongTensor] = None,
) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
    residual = hidden_states

    hidden_states = self.input_layernorm(hidden_states)

    # Self Attention
    hidden_states, self_attn_weights = self.self_attn(
        hidden_states=hidden_states,
        attention_mask=attention_mask,
        position_ids=position_ids,
        kv_cache=kv_cache,
        layer_idx=layer_idx,
        output_attentions=output_attentions,
        use_cache=use_cache,
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

    return outputs


def enable_duo_attention_static_kv_cache_for_llama(model: LlamaForCausalLM):
    model.model._prepare_decoder_attention_mask = lambda *args, **kwargs: None

    model.model.forward = types.MethodType(
        duo_attn_static_kv_cache_llama_model_forward, model.model
    )
    for idx in range(len(model.model.layers)):
        model.model.layers[idx].forward = types.MethodType(
            duo_attn_static_kv_cache_llama_decoder_layer_forward,
            model.model.layers[idx],
        )
    model.forward = types.MethodType(
        duo_attn_static_kv_cache_llama_for_causal_lm_forward, model
    )


from transformers.models.llama.modeling_llama import LlamaRMSNorm
import flashinfer
from typing import Optional


def flashinfer_rmsnorm_forward(self, hidden_states):
    bsz, seq_len, hidden_size = hidden_states.size()
    hidden_states = flashinfer.norm.rmsnorm(
        hidden_states.view(bsz * seq_len, hidden_size),
        self.weight,
        eps=self.variance_epsilon,
    )
    return hidden_states.view(bsz, seq_len, hidden_size)


def enable_flashinfer_rmsnorm(model):
    print("Replacing RMSNorm with Flashinfer's RMSNorm")
    for name, module in model.named_modules():
        if isinstance(module, LlamaRMSNorm):
            module.forward = types.MethodType(flashinfer_rmsnorm_forward, module)
    return model


def apply_rope_inplace(
    q: torch.Tensor,
    k: torch.Tensor,
    offsets: torch.Tensor,
    rope_scale: float,
    rope_theta: float,
    indptr: Optional[torch.Tensor] = None,
):
    bsz, seq_len, num_heads, head_dim = q.size()
    _, _, num_kv_heads, _ = k.size()
    nnz = bsz * seq_len
    q = q.view(nnz, num_heads, head_dim)
    k = k.view(nnz, num_kv_heads, head_dim)
    if indptr is None:
        indptr = torch.tensor(
            [i * seq_len for i in range(bsz + 1)], dtype=torch.int32, device=q.device
        )
    if offsets.numel() == 1:
        offsets = offsets.expand(bsz).contiguous()
    flashinfer.rope.apply_rope_inplace(
        q,
        k,
        indptr,
        offsets,
        interleave=False,
        rope_scale=rope_scale,
        rope_theta=rope_theta,
    )
    q = q.view(bsz, seq_len, num_heads, head_dim)
    k = k.view(bsz, seq_len, num_kv_heads, head_dim)
    return q, k


@torch.no_grad()
def reorder_linear_weights(
    linear_module: torch.nn.Linear,
    full_attention_heads: torch.Tensor,
    repeat_num,
    reorder_channel,
):
    assert reorder_channel in ["in", "out"]
    full_attention_heads = torch.repeat_interleave(
        full_attention_heads, repeats=repeat_num
    ).to(linear_module.weight.device)
    full_attn_mask = full_attention_heads > 0.5
    if reorder_channel == "in":
        weight1 = linear_module.weight.data[:, full_attn_mask]
        weight2 = linear_module.weight.data[:, ~full_attn_mask]
        reordered_weight = torch.cat([weight1, weight2], dim=1)
    else:
        weight1 = linear_module.weight.data[full_attn_mask, :]
        weight2 = linear_module.weight.data[~full_attn_mask, :]
        reordered_weight = torch.cat([weight1, weight2], dim=0)
    linear_module.weight.data = reordered_weight
    # for linear modules with bias
    if linear_module.bias is not None:
        bias1 = linear_module.bias.data[full_attn_mask]
        bias2 = linear_module.bias.data[~full_attn_mask]
        reordered_bias = torch.cat([bias1, bias2], dim=0)
        linear_module.bias.data = reordered_bias

    return linear_module


@torch.no_grad()
def reorder_full_attn_heads(
    full_attention_heads: torch.Tensor,
):
    full_attn_mask = full_attention_heads > 0.5
    num_full_attn_heads = full_attn_mask.sum().item()
    full_attention_heads[:num_full_attn_heads] = 1
    full_attention_heads[num_full_attn_heads:] = 0
    return full_attention_heads
