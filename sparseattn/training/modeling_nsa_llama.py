from transformers.models.llama import modeling_llama
# modeling_nsa_llama.py

import math
import torch
import torch.nn as nn
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel, LlamaModel, LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC, KwargsForCausalLM
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.generation import GenerationMixin
from typing import Optional, Tuple, Union

from functools import partial
from typing import Callable, Optional, Tuple, Union

import torch
from torch import nn

from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.processing_utils import Unpack
from transformers.utils import (
    add_start_docstrings_to_model_forward,
    can_return_tuple,
    replace_return_docstrings,
)
from transformers.utils.deprecation import deprecate_kwarg


from .block_sparse_attention_triton.native_sparse_attention.module.llama_nsa import LlamaNSA


class NSALlamaForCausalLM(LlamaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        config._attn_implementation = "flash_attention_2"
        config.compress_type = "linear"#"avgpool","weightedpool"
        config.kernel_size = 32
        config.kernel_stride = 16
        config.block_size = 64
        config.topk = 8
        config.init_blocks = 1
        config.local_blocks = 2
        config.window_size = 500

        # --- CRITICAL: Replace attention layers with LlamaNSA ---
        for layer_idx, layer in enumerate(self.model.layers):
             # Create new NSA layer with same config
            new_attn = LlamaNSA(
                config=config,
                layer_idx=layer_idx # Pass layer index
            )
             # Assign the new NSA layer
            layer.self_attn = new_attn

        # Initialize weights and apply final processing
        self.post_init()

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

    @can_return_tuple
    @deprecate_kwarg("num_logits_to_keep", version="4.50", new_name="logits_to_keep")
    @add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

            logits_to_keep (`int` or `torch.Tensor`, *optional*):
                If an `int`, compute logits for the last `logits_to_keep` tokens. If `0`, calculate logits for all
                `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
                token can save memory, which becomes pretty significant for long sequences or large vocabulary size.
                If a `torch.Tensor`, must be 1D corresponding to the indices to keep in the sequence length dimension.
                This is useful when using packed tensor format (single dimension for batch and sequence length).

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
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )



if __name__ == "__main__":

    from transformers.models.llama import LlamaForCausalLM
    from transformers import AutoTokenizer
    
    #model_path = "/root/.cache/modelscope/hub/models/LLM-Research/Llama-3.2-1B-Instruct" # e.g., "/storage/hf_models/Llama-3.2-3B-Instruct"
    model_path = "/data1/hf_model/Meta-Llama-3.1-8B-Instruct"
    #save_directory = "/data1/lcm_lab/yy/checkpoint/NSALlama-3.2-1B-Instruct" # Where you want to save the final model
    save_directory = "/data1/lcm_lab/yy/checkpoint/Meta-NSALlama-3.1-8B-Instruct"


    print(f"Loading base model from {model_path}...")
    # Load the base Llama model
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        from_tf=bool(".ckpt" in model_path),
        torch_dtype=torch.bfloat16,
        # device_map="auto", # 如果需要，可以指定设备
    )
    print("Base model loaded.")

    config = base_model.config
    config._attn_implementation = "flash_attention_2"
    config.compress_type = "linear"#"avgpool","weightedpool"
    config.kernel_size = 32
    config.kernel_stride = 16
    config.block_size = 64
    config.topk = 8
    config.init_blocks = 1
    config.local_blocks = 2
    config.window_size = 500

    print("Creating NSALlamaForCausalLM with base model config...")
    # Create the custom model instance
    nsa_model = NSALlamaForCausalLM(config=config)

    print("Loading base model state dict into NSALlamaForCausalLM...")
    # Load state dict from the base model, ignoring keys that don't match
    # This initializes embed_tokens, lm_head, and the *base* parts of each layer.
    # The LlamaNSA layers will have their new parameters initialized according to their __init__.
    # If LlamaNSA's q_proj, k_proj, v_proj, o_proj have the same structure as original LlamaAttention,
    # their weights will be loaded. Otherwise, they will be initialized randomly/default.
    nsa_model.load_state_dict(base_model.state_dict(), strict=False)
    print("State dict loaded.")

    # --- CRITICAL: Update the config to indicate this is a custom architecture ---
    nsa_model.config.architectures = ["NSALlamaForCausalLM"]

    print(f"Saving initialized NSALlamaForCausalLM to {save_directory}...")
    # Save the model and updated config
    nsa_model.save_pretrained(save_directory)
    # Also save the tokenizer if needed
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

    # --- Optional: Verify the number of parameters ---
    print(f"Total parameters in NSALlamaForCausalLM: {nsa_model.num_parameters()}")
    print(f"Total parameters in base LlamaForCausalLM: {base_model.num_parameters()}")
    # The number might be different if LlamaNSA adds new parameters.


