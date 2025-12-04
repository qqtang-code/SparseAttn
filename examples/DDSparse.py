import json
import torch
from transformers import AutoTokenizer, AutoConfig
from transformers import AutoModelForCausalLM



def load_sparse_model(model_path):
    config_path = f"{model_path}/config.json"
    with open(config_path, "r") as f:
        config_data = json.load(f)

    arch = config_data.get("architectures", [])
    if not arch:
        raise ValueError("No architecture found in config.json")

    arch_name = arch[0]
    print(f"Detected architecture: {arch_name}")

    if "PawLlama" in arch_name:
        from sparseattn.training.eval.modeling_flash_llama import (
            PawLlamaForCausalLM,
            PawLlamaConfig,
        )

        AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
        model_cls = PawLlamaForCausalLM
    elif "PawQwen" in arch_name:
        from sparseattn.training.eval.modeling_flash_qwen import (
            PawQwen3ForCausalLM,
            PawQwen3Config,
        )

        AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
        model_cls = PawQwen3ForCausalLM
    else:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model


def main():
    model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/masksonly_Qwen3-4B_bsz16_steps125_lr1e-5_warmup0.1_sp0.3_cw2048_mlr1.0_rlr1.0sft3_pretrain_64k_xattn_mlp_linear_first_token_10reg_64k_12.4_wfrozen/checkpoint-60"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    from datasets import Dataset as HFDataset
    from sparseattn.training.dataset_batch import ParquetDataset

    dummy_data = {
        "context": [""],
        "question": [""],
        "answer": [""],
        "metadata": [""],
    }
    dummy_hf_dataset = HFDataset.from_dict(dummy_data)

    dataset = ParquetDataset(
        raw_dataset=dummy_hf_dataset,
        tokenizer=tokenizer,
        data_args=None,
        max_seq_len=None,
        is_training=False,
    )

    model = load_sparse_model(model_path)

    config = model.config

    model.eval()

    prompt = "Hello, how are you?"

    fake_item = {
        "context": "",                     # no context
        "question": prompt,                # your prompt
        "answer": "",                      # empty for inference
        "metadata": {"task": "Single QA", "flag": "0"},
    }

    input_ids, labels, attention_mask, segment_ids, range_ids, class_id = dataset._build_sft_input_and_labels(
        fake_item, tokenizer, max_seq_len=32768
    )

    actual_len = attention_mask.sum().item()  # number of non-pad tokens
    input_ids = input_ids[:actual_len].unsqueeze(0).to(model.device)          # [1, L]
    attention_mask = attention_mask[:actual_len].unsqueeze(0).to(model.device)  # [1, L]
    segment_ids = segment_ids[:actual_len].unsqueeze(0).to(model.device)       # [1, L]
    range_ids = range_ids.unsqueeze(0).to(model.device)                        # [1, 8] — no seq dim, keep as-is
    task_ids = class_id.unsqueeze(0).to(model.device)                          # [1]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "segment_ids": segment_ids,
        "range_ids": range_ids,
        "task_ids": task_ids,
    }

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=50,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][actual_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("Response:", response)


if __name__ == "__main__":
    main()
