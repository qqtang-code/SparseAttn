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

def get_task(metadata_str):
    try:
        if isinstance(metadata_str, str):
            meta_dict = ast.literal_eval(metadata_str)
        elif isinstance(metadata_str, dict):
            meta_dict = metadata_str
        else:
            return None
        return meta_dict.get('task')
    except Exception:
        return None

def main():
    model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/qwen3-4b-lm-reg-softmaxmlp"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    from datasets import Dataset as HFDataset
    from sparseattn.training.dataset_batch import ParquetDataset

    import pandas as pd
    import ast
    import numpy as np

    file_path = "/data2/public_data/qwen_mix_sft_32K/all.parquet"

    df = pd.read_parquet(file_path)

    target_task = "Summarization"# Single QA, Summarization

    df["task"] = df["metadata"].apply(get_task)

    df_target = df[df["task"] == target_task]

    row = df_target.iloc[0]
    prompt = row['question']
    context = row['context']
    answer = row['answer']
    metadata = row['metadata']

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

    fake_item = {
        "context": context,                  
        "question": prompt,               
        "answer": answer,           
        "metadata": metadata,# Single QA, Summarization
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
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][actual_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("Response:", response)


if __name__ == "__main__":
    main()
