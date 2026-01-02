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
    model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.1router4steps266_full_streaming_64k_qwen3-4b_wfrozen/checkpoint-230"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_sparse_model(model_path)
    model.eval()
    
    sparsity = []
    
    longbench_prediction = "/data1/lcm_lab/sora/loomeval/benchmarks/General/RULER/data/cwe_8192.jsonl"
    
    # 读取jsonl文件
    with open(longbench_prediction, 'r') as f:
        data = [json.loads(line) for line in f]
    for i in range(len(data)):
        input_ids = tokenizer.encode(data[i]["input"], return_tensors="pt").to(model.device)
        attention_mask = torch.ones_like(input_ids).to(model.device)
        actual_len = input_ids.shape[-1]

    model_inputs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
    }

    with torch.no_grad():
        outputs = model.generate(
            **model_inputs,
            max_new_tokens=10,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    print(f"sparsity:{model.prefill_sparsity}")
    sparsity.append(model.prefill_sparsity)

    generated_ids = outputs[0][actual_len:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print("Response:", response)
    print(f"Average Sparsity:{sum(sparsity)/len(sparsity)}")


if __name__ == "__main__":
    main()
