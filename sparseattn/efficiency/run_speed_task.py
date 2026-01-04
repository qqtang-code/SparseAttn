import json
import torch
import time
import os
import gc
import datetime
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import logging
logging.set_verbosity_error()  # 只显示错误，不显示警告和通知

# -----------------------------------------------------------------------------
# 0. 任务配置 (LongBench 任务分类映射)
# -----------------------------------------------------------------------------
TASK_GROUPS = {
    "Single-Document QA": [
        "qasper_e.jsonl",
        "multifieldqa_en_e.jsonl"
    ],
    "Multi-Document QA": [
        "hotpotqa_e.jsonl",
        "2wikimqa_e.jsonl"
    ],
    "Summarization": [
        "gov_report_e.jsonl",
        "multi_news_e.jsonl"
    ],
    "Few-shot Learning": [
        "trec_e.jsonl",
        "triviaqa_e.jsonl",
        "samsum_e.jsonl" 
    ],
    "Synthetic Tasks": [
        "passage_count_e.jsonl",
        "passage_retrieval_en_e.jsonl"
    ],
    "Code": [
        "repobench-p_e.jsonl",
        "lcc_e.jsonl"
    ]
}

TASK_GROUPS = {
    "Multi-Document QA": [
        "hotpotqa_e.jsonl",
        "2wikimqa_e.jsonl"
    ],
}


# -----------------------------------------------------------------------------
# 1. 统一模型加载器 (保持不变)
# -----------------------------------------------------------------------------
def load_model(model_path, is_sparse):
    print(f"\n📥 [System] Loading model from: {model_path} ...")
    config_path = f"{model_path}/config.json"
    if not os.path.exists(config_path):
        raise ValueError(f"❌ Config not found at {config_path}")
        
    with open(config_path, "r") as f:
        config_data = json.load(f)

    archs = config_data.get("architectures", [])
    arch_name = archs[0] if archs else "Unknown"

    if is_sparse:
        if "PawLlama" in arch_name:
            from sparseattn.training.eval.modeling_flash_llama import PawLlamaForCausalLM, PawLlamaConfig
            AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
            model_cls = PawLlamaForCausalLM
        elif "PawQwen" in arch_name:
            from sparseattn.efficiency.model.modeling_flash_qwen_copy import PawQwen3ForCausalLM, PawQwen3Config
            AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
            model_cls = PawQwen3ForCausalLM
    else:
        if "PawLlama" in arch_name:
            from sparseattn.training.eval.modeling_flash_llama import PawLlamaForCausalLM, PawLlamaConfig
            AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
            model_cls = PawLlamaForCausalLM
        elif "PawQwen" in arch_name:
            from sparseattn.efficiency.model.modeling_flash_qwen_full import PawQwen3ForCausalLM, PawQwen3Config
            AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
            model_cls = PawQwen3ForCausalLM

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return model, is_sparse

# -----------------------------------------------------------------------------
# 2. 核心评测函数 (保持不变)
# -----------------------------------------------------------------------------
def evaluate_efficiency(model, input_ids, gen_len=10, is_sparse=False):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # --- A. Prefill ---
    torch.cuda.synchronize()
    start_event.record()
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
    end_event.record()
    torch.cuda.synchronize()
    prefill_time_ms = start_event.elapsed_time(end_event)
    
    past_key_values = outputs.past_key_values
    current_sparsity = 0.0
    if is_sparse:
        try:
            sp = getattr(model, "prefill_sparsity", None)
            if isinstance(sp, torch.Tensor):
                current_sparsity = sp.item()
        except: pass

    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
    
    # --- B. Decode ---
    torch.cuda.synchronize()
    start_event.record()
    with torch.inference_mode():
        for _ in range(gen_len):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
    end_event.record()
    torch.cuda.synchronize()
    decode_time_ms = start_event.elapsed_time(end_event)
    
    return {
        "prefill_ms": prefill_time_ms,
        "decode_ms_total": decode_time_ms,
        "decode_ms_per_token": decode_time_ms / gen_len,
        "sparsity": current_sparsity
    }

# -----------------------------------------------------------------------------
# 3. 批量测试执行器
# -----------------------------------------------------------------------------
def run_benchmark_suite(model_path, samples, tokenizer, gen_len=10, max_len=4096, is_sparse=False):
    model, is_sparse = load_model(model_path, is_sparse)
    results = []
    
    print(f"🏃 [Run] Processing {len(samples)} samples (Max Len: {max_len})...")
    
    for i, item in enumerate(samples):
        input_text = item["input_text"]
        # breakpoint()
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        seq_len = input_ids.shape[-1]
        
        # === 截断逻辑 ===
        if seq_len > max_len:
            half_len = max_len // 2
            head_part = input_ids[:, :half_len]
            tail_part = input_ids[:, -(max_len - half_len):]
            input_ids = torch.cat([head_part, tail_part], dim=1)
            seq_len = max_len
            
        res = evaluate_efficiency(model, input_ids, gen_len=gen_len, is_sparse=is_sparse)
        
        if i == 0: continue # Skip warmup
        
        res["seq_len"] = seq_len
        results.append(res)
        print(f"  Sample {i} : 📏 Len {seq_len} | ⚡ Prefill {res['prefill_ms']:.1f}ms | ⏩ Decode {res['decode_ms_per_token']:.2f}ms/tok")
        del input_ids
    
    del model
    torch.cuda.empty_cache()
    gc.collect()
    return results

# -----------------------------------------------------------------------------
# 4. 主程序
# -----------------------------------------------------------------------------
def main():
    # ================= 配置区域 =================
    sparse_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.1router4steps266_full_streaming_64k_qwen3-4b_wfrozen/checkpoint-230"
    full_model_path   = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.1router4steps266_full_streaming_64k_qwen3-4b_wfrozen/checkpoint-200"
    base_data_dir     = "/data2/public_data/sort_longbench/"

    
    num_samples = 5       
    gen_len = 1           
    max_len = 128 * 1024  
    # ===========================================

    print(f"📂 [Init] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(sparse_model_path, trust_remote_code=True)
    
    all_rows = []        # 存储所有具体数据的行
    global_metrics = []  # 存储所有有效样本的指标，用于计算 Overall
    
    print("\n" + "="*60)
    print(f"🚀 STARTING LONGBENCH EFFICIENCY TEST (MaxLen: {max_len})")
    print("="*60)

    # === 遍历每一个任务大类 ===
    for category, file_list in TASK_GROUPS.items():
        print(f"\n🧩 Category: {category}")
        
        category_metrics = [] # 用于计算当前 Category 的 Average

        # 遍历数据集
        for filename in file_list:
            data_path = os.path.join(base_data_dir, filename)
            dataset_name = filename.replace(".jsonl", "").replace("_e", "") 
            
            if not os.path.exists(data_path):
                print(f"⚠️  Skipping {filename}: File not found.")
                continue
            
            # 读取数据
            raw_samples = []
            try:
                with open(data_path, 'r') as f:
                    for i, line in enumerate(f):
                        if len(raw_samples) >= num_samples + 1: break 
                        try: raw_samples.append(json.loads(line))
                        except: pass
            except: continue

            if len(raw_samples) < 2: continue

            print(f"   📄 Processing {dataset_name}...")

            # 运行测试
            full_res = run_benchmark_suite(full_model_path, raw_samples, tokenizer, gen_len, max_len, False)
            sparse_res = run_benchmark_suite(sparse_model_path, raw_samples, tokenizer, gen_len, max_len, True)

            if not full_res or not sparse_res: continue

            # 计算单个 Dataset 的平均值
            valid_count = min(len(full_res), len(sparse_res))
            
            # 临时累加器
            acc = {
                "s_decode": 0.0, "f_prefill": 0.0, "f_decode": 0.0,
                "speedup_p": 0.0, "speedup_d": 0.0, "sparsity": 0.0
            }

            for k in range(valid_count):
                f_item = full_res[k]
                s_item = sparse_res[k]
                
                sp_p = f_item['prefill_ms'] / s_item['prefill_ms'] if s_item['prefill_ms'] > 0 else 0
                sp_d = f_item['decode_ms_per_token'] / s_item['decode_ms_per_token'] if s_item['decode_ms_per_token'] > 0 else 0

                acc["s_decode"] += s_item['decode_ms_per_token']
                acc["f_prefill"] += f_item['prefill_ms']
                acc["f_decode"] += f_item['decode_ms_per_token']
                acc["speedup_p"] += sp_p
                acc["speedup_d"] += sp_d
                acc["sparsity"] += s_item['sparsity']

            # Dataset 平均值
            row_data = {
                "LongBench": category,  # 第一列：大类名
                "subtask": dataset_name, # 第二列：数据集名
                "Sparse Decode (ms)": acc["s_decode"] / valid_count,
                "Full Prefill (ms)": acc["f_prefill"] / valid_count,
                "Full Decode (ms)": acc["f_decode"] / valid_count,
                "Speedup Prefill": acc["speedup_p"] / valid_count,
                "Speedup Decode": acc["speedup_d"] / valid_count,
                "Sparsity": acc["sparsity"] / valid_count
            }
            
            all_rows.append(row_data)       # 添加到总表
            category_metrics.append(row_data) # 添加到大类统计
            global_metrics.append(row_data)   # 添加到全局统计
            
            print(f"      ✅ {dataset_name}: Speedup(P) {row_data['Speedup Prefill']:.2f}x | Sparsity {row_data['Sparsity']:.4f}")

        # === 计算当前 Category 的 Average ===
        if category_metrics:
            df_cat = pd.DataFrame(category_metrics)
            avg_row = {
                "LongBench": category,  # 保持大类名，方便查看
                "subtask": "Average",   # 第二列显示 Average
                "Sparse Decode (ms)": df_cat["Sparse Decode (ms)"].mean(),
                "Full Prefill (ms)": df_cat["Full Prefill (ms)"].mean(),
                "Full Decode (ms)": df_cat["Full Decode (ms)"].mean(),
                "Speedup Prefill": df_cat["Speedup Prefill"].mean(),
                "Speedup Decode": df_cat["Speedup Decode"].mean(),
                "Sparsity": df_cat["Sparsity"].mean()
            }
            all_rows.append(avg_row) # 将 Average 行加入总表
            print(f"   ⭐️ {category} Average Calculated.")

    # ================= 结果汇总与保存 =================
    if all_rows:
        # 1. 计算 Global Overall (最顶部的总平均)
        if global_metrics:
            df_global = pd.DataFrame(global_metrics)
            overall_row = {
                "LongBench": "Overall", # 或者留空 ""
                "subtask": "Overall",   # 或者 "Average"
                "Sparse Decode (ms)": df_global["Sparse Decode (ms)"].mean(),
                "Full Prefill (ms)": df_global["Full Prefill (ms)"].mean(),
                "Full Decode (ms)": df_global["Full Decode (ms)"].mean(),
                "Speedup Prefill": df_global["Speedup Prefill"].mean(),
                "Speedup Decode": df_global["Speedup Decode"].mean(),
                "Sparsity": df_global["Sparsity"].mean()
            }
            # 将 Overall 插到最前面
            all_rows.insert(0, overall_row)

        df = pd.DataFrame(all_rows)
        
        # 2. 强制指定列顺序 (完全对齐截图)
        cols_order = [
            "LongBench", 
            "subtask", 
            "Sparse Decode (ms)", 
            "Full Prefill (ms)", 
            "Full Decode (ms)", 
            "Speedup Prefill", 
            "Speedup Decode", 
            "Sparsity"
        ]
        df = df[cols_order]

        ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        file_name = f"/data1/lcm_lab/qqt/SparseAttn/sparseattn/efficiency/results/benchmark_task_{ts}.xlsx"
        
        os.makedirs(os.path.dirname(file_name), exist_ok=True)
        
        print(f"\n💾 Saving formatted report to {file_name}...")
        df.to_excel(file_name, index=False)
        print("✅ All Done.")
    else:
        print("⚠️ No results to save.")

if __name__ == "__main__":
    main()