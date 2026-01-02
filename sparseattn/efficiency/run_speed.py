import json
import torch
import time
import os
import gc
from transformers import AutoTokenizer, AutoModelForCausalLM

# -----------------------------------------------------------------------------
# 1. 统一模型加载器
# -----------------------------------------------------------------------------
def load_model(model_path):
    print(f"\n📥 [System] Loading model from: {model_path} ...")
    config_path = f"{model_path}/config.json"
    if not os.path.exists(config_path):
        raise ValueError(f"❌ Config not found at {config_path}")
        
    with open(config_path, "r") as f:
        config_data = json.load(f)

    archs = config_data.get("architectures", [])
    arch_name = archs[0] if archs else "Unknown"
    print(f"🏗️  [System] Detected architecture: {arch_name}")

    # --- 自定义 Sparse 模型注册逻辑 ---
    if "PawLlama" in arch_name:
        from sparseattn.training.eval.modeling_flash_llama import (
            PawLlamaForCausalLM, PawLlamaConfig
        )
        AutoModelForCausalLM.register(PawLlamaConfig, PawLlamaForCausalLM)
        model_cls = PawLlamaForCausalLM
        is_sparse = True
    elif "PawQwen" in arch_name:
        from sparseattn.efficiency.model.modeling_flash_qwen import (
            PawQwen3ForCausalLM, PawQwen3Config
        )
        AutoModelForCausalLM.register(PawQwen3Config, PawQwen3ForCausalLM)
        model_cls = PawQwen3ForCausalLM
        is_sparse = True
    else:
        # --- 标准 Full Attention 模型 (如 Qwen2/3) ---
        print("ℹ️  [System] Loading as Standard (Full Attention) Model.")
        model_cls = AutoModelForCausalLM
        is_sparse = False

    model = model_cls.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model.eval()
    return model, is_sparse

# -----------------------------------------------------------------------------
# 2. 核心评测函数
# -----------------------------------------------------------------------------
def evaluate_efficiency(model, input_ids, gen_len=10, is_sparse=False):
    # 计时器初始化
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    # --- A. Prefill 阶段 ---
    torch.cuda.synchronize()
    start_event.record()
    
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
        
    end_event.record()
    torch.cuda.synchronize()
    prefill_time_ms = start_event.elapsed_time(end_event)
    
    past_key_values = outputs.past_key_values
    
    # 获取 Sparsity (仅 Sparse 模型有)
    current_sparsity = 0.0
    if is_sparse:
        try:
            sp = getattr(model, "prefill_sparsity", None)
            if isinstance(sp, torch.Tensor):
                current_sparsity = sp.item()
        except:
            pass

    # 准备 Decode
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
    
    # --- B. Decode 阶段 ---
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
# 3. 批量测试执行器 (修改：截断逻辑)
# -----------------------------------------------------------------------------
def run_benchmark_suite(model_path, samples, tokenizer, gen_len=10, max_len=4096):
    model, is_sparse = load_model(model_path)
    results = []
    
    # Warmup
    print("🔥 [System] Warming up GPU...")
    dummy = tokenizer.encode("Warmup " * 10, return_tensors="pt").to(model.device)
    evaluate_efficiency(model, dummy, gen_len=2, is_sparse=is_sparse)
    
    print(f"🏃 [System] Running benchmark on {len(samples)} samples (Max Len: {max_len})...")
    
    for i, item in enumerate(samples):
        input_text = item["input"]
        input_ids = tokenizer.encode(input_text, return_tensors="pt").to(model.device)
        seq_len = input_ids.shape[-1]
        
        note = ""
        # === 截断逻辑 ===
        if seq_len > max_len:
            input_ids = input_ids[:, :max_len]
            note = f"✂️ (Truncated {seq_len}->{max_len})"
            seq_len = max_len
        # ===============
            
        res = evaluate_efficiency(model, input_ids, gen_len=gen_len, is_sparse=is_sparse)
        res["seq_len"] = seq_len
        results.append(res)
        
        # 实时打印进度
        print(f"  Sample {i+1} {note}: 📏 Len {seq_len} | ⚡ Prefill {res['prefill_ms']:.1f}ms | ⏩ Decode {res['decode_ms_per_token']:.2f}ms/tok")
        
        del input_ids
    
    # 清理模型释放显存
    del model
    torch.cuda.empty_cache()
    gc.collect()
    print("🧹 [System] Model unloaded & Memory cleared.\n")
    
    return results

# -----------------------------------------------------------------------------
# 4. 主程序
# -----------------------------------------------------------------------------
def main():
    # ================= 配置区域 =================
    sparse_model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.1router4steps266_full_streaming_64k_qwen3-4b_wfrozen/checkpoint-230"
    full_model_path   = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.1router4steps266_full_streaming_64k_qwen3-4b_wfrozen/checkpoint-200" # Full Attention 模型路径
    
    data_path = "/data1/lcm_lab/sora/loomeval/benchmarks/General/RULER/data/niah_single_3_262144.jsonl"
    
    num_samples = 5       # 测试样本数
    gen_len = 1          # 生成长度
    max_input_len = 64 * 1024 # 最大长度限制 (超过此长度将被截断)
    # ===========================================

    # 1. 准备数据
    print(f"📂 [Init] Reading data from {data_path}")
    # 使用 Sparse 模型的 tokenizer 预处理
    tokenizer = AutoTokenizer.from_pretrained(sparse_model_path, trust_remote_code=True)
    
    raw_samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if len(raw_samples) >= num_samples: break
            try:
                raw_samples.append(json.loads(line))
            except: pass
            
    if not raw_samples:
        print("❌ Error: No data loaded.")
        return

    # 2. 运行 Sparse 模型
    print("🔹" * 20 + " PHASE 1: Benchmarking SPARSE Model " + "🔹" * 20)
    sparse_results = run_benchmark_suite(sparse_model_path, raw_samples, tokenizer, gen_len, max_input_len)

    # 3. 运行 Full 模型
    print("🔸" * 20 + " PHASE 2: Benchmarking FULL Model " + "🔸" * 20)
    full_results = run_benchmark_suite(full_model_path, raw_samples, tokenizer, gen_len, max_input_len)

    # 4. 对比与汇总
    print("\n" + "📊" * 15 + " FINAL COMPARISON REPORT " + "📊" * 15)
    print(f"{'ID':<4} | {'Len':<6} | {'Sparse (ms)':<18} | {'Full (ms)':<18} | {'🚀 Speedup (Full/Sparse)':<22}")
    print(f"{'':<4} | {'':<6} | {'⚡Prefill':<9} {'⏩Decode':<8} | {'⚡Prefill':<9} {'⏩Decode':<8} | {'⚡Prefill':<9} {'⏩Decode':<8}")
    print("-" * 100)

    avg_speedup_prefill = []
    avg_speedup_decode = []

    for i, (res_s, res_f) in enumerate(zip(sparse_results, full_results)):
        if res_s is None or res_f is None:
            continue
            
        len_tok = res_s['seq_len']
        
        # 提取指标
        p_s, d_s = res_s['prefill_ms'], res_s['decode_ms_per_token']
        p_f, d_f = res_f['prefill_ms'], res_f['decode_ms_per_token']
        
        # 计算加速比
        speedup_p = p_f / p_s if p_s > 0 else 0
        speedup_d = d_f / d_s if d_s > 0 else 0
        
        avg_speedup_prefill.append(speedup_p)
        avg_speedup_decode.append(speedup_d)
        
        # 高亮逻辑
        sp_p_str = f"\033[92m{speedup_p:<8.2f}x\033[0m" if speedup_p > 1.0 else f"{speedup_p:<8.2f}x"
        sp_d_str = f"\033[92m{speedup_d:<8.2f}x\033[0m" if speedup_d > 1.0 else f"{speedup_d:<8.2f}x"
        
        # 加个火焰 emoji 如果加速明显
        if speedup_p > 1.0: sp_p_str += "🔥"
        if speedup_d > 1.0: sp_d_str += "🔥"

        print(f"{i+1:<4} | {len_tok:<6} | {p_s:<9.1f} {d_s:<8.2f} | {p_f:<9.1f} {d_f:<8.2f} | {sp_p_str} {sp_d_str}")

    print("-" * 100)
    if avg_speedup_prefill:
        print(f"✨ Average Speedup -> Prefill: \033[1m{sum(avg_speedup_prefill)/len(avg_speedup_prefill):.2f}x\033[0m")
        print(f"✨ Average Speedup -> Decode : \033[1m{sum(avg_speedup_decode)/len(avg_speedup_decode):.2f}x\033[0m")
        
        # 打印平均稀疏度
        avg_spa = sum([r['sparsity'] for r in sparse_results if r])/len([r for r in sparse_results if r])
        print(f"📉 Average Sparse Rate: {avg_spa:.4f}")
    else:
        print("⚠️ No valid samples comparing.")

if __name__ == "__main__":
    main()