from sparseattn.efficiency.model.modeling_flash_qwen_streaming_decode import PawQwen3ForCausalLM, PawQwen3Config
from sparseattn.efficiency.model.modeling_flash_qwen import PawQwen3ForCausalLM as PawQwen3ForCausalLM_full_decode
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import pandas as pd
import gc
import os
from datetime import datetime

def truncate_input_ids(input_ids, max_len):
    """将input_ids截断到指定长度，保留头部一半和尾部一半"""
    seq_len = input_ids.shape[-1]
    
    if seq_len <= max_len:
        return input_ids, seq_len, ""
    
    # 策略：保留头部一半配额，保留尾部一半配额，中间切掉
    half_len = max_len // 2
    
    # 1. 取前 half_len
    head_part = input_ids[:, :half_len]
    
    # 2. 取后 (max - half) (为了处理奇数长度的情况)
    tail_part = input_ids[:, -(max_len - half_len):]
    
    # 3. 拼接
    truncated_input_ids = torch.cat([head_part, tail_part], dim=1)
    
    note = f"✂️ (Mid-Trunc {seq_len} -> {max_len})"
    return truncated_input_ids, max_len, note

def test_model_decode(model, model_name, tokenizer, input_ids, gen_len=10):
    """测试模型decode阶段的时间间隔"""
    print(f"  Testing {model_name}...")
    
    # Prefill阶段
    device = next(model.parameters()).device
    if input_ids.device != device:
        input_ids = input_ids.to(device)
    
    with torch.inference_mode():
        outputs = model(input_ids, use_cache=True)
    
    past_key_values = outputs.past_key_values
    next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
    
    # Decode阶段 - 收集时间数据
    decode_times = []
    
    with torch.inference_mode():
        for i in range(gen_len):
            outputs = model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1).unsqueeze(1)
            
            # 保存时间间隔数据
            if hasattr(outputs, 'time_intervals'):
                decode_times.append(outputs.time_intervals.copy())
            else:
                decode_times.append({})
    
    return decode_times

def analyze_decode_times(decode_times_list, model_type, gen_len=10):
    """分析decode阶段的时间数据，返回每个token的平均时间(ms)"""
    if model_type == 'streaming':
        # 对于streaming模型，计算总时间然后平均到每个token
        total_times = {
            'sparse_prep_time': 0,
            'sparse_attn_time': 0,
            'full_prep_time': 0,
            'full_attn_time': 0
        }
        
        if not decode_times_list:
            return total_times
            
        for times in decode_times_list:
            for key in total_times.keys():
                # 累加所有decode步骤的时间（秒）
                total_times[key] += times.get(key, 0)
        
        # 计算每个token的平均时间，并转换为毫秒
        avg_times = {}
        for key in total_times.keys():
            # 总时间除以token数，然后转换为毫秒 (1s = 1000ms)
            avg_times[key] = (total_times[key] / gen_len) * 1000 if gen_len > 0 else 0
        
        return avg_times
    
    elif model_type == 'full_decode':
        # 对于full_decode模型
        total_time = 0
        
        if not decode_times_list:
            return {'flash_attn_decode_time': 0}
            
        for times in decode_times_list:
            total_time += times.get('flash_attn_decode_time', 0)
        
        # 计算每个token的平均时间，并转换为毫秒
        avg_time = (total_time / gen_len) * 1000 if gen_len > 0 else 0
        
        return {'flash_attn_decode_time': avg_time}

def cleanup_memory():
    """清理GPU和内存"""
    torch.cuda.empty_cache()
    gc.collect()

if __name__ == "__main__":
    model_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/1.1router4steps266_full_streaming_64k_qwen3-4b_wfrozen/checkpoint-230"
    data_path = "/data1/lcm_lab/sora/loomeval/benchmarks/General/RULER/data/niah_single_3_262144.jsonl"
    
    num_samples = 5       # 每个长度测试的样本数
    gen_len = 10          # decode阶段的生成长度
    
    target_lengths_k = [32]  # 可以扩展更多长度
    target_lengths = [k * 1024 for k in target_lengths_k] 
    
    # 读取数据
    print(f"📂 [Init] Reading data from {data_path}")
    samples = []
    with open(data_path, 'r') as f:
        for i, line in enumerate(f):
            if len(samples) >= num_samples: 
                break
            try:
                item = json.loads(line)
                samples.append(item)
            except:
                pass
    
    print(f"📊 Loaded {len(samples)} samples for testing")
    print(f"🔧 Config: {gen_len} decode tokens per sample")
    
    # 准备结果存储
    results = []
    
    # 加载tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # ==================== 第一阶段：测试Full Decode模型 ====================
    print("\n" + "="*60)
    print("🔍 [Phase 1] Testing Full Decode Model")
    print("="*60)
    
    print("🔄 [Loading] Loading full decode model...")
    model_full = PawQwen3ForCausalLM_full_decode.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model_full.eval()
    
    # 存储full decode模型的结果
    full_decode_results = {}
    
    # 对每个目标长度进行测试
    for max_len in target_lengths:
        print(f"\n📏 [Testing] Testing full decode with max length: {max_len} tokens")
        
        for i, item in enumerate(samples):
            # 获取输入文本
            input_text = item.get("input", item.get("text", item.get("content", "")))
            if not input_text:
                # 尝试第一个键值对
                keys = list(item.keys())
                if keys:
                    input_text = str(item[keys[0]])
                else:
                    continue
            
            # 编码并截断
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            input_ids, seq_len, note = truncate_input_ids(input_ids, max_len)
            
            print(f"  Sample {i+1}: {seq_len} tokens {note}")
            
            # 测试full decode模型
            full_times = test_model_decode(model_full, "Full Decode", tokenizer, input_ids, gen_len)
            full_avg = analyze_decode_times(full_times, 'full_decode', gen_len)
            
            # 存储结果
            key = (seq_len, i)
            full_decode_results[key] = {
                'flash_attn_decode_time': full_avg.get('flash_attn_decode_time', 0)
            }
    
    # 清理full decode模型
    print("\n🧹 [Cleaning] Cleaning up full decode model...")
    del model_full
    cleanup_memory()
    
    # ==================== 第二阶段：测试Streaming Decode模型 ====================
    print("\n" + "="*60)
    print("🔍 [Phase 2] Testing Streaming Decode Model")
    print("="*60)
    
    print("🔄 [Loading] Loading streaming decode model...")
    model_streaming = PawQwen3ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    model_streaming.eval()
    
    # 对每个目标长度进行测试
    for max_len in target_lengths:
        print(f"\n📏 [Testing] Testing streaming decode with max length: {max_len} tokens")
        
        for i, item in enumerate(samples):
            # 获取输入文本
            input_text = item.get("input", item.get("text", item.get("content", "")))
            if not input_text:
                # 尝试第一个键值对
                keys = list(item.keys())
                if keys:
                    input_text = str(item[keys[0]])
                else:
                    continue
            
            # 编码并截断（与第一阶段保持相同的截断方式）
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            input_ids, seq_len, note = truncate_input_ids(input_ids, max_len)
            
            print(f"  Sample {i+1}: {seq_len} tokens {note}")
            
            # 测试streaming模型
            streaming_times = test_model_decode(model_streaming, "Streaming Decode", tokenizer, input_ids, gen_len)
            streaming_avg = analyze_decode_times(streaming_times, 'streaming', gen_len)
            
            # 合并两个模型的结果
            key = (seq_len, i)
            full_result = full_decode_results.get(key, {'flash_attn_decode_time': 0})
            
            result = {
                'prefill_length': seq_len,
                'sample_idx': i,
                'flash_attn_decode_time': full_result['flash_attn_decode_time'],
                'sparse_prep_time': streaming_avg.get('sparse_prep_time', 0),
                'sparse_attn_time': streaming_avg.get('sparse_attn_time', 0),
                'full_prep_time': streaming_avg.get('full_prep_time', 0),
                'full_attn_time': streaming_avg.get('full_attn_time', 0)
            }
            results.append(result)
    
    # 清理streaming模型
    print("\n🧹 [Cleaning] Cleaning up streaming decode model...")
    del model_streaming
    cleanup_memory()
    
    # ==================== 结果处理 ====================
    print("\n" + "="*60)
    print("📊 [Processing] Processing results")
    print("="*60)
    
    # 创建DataFrame
    df = pd.DataFrame(results)
    
    # 按prefill_length分组计算平均值
    if not df.empty:
        df_avg = df.groupby('prefill_length').mean().reset_index()
        df_avg = df_avg.round(3)  # 保留3位小数，毫秒级别精度足够
        
        # 计算speedup列：flash_attn_decode_time / (sparse_attn_time + full_attn_time)
        # 注意：这里只比较注意力计算时间，不包括prep时间
        df_avg['speedup'] = df_avg.apply(
            lambda row: row['flash_attn_decode_time'] / (row['sparse_attn_time'] + row['full_attn_time']) 
            if (row['sparse_attn_time'] + row['full_attn_time']) > 0 else 0,
            axis=1
        )
        df_avg['speedup'] = df_avg['speedup'].round(2)
        
        # 重新排列列顺序，将speedup放在最右边
        column_order = ['prefill_length', 'flash_attn_decode_time', 'sparse_prep_time', 
                        'sparse_attn_time', 'full_prep_time', 'full_attn_time', 'speedup']
        df_avg = df_avg[column_order]
        
        # 按prefill_length排序
        df_avg = df_avg.sort_values('prefill_length')
        
        # ==================== 保存CSV文件 ====================
        # 生成时间戳，避免文件名冲突
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 主要结果CSV文件 - 每个prefill长度的平均值（包含speedup列）
        main_csv = f"decode_time_comparison_ms_{timestamp}.csv"
        df_avg.to_csv(main_csv, index=False)
        
        print(f"\n✅ [CSV Saved] Main results saved to: {main_csv}")
        print(f"   - 文件位置: {os.path.abspath(main_csv)}")
        print(f"   - 数据行数: {len(df_avg)} (每个prefill长度一行)")
        print(f"   - 数据列数: {len(df_avg.columns)}")
        print(f"   - 包含列: {', '.join(df_avg.columns.tolist())}")
        print(f"   - Speedup定义: flash_attn_decode_time / (sparse_attn_time + full_attn_time)")
        print(f"     * speedup > 1: Full decode更慢")
        print(f"     * speedup < 1: Streaming decode更慢")
        
        # 2. 详细结果CSV文件 - 每个样本的原始数据
        detailed_csv = f"detailed_decode_results_ms_{timestamp}.csv"
        df.to_csv(detailed_csv, index=False)
        print(f"\n📋 [CSV Saved] Detailed results saved to: {detailed_csv}")
        print(f"   - 文件位置: {os.path.abspath(detailed_csv)}")
        print(f"   - 数据行数: {len(df)} (每个样本一行)")
        print(f"   - 包含sample_idx列用于追踪原始样本")
        
        # 3. 统计摘要CSV文件 - 汇总统计
        summary_stats = {
            'total_samples': len(df),
            'unique_prefill_lengths': len(df_avg),
            'gen_len_per_sample': gen_len,
            'test_timestamp': timestamp,
            'data_source': os.path.basename(data_path),
            'model_path': os.path.basename(model_path)
        }
        
        # 为每个时间指标添加统计信息
        time_columns = ['flash_attn_decode_time', 'sparse_prep_time', 'sparse_attn_time', 
                       'full_prep_time', 'full_attn_time', 'speedup']
        
        stats_df = pd.DataFrame([summary_stats])
        for col in time_columns:
            if col in df_avg.columns:
                if col == 'speedup':
                    # 对于speedup，计算几何平均值可能更有意义
                    stats_df[f'{col}_mean'] = df_avg[col].mean()
                    stats_df[f'{col}_min'] = df_avg[col].min()
                    stats_df[f'{col}_max'] = df_avg[col].max()
                else:
                    stats_df[f'{col}_mean'] = df_avg[col].mean()
                    stats_df[f'{col}_min'] = df_avg[col].min()
                    stats_df[f'{col}_max'] = df_avg[col].max()
                    stats_df[f'{col}_std'] = df_avg[col].std()
        
        stats_csv = f"summary_stats_ms_{timestamp}.csv"
        stats_df.to_csv(stats_csv, index=False)
        print(f"\n📈 [CSV Saved] Summary statistics saved to: {stats_csv}")
        
        print("\n📊 Summary Table (单位: ms/token):")
        print(df_avg.to_string(index=False))
        
        print(f"\n📝 Note: All times are averaged over {gen_len} decode tokens (ms per token)")
        print(f"\n💾 [File Summary] Saved CSV files:")
        print(f"   1. {main_csv} - 主要结果 (按prefill长度分组平均值，包含speedup列)")
        print(f"   2. {detailed_csv} - 详细结果 (每个样本的原始数据)")
        print(f"   3. {stats_csv} - 统计摘要 (汇总统计信息)")
        
        # 简单总结speedup结果
        print(f"\n🚀 [Speedup Summary]:")
        avg_speedup = df_avg['speedup'].mean()
        min_speedup = df_avg['speedup'].min()
        max_speedup = df_avg['speedup'].max()
        
        print(f"   - Average speedup: {avg_speedup:.2f}x")
        print(f"   - Min speedup: {min_speedup:.2f}x")
        print(f"   - Max speedup: {max_speedup:.2f}x")
        
        if avg_speedup > 1:
            print(f"   - Overall: Full decode is {avg_speedup:.1f}x slower than streaming decode")
        else:
            print(f"   - Overall: Streaming decode is {1/avg_speedup:.1f}x slower than full decode" if avg_speedup > 0 else "   - Cannot compare")
        
    else:
        print("❌ No valid results collected!")