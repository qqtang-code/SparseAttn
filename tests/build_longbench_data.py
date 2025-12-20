import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional

# --- 配置参数 ---
SOURCE_DIR = "/data1/lcm_lab/sora/LOOM-Eval/benchmarks/General/LongBench/prediction/12.16sp3_templatesteps200_full_xattn_32k_qwen3-8b_wfrozen_LongBench_noise_64k"
OUTPUT_FILE = "/data2/lcm_lab/public_data/Longbench_50/all.parquet"

# 【核心修改点 1】：采样限制
# 设置为一个整数（如 50）表示每个文件最多取 50 条；设置为 None 则表示获取全部数据
MAX_SAMPLES_PER_FILE: Optional[int] = 50 

SKIP_FILES_PREFIX = {
    "triviaqa", "samsum", "lsht","trec",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh"
}

TASK_MAP = {
    "narrativeqa": "Single QA", "qasper": "Single QA",
    "multifieldqa_en": "Single QA", "multifieldqa_zh": "Single QA",
    "hotpotqa": "MultiHop QA", "2wikimqa": "MultiHop QA",
    "musique": "MultiHop QA", "dureader": "MultiHop QA",
    "gov_report": "Summarization", "qmsum": "Summarization",
    "multi_news": "Summarization", "vcsum": "Summarization",
    "repobench-p": "Code", "lcc": "Code"
}

def get_file_task(filename: str) -> str:
    for prefix, task in TASK_MAP.items():
        if filename.startswith(prefix):
            return task
    return "Other"

def process_data(source_dir: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    processed_records = []
    global_id_counter = 0

    print(f"🔄 开始处理目录: {source_dir}")
    if limit:
        print(f"📌 模式: 每个任务最多提取前 {limit} 条数据。")
    else:
        print("📌 模式: 提取所有数据。")

    for filename in sorted(os.listdir(source_dir)): # 排序一下让结果更稳定
        if not filename.endswith(".jsonl"):
            continue

        should_skip = any(filename.startswith(prefix) for prefix in SKIP_FILES_PREFIX)
        if should_skip:
            print(f"➡️ 跳过文件: {filename}")
            continue

        task = get_file_task(filename)
        file_path = os.path.join(source_dir, filename)
        
        # --- 【核心修改点 2】：文件内计数器 ---
        file_sample_count = 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                # 检查是否达到当前文件的上限
                if limit is not None and file_sample_count >= limit:
                    break

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"🚨 错误: 解析 {filename} 失败: {e}")
                    continue

                input_text = record.get("input_text", "")
                answers = record.get("answers", [])

                new_record = {
                    "id": str(global_id_counter),
                    "context": input_text,
                    "question": "",
                    "answer": json.dumps(answers, ensure_ascii=False),
                    "metadata": {
                        "flag": "0",
                        "source": filename,
                        "template": "",
                        "context_type": "",
                        "answer_type": "",
                        "length": len(input_text),
                        "task": task,
                        "is_prefix": False
                    },
                    "others": []
                }
                
                processed_records.append(new_record)
                global_id_counter += 1
                file_sample_count += 1 # 增加计数
        
        print(f"📖 文件 {filename} 处理完毕，提取了 {file_sample_count} 条。")

    print(f"✅ 处理完成。总计生成 {len(processed_records)} 条记录。")
    return processed_records

def save_to_parquet(data: List[Dict[str, Any]], output_path: str):
    df = pd.DataFrame(data)
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        print(f"💾 正在保存到: {output_path}")
        df.to_parquet(output_path, index=False)
        print("🎉 保存成功!")
    except Exception as e:
        print(f"🚨 保存失败: {e}")

if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误: 目录不存在: {SOURCE_DIR}")
    else:
        # 传入限制参数
        final_data = process_data(SOURCE_DIR, limit=MAX_SAMPLES_PER_FILE)
        
        if final_data:
            save_to_parquet(final_data, OUTPUT_FILE)
        else:
            print("🚫 无数据生成。")