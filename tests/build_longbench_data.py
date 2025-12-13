import os
import json
import pandas as pd
from typing import List, Dict, Any

# --- 配置参数 ---
# 原始 JSONL 文件目录
SOURCE_DIR = "/data1/lcm_lab/sora/LOOM-Eval/benchmarks/General/LongBench/prediction/12.11steps125_qwen_mix_sft_32K_xattn_mlp_ctx_q_new_softmax_wfrozen_LongBench_64k"
# 最终 Parquet 文件的输出路径
OUTPUT_FILE = "/data2/lcm_lab/public_data/Longbench/all.parquet"

# 需要跳过的文件名前缀（不读取这些文件）
SKIP_FILES_PREFIX = {
    "triviaqa", "samsum", "lsht","trec",
    "passage_count", "passage_retrieval_en", "passage_retrieval_zh"
}

# 任务分类映射 (文件名 -> 任务类型)
TASK_MAP = {
    # Single QA
    "narrativeqa": "Single QA", "qasper": "Single QA",
    "multifieldqa_en": "Single QA", "multifieldqa_zh": "Single QA",
    # MultiHop QA
    "hotpotqa": "MultiHop QA", "2wikimqa": "MultiHop QA",
    "musique": "MultiHop QA", "dureader": "MultiHop QA",
    # Summarization
    "gov_report": "Summarization", "qmsum": "Summarization",
    "multi_news": "Summarization", "vcsum": "Summarization",
    # Code
    "repobench-p": "Code", "lcc": "Code"
}

def get_file_task(filename: str) -> str:
    """根据文件名获取对应的任务类型。"""
    for prefix, task in TASK_MAP.items():
        if filename.startswith(prefix):
            return task
    # 对于其他未明确分类但未跳过的文件，给一个默认的任务类型
    return "Other"

def process_data(source_dir: str) -> List[Dict[str, Any]]:
    """
    遍历目录，读取和处理 JSONL 文件，并生成新的数据结构。
    """
    processed_records = []
    global_id_counter = 0

    print(f"🔄 开始处理目录: {source_dir}")

    # 1. 遍历目录下的所有文件
    for filename in os.listdir(source_dir):
        # 检查文件是否为 JSONL 格式
        if not filename.endswith(".jsonl"):
            continue

        file_path = os.path.join(source_dir, filename)

        # 2. 检查是否需要跳过该文件
        should_skip = False
        for prefix in SKIP_FILES_PREFIX:
            if filename.startswith(prefix):
                should_skip = True
                break
        
        if should_skip:
            print(f"➡️ 跳过文件: {filename} (在 SKIP_FILES_PREFIX 中)")
            continue

        # 3. 获取任务类型
        task = get_file_task(filename)
        if task == "Other":
             print(f"⚠️ 警告: 文件 {filename} 未明确分类，任务类型设置为 'Other'。")
        
        print(f"📖 正在处理文件: {filename}, 任务类型: {task}")

        # 4. 逐行读取 JSONL 文件内容
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    print(f"🚨 错误: 解析文件 {filename} 中的 JSONL 行失败: {e}")
                    continue

                # 提取所需的字段
                input_text = record.get("input_text", "")
                answers = record.get("answers", [])

                # 5. 构造新的数据结构
                new_record = {
                    "id": str(global_id_counter),
                    "context": input_text,
                    "question": "",  # 原始数据中未提供，留空或根据需要填充
                    "answer": json.dumps(answers, ensure_ascii=False), # 将 answers 列表转换为 JSON 字符串
                    "metadata": {
                        "flag": "0",
                        "source": filename, # 使用文件名作为 source
                        "template": "",
                        "context_type": "",
                        "answer_type": "",
                        "length": len(input_text), # 使用 input_text 的长度
                        "task": task,
                        "is_prefix": False # 默认设置为 False
                    },
                    "others": [] # 默认留空，可根据需要添加其他键值对
                }
                
                processed_records.append(new_record)
                global_id_counter += 1

    print(f"✅ 数据处理完成。共生成 {len(processed_records)} 条记录。")
    return processed_records

def save_to_parquet(data: List[Dict[str, Any]], output_path: str):
    """
    将处理后的数据列表转换为 Pandas DataFrame 并保存为 Parquet 文件。
    """
    # 1. 将数据列表转换为 DataFrame
    # 这一步会自动将 'metadata' 字段作为一个字典列处理
    df = pd.DataFrame(data)

    # 2. 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 3. 写入 Parquet 文件
    try:
        print(f"💾 正在保存数据到 Parquet 文件: {output_path}")
        df.to_parquet(output_path, index=False)
        print("🎉 数据成功保存!")
    except Exception as e:
        print(f"🚨 错误: 保存 Parquet 文件失败: {e}")

# --- 主执行逻辑 ---
if __name__ == "__main__":
    if not os.path.exists(SOURCE_DIR):
        print(f"❌ 错误: 原始目录不存在: {SOURCE_DIR}")
    else:
        # 1. 处理数据
        final_data = process_data(SOURCE_DIR)
        
        # 2. 保存数据
        if final_data:
            save_to_parquet(final_data, OUTPUT_FILE)
        else:
            print("🚫 没有生成任何数据，跳过保存。")