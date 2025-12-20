import os
import datasets
import torch
import logging
import ast
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from typing import Optional
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import shutil

# 设置日志
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# 全局常量
CLASS_MAP = {
    'Single QA': 0, 
    'MultiHop QA': 1, 
    'Summarization': 2, 
    'Code': 3
}

@dataclass
class PackedDataArguments:
    single_seq: bool = False
    subsplit_length: Optional[int] = None
    per_device_max_tokens: int = 128*1024
    apply_instruct_masks: bool = False
    prepack: bool = False
    streaming: bool = False
    min_seq_len: Optional[int] = 1000
    task_type: str = "pretrain" 
    use_packing: bool = False
    data_cache_dir: Optional[str] = None
    preprocessing_num_workers: int = 32

# =========================================================
#  独立的处理函数 (Worker Function)
#  必须放在顶层，以便多进程序列化 (Pickle)
# =========================================================

def _process_single_item(item, tokenizer, class_map):
    """处理单条数据为 token ids (无截断)"""
    ctx = item.get("context", "") or ""
    q = item.get("question", "") or ""
    a = item.get("answer", "") or ""
    meta = item.get("metadata", {}) or {}
    task_type = "Other"
    is_prefix = True
    try:
        meta_dict = ast.literal_eval(meta) if isinstance(meta, str) else meta
        task_type = meta_dict.get('task', 'Other')
        is_prefix = meta_dict.get('is_prefix', True)
    except Exception:
        pass


    separator = "\n\n"


    # Context (Segment ID 1)
    ctx_text = "\n" + ctx.rstrip()
    # ctx_ids = tokenizer(ctx_text, add_special_tokens=False)["input_ids"]

    # Question (Segment ID 2)
    q_text = "\n" + q.lstrip()
    # q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

    if is_prefix:
        user_text = q_text + "\n" + ctx_text

    else:
        user_text = ctx_text + "\n" + q_text
        
    if task_type == "Summarization":
        user_text = "You are given several news passages. Write a one-page summary of all news." + user_text + "\n\nSummary:"
    if task_type == "Code":
        user_text = "Please complete the code given below." + user_text

    messages = [
        {"role": "user", "content": user_text}
    ]
    user_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        enable_thinking=False,
    )
    user_text_ids = tokenizer(user_text, add_special_tokens=False)["input_ids"]
    

    # Separator + Answer (Segment ID 3)
    if a:
        
        messages = [
            {"role": "assistant", "content": a}
        ]
        a_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            enable_thinking=False,
        )
        a_text = separator + a_text
        a_ids = tokenizer(a_text, add_special_tokens=False)["input_ids"]

    else:
        a_ids = []

    # --- 3. Construct Full Sequence, Segment IDs, and Ranges ---

    # [TASK_TOKEN] + [CTX] + [Q] + [SEPARATOR] + [ANSWER]

    current_len = 0

    # Task (Segment 0)
    full_input_ids = []
    segment_ids = []
    special_start = special_end = 0

    user_text_start = current_len
    full_input_ids.extend(user_text_ids)
    segment_ids.extend([1] * len(user_text_ids))
    current_len += len(user_text_ids)
    user_text_end = current_len - 1

    # Answer (Segment 3) + Separator
    a_start = current_len
    full_input_ids.extend(a_ids)
    segment_ids.extend([3] * len(a_ids))
    current_len += len(a_ids)
    a_end = current_len - 1 if a_ids else a_start

    # Add EOS token at the very end
    if tokenizer.eos_token_id is not None and full_input_ids[-1] != tokenizer.eos_token_id:
        full_input_ids.append(tokenizer.eos_token_id)
        segment_ids.append(3) 
        current_len += 1
        a_end = current_len - 1

    # --- 4. Apply Truncation ---
    original_len = len(full_input_ids)

    # labels = [-100] * len(full_input_ids)
    # if a_ids:
    #     labels[a_start:len(full_input_ids)] = full_input_ids[a_start:len(full_input_ids)]
    labels = full_input_ids.copy()

    range_ids = [special_start, special_end, user_text_start, user_text_end, user_text_start, user_text_end, a_start, a_end]

    class_id = class_map.get(task_type, 4) # 4 for Other
    labels = list(full_input_ids)

    return {
        "input_ids": full_input_ids,
        "labels": labels,
        "task_id": class_id,
        "task_type": task_type,
        "range_ids": range_ids,
    }

def _finalize_pack(tokenizer, input_ids, labels, task_ids, lengths, task_types, range_ids):
    """打包收尾：Padding并转换为Tensor结构"""
    seq_lengths = [0] + list(np.cumsum(lengths))

    return {
        "input_ids": input_ids,          # List[int]
        "labels": labels,                # List[int]
        "seq_lengths": seq_lengths,      # List[int]
        "task_ids": task_ids,            # List[int]
        "task_type": task_types,         # List[str]
        "range_ids": range_ids,          # List[int] [8]
    }

def worker_pack_chunk(chunk_dataset, tokenizer, max_seq_len, worker_id):
    """
    子进程执行的函数：处理分配给它的那一部分数据
    """
    # 重要：防止 tokenizer 内部再次并行导致死锁或性能下降
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_packed_data = []

    # Buffers
    buf_input_ids = []
    buf_labels = []
    buf_task_ids = []    
    buf_lengths = []     
    buf_task_types = []
    buf_range_ids = []

    # 遍历当前 chunk 的数据
    # 使用 tqdm 需要指定 position 避免多进程打印混乱，或者直接去掉
    iterator = chunk_dataset
    if worker_id % 4 == 3:
        iterator = tqdm(chunk_dataset, desc=f"Worker {worker_id} Packing", position=worker_id)

    for item in iterator:
        processed = _process_single_item(item, tokenizer, CLASS_MAP)

        p_input_ids = processed["input_ids"]
        p_len = len(p_input_ids)

        if p_len > max_seq_len:
            # 单条过长直接跳过
            continue

        # 贪心打包检查
        if len(buf_input_ids) + p_len <= max_seq_len:
            buf_input_ids.extend(p_input_ids)
            buf_labels.extend(processed["labels"])
            buf_task_ids.append(processed["task_id"])
            buf_lengths.append(p_len)
            buf_task_types.append(processed["task_type"])
            buf_range_ids.append(processed["range_ids"])
        else:
            # Buffer 满了，finalize
            packed_item = _finalize_pack(tokenizer, buf_input_ids, buf_labels, buf_task_ids, buf_lengths, buf_task_types, buf_range_ids)
            local_packed_data.append(packed_item)

            # Reset buffer
            buf_input_ids = list(p_input_ids)
            buf_labels = list(processed["labels"])
            buf_task_ids = [processed["task_id"]]
            buf_lengths = [p_len]
            buf_task_types = [processed["task_type"]]
            buf_range_ids = [processed["range_ids"]]

    # 处理最后一个 buffer
    if buf_input_ids:
        packed_item = _finalize_pack(tokenizer, buf_input_ids, buf_labels, buf_task_ids, buf_lengths, buf_task_types, buf_range_ids)
        local_packed_data.append(packed_item)

    return local_packed_data

# =========================================================
#  主 Dataset 类
# =========================================================

class PackedDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_seq_len=128*1024, cache_dir=None, num_proc=8, raw_path = None):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packed_data = None

        # 缓存逻辑
        self.cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            # 这里的后缀改为 .parquet
            cache_filename = f"{os.path.basename(raw_path)}_packed_maxseq{max_seq_len}.parquet"
            self.cache_path = os.path.join(cache_dir, cache_filename)

        if self.cache_path and os.path.exists(self.cache_path):
            print(f"🚀 发现缓存文件: {self.cache_path}")
            try:
                self.packed_data = load_dataset("parquet", data_files=self.cache_path, split="train",
                                                cache_dir="/data2/public_data/data_cache")
                print(f"✅ 成功加载 Parquet 缓存! 包含 {len(self.packed_data)} 条序列。")
                return 
            except Exception as e:
                logger.warning(f"⚠️ 加载缓存失败 ({e})，准备重新打包...")

        print(f"开始多进程 Packing... 目标长度: {max_seq_len}, 进程数: {num_proc}")

        # 多进程处理，得到一个巨大的 List[Dict]
        packed_data_list = self._parallel_pack_dataset(raw_dataset, num_proc)
        
        keys = ["input_ids", "labels", "seq_lengths", "task_ids", "task_type", "range_ids"]
        columnar = {k: [] for k in keys}
        for item in packed_data_list:
            for k in keys:
                columnar[k].append(item[k])

        print("正在转换为 HuggingFace Dataset 对象...")
        #self.packed_data = datasets.Dataset.from_list(packed_data_list)
        self.packed_data = datasets.Dataset.from_dict(columnar)

        # 保存最终缓存
        if self.cache_path:
            print(f"💾 正在保存 Parquet 到: {self.cache_path} ...")
            try:
                self.packed_data.to_parquet(self.cache_path) 
                print("✅ Parquet 保存成功!")
            except Exception as e:
                logger.error(f"❌ 缓存保存失败: {e}")

    def _parallel_pack_dataset(self, raw_dataset, num_proc):
        total_size = len(raw_dataset)
        num_proc = min(num_proc, total_size)
        if num_proc < 1: num_proc = 1

        print(f"Splitting dataset into {num_proc} chunks...")

        chunks = []
        for i in range(num_proc):
            chunks.append(raw_dataset.shard(num_shards=num_proc, index=i, contiguous=True))

        # 提交任务
        futures = []
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(worker_pack_chunk, chunk, self.tokenizer, self.max_seq_len, i)
                )
        print(f"所有子进程处理完毕，开始汇总数据...")

        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Waiting for workers"):
            try:
                res = f.result()
                results.extend(res)
            except Exception as e:
                logger.error(f"Worker failed with error: {e}")
                raise e

        print(f"多进程 Packing 完成。原始: {total_size} -> Packed: {len(results)}")
        return results

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        # HF Dataset 默认返回 Python List，这里可以不转 Tensor，
        # 留给 Collator 转，或者在这里转。建议在这里转，保持旧接口习惯。
        item = self.packed_data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "seq_lengths": torch.tensor(item["seq_lengths"], dtype=torch.int32),
            "task_ids": torch.tensor(item["task_ids"], dtype=torch.long),
            "task_type": item["task_type"], # 字符串列表保持原样
            "range_ids": torch.tensor(item["range_ids"], dtype=torch.long),
        }

# =========================================================
#  Utilities & Main
# =========================================================


def build_packed_dataset(paths: str, data_args, tokenizer=None):
    # if isinstance(paths, str):
    #     paths = [paths]

    parquet_files = []
    # for p in paths:
    if os.path.isdir(paths):
        parquet_files.extend(glob.glob(os.path.join(paths, "*.parquet")))
    elif os.path.isfile(paths) and paths.endswith(".parquet"):
        parquet_files.append(paths)

    if not parquet_files:
        raise ValueError("No parquet files found")

    # Load raw
    raw = load_dataset(
        "parquet", 
        data_files=parquet_files, 
        split="train", 
        cache_dir=os.path.join(data_args.data_cache_dir, "raw") if data_args.data_cache_dir else None
    )
    
    # def filter_fn(x):
    #     task_type = x.get("metadata", {}).get('task', 'Other')
    #     if task_type == "Summarization" or task_type == "Code":
    #         return False
    #     return task_type == "Single QA" or task_type == "MultiHop QA"
    # raw = raw.filter(filter_fn, num_proc=os.cpu_count())

    # 2. 检查并计算 length 字段 (如果原数据没有)
    if "length" not in raw.column_names:
        print("Extracting 'length' from metadata for sorting...")

        # 这里的 int() 很重要：
        # 1. 你的 JSON 示例里 length 是字符串 ("length": "")
        # 2. 如果不转 int，排序会按字典序 ("10" 排在 "2" 前面)，导致打包效率变差
        raw = raw.map(
            lambda x: {"length": int(x["metadata"]["length"]) if x["metadata"]["length"] else 0},
            num_proc=data_args.preprocessing_num_workers,
            desc="Extracting lengths"
        )

    # 3. 按照 length 从小到大排序
    print("📉 正在按 length 从小到大排序数据...")
    raw = raw.sort("length", reverse=False)

    max_len = data_args.per_device_max_tokens

    # 实例化并触发多进程处理
    return PackedDataset(
        raw, 
        tokenizer, 
        max_seq_len=max_len, # 根据需要调整
        cache_dir=data_args.data_cache_dir,
        num_proc=data_args.preprocessing_num_workers, # 使用参数控制核数
        raw_path = paths,
    )


if __name__ == "__main__":
    # 1. 多进程启动方式设置 (CUDA环境必备)
    multiprocessing.set_start_method("spawn", force=True) 

    # 2. 配置参数
    # 建议先用小数据或少量 worker 测试，跑通后再调大
    path = "/data2/public_data/qwen_mix_sft_128K" 
    data_args = PackedDataArguments(
        preprocessing_num_workers=32,
        data_cache_dir="/data2/public_data/data_cache",
        per_device_max_tokens=131072
    )

    # 3. 加载 Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("/data2/hf_models/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 构建/加载数据集 (自动触发 排序 -> Packing -> Parquet保存)
    import time
    print(f"\n⏱️  Start building dataset...")
    start_time = time.time()
    dataset = build_packed_dataset(
        paths=path,
        data_args=data_args,
        tokenizer=tokenizer
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️  Done! Total time cost: {elapsed:.2f} s")

    print(f"\n✅ Dataset ready. Size: {len(dataset)}")

    # 5. 【验证环节 1】检查单条数据
    # 注意：根据 PackedDataset.__getitem__ 的实现，这里打印出来的应该是 Tensor
    item0 = dataset[1000]
    print("\n--- Sample 0 Check ---")
    print(f"Keys: {item0.keys()}")
    print(f"Input IDs Shape: {item0['input_ids'].shape}")
    print(f"Task Types: {item0['task_type']}")
    print(f"Seq Lengths (cum): {item0['seq_lengths']}")
    print(f"Range ids: {item0['range_ids']}")


    # breakpoint()