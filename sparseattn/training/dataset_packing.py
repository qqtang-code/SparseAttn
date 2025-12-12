import os
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
    
    if isinstance(meta, str):
        try:
            meta = ast.literal_eval(meta)
        except:
            meta = {}
    
    flag = str(meta.get("flag", "0"))
    task_type = meta.get('task', 'Other')
    class_id = class_map.get(task_type, 4) # 4 for Other

    separator = "\n\n"

    # Context
    if flag == "1" or not ctx:
        ctx_text = ""
    else:
        ctx_text = "\n" + ctx.rstrip()
    ctx_ids = tokenizer(ctx_text, add_special_tokens=False)["input_ids"]

    # Question
    if flag == "1":
        q_text = "\n" + q.lstrip()
    else:
        q_text = "\n" + q.lstrip() if ctx and q else (q.lstrip() if q and not ctx else "")
    q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

    # Answer
    if a:
        a_text = separator + a
        a_ids = tokenizer(a_text, add_special_tokens=False)["input_ids"]
    else:
        a_ids = []

    full_input_ids = []
    # Segment 1: Context
    full_input_ids.extend(ctx_ids)
    # Segment 2: Question
    full_input_ids.extend(q_ids)
    # Segment 3: Answer
    full_input_ids.extend(a_ids)

    # Add EOS
    if tokenizer.eos_token_id is not None and (not full_input_ids or full_input_ids[-1] != tokenizer.eos_token_id):
        full_input_ids.append(tokenizer.eos_token_id) 
        
    labels = list(full_input_ids)
    # (如果需要mask answer前面的部分，可以在这里加逻辑)

    return {
        "input_ids": full_input_ids,
        "labels": labels,
        "task_id": class_id,
    }

def _finalize_pack(tokenizer, input_ids, labels, task_ids, lengths):
    """打包收尾：Padding并转换为Tensor结构"""
    curr_len = len(input_ids)
    remainder = curr_len % 8
    if remainder != 0:
        pad_len = 8 - remainder
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
        input_ids.extend([pad_id] * pad_len)
        labels.extend([-100] * pad_len)
    
    seq_lengths = [0] + list(np.cumsum(lengths))
    
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "seq_lengths": torch.tensor(seq_lengths, dtype=torch.int32),
        "task_ids": torch.tensor(task_ids, dtype=torch.long),
    }

def worker_pack_chunk(chunk_dataset, tokenizer, max_seq_len, worker_id=0):
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

    # 遍历当前 chunk 的数据
    # 使用 tqdm 需要指定 position 避免多进程打印混乱，或者直接去掉
    iterator = chunk_dataset
    if worker_id == 0: # 只让第一个进程打印进度条，或者每个都打但不换行
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
        else:
            # Buffer 满了，finalize
            packed_item = _finalize_pack(tokenizer, buf_input_ids, buf_labels, buf_task_ids, buf_lengths)
            local_packed_data.append(packed_item)
            
            # Reset buffer
            buf_input_ids = list(p_input_ids)
            buf_labels = list(processed["labels"])
            buf_task_ids = [processed["task_id"]]
            buf_lengths = [p_len]

    # 处理最后一个 buffer
    if buf_input_ids:
        packed_item = _finalize_pack(tokenizer, buf_input_ids, buf_labels, buf_task_ids, buf_lengths)
        local_packed_data.append(packed_item)
        
    return local_packed_data

# =========================================================
#  主 Dataset 类
# =========================================================

class PackedDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_seq_len=128*1024, cache_dir=None, num_proc=8):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packed_data = []
        
        # 缓存逻辑
        self.cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = f"packed_sft_len{len(raw_dataset)}_seq{max_seq_len}.pt"
            self.cache_path = os.path.join(cache_dir, cache_filename)

        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"🚀 发现缓存文件: {self.cache_path}")
            try:
                self.packed_data = torch.load(self.cache_path)
                logger.info(f"✅ 成功加载缓存! 包含 {len(self.packed_data)} 条序列。")
                return 
            except Exception as e:
                logger.warning(f"⚠️ 加载缓存失败 ({e})，准备重新打包...")

        logger.info(f"开始多进程 Packing... 目标长度: {max_seq_len}, 进程数: {num_proc}")
        
        # ================= 多进程处理逻辑 =================
        self._parallel_pack_dataset(raw_dataset, num_proc)
        # ================================================

        if self.cache_path:
            logger.info(f"💾 正在保存缓存到: {self.cache_path} ...")
            try:
                torch.save(self.packed_data, self.cache_path)
                logger.info("✅ 缓存保存成功!")
            except Exception as e:
                logger.error(f"❌ 缓存保存失败: {e}")

    def _parallel_pack_dataset(self, raw_dataset, num_proc):
        # 1. 切分数据集
        # HuggingFace dataset 支持 .shard(), 或者简单的切片
        # 为了更均匀，我们手动计算 indices 或使用 shard
        total_size = len(raw_dataset)
        
        # 确保进程数不超过数据量
        num_proc = min(num_proc, total_size)
        if num_proc < 1: num_proc = 1
        
        logger.info(f"Splitting dataset into {num_proc} chunks...")
        
        # 使用 shard 切分，这是 HF Dataset 最高效的方式（Lazy）
        chunks = []
        for i in range(num_proc):
            # contiguous=True is important for speed on read
            chunks.append(raw_dataset.shard(num_shards=num_proc, index=i, contiguous=True))

        # 2. 提交任务到进程池
        futures = []
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            for i, chunk in enumerate(chunks):
                # 提交任务
                # 注意：tokenizer 需要被 pickle 传过去，通常没问题
                # raw_dataset 的 shard 也是 lazy 的，传输开销很小
                futures.append(
                    executor.submit(worker_pack_chunk, chunk, self.tokenizer, self.max_seq_len, i)
                )
            
            # 3. 收集结果
            results = []
            for f in tqdm(as_completed(futures), total=len(futures), desc="Waiting for workers"):
                try:
                    res = f.result()
                    results.extend(res)
                except Exception as e:
                    logger.error(f"Worker failed with error: {e}")
                    raise e
        
        self.packed_data = results
        logger.info(f"多进程 Packing 完成。原始: {total_size} -> Packed: {len(self.packed_data)}")

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        return self.packed_data[idx]

# =========================================================
#  Utilities & Main
# =========================================================

@dataclass
class PackedDataArguments:
    per_device_max_tokens: int = 32768
    min_seq_len: Optional[int] = None
    data_cache_dir: Optional[str] = None
    # Add a param for num_proc
    preprocessing_num_workers: int = 64

def build_packed_dataset(paths, data_args, tokenizer=None):
    if isinstance(paths, str):
        paths = [paths]
    
    parquet_files = []
    for p in paths:
        if os.path.isdir(p):
            parquet_files.extend(glob.glob(os.path.join(p, "*.parquet")))
        elif os.path.isfile(p) and p.endswith(".parquet"):
            parquet_files.append(p)
    
    if not parquet_files:
        raise ValueError("No parquet files found")

    # Load raw
    raw = load_dataset(
        "parquet", 
        data_files=parquet_files, 
        split="train", 
        cache_dir=os.path.join(data_args.data_cache_dir, "raw") if data_args.data_cache_dir else None
    )

    # Filter short
    if data_args.min_seq_len is not None:
        # 过滤也可以考虑多进程: raw.filter(..., num_proc=os.cpu_count())
        pass

    max_len = data_args.per_device_max_tokens
    
    # 实例化并触发多进程处理
    return PackedDataset(
        raw, 
        tokenizer, 
        max_seq_len=128*1024, # 根据需要调整
        cache_dir="data_cache",
        num_proc=data_args.preprocessing_num_workers # 使用参数控制核数
    )

class PackedDataCollator:
    def __init__(self, tokenizer=None, data_args=None, max_seq_len=None):
        # 保留接口兼容性，但在 Packing 模式下通常不需要 pad，因为都已经 pack 满或 pad 好了
        self.tokenizer = tokenizer 
        self.data_args = data_args
        self.max_seq_len = max_seq_len

    def __call__(self, batch: List[Dict]):
        # batch 是一个 list，包含多个 dataset[i] 的结果
        
        # 1. 处理定长 Tensor (Input IDs, Labels)
        # 这些已经是 padding 到 max_seq_len 的，可以直接 stack
        input_ids = torch.stack([item['input_ids'] for item in batch], dim=0)
        labels = torch.stack([item['labels'] for item in batch], dim=0)
        
        # 2. 处理变长 Tensor (seq_lengths, task_ids)
        # 因为每个 packing 样本包含的子样本数量不同，无法直接 stack
        # 策略：如果是 batch_size=1，可以直接取出来；如果是 >1，通常保持为 list 或 flatten
        
        seq_lengths = None
        if 'seq_lengths' in batch[0]:
            # 保持为 List[Tensor]，交给模型内部处理 (例如 FlashAttn 的 varlen 接口通常需要把它 flatten)
            seq_lengths = [item['seq_lengths'] for item in batch]
            
        task_ids = None
        if 'task_ids' in batch[0]:
            task_ids = [item['task_ids'] for item in batch]

        
        res = {
            "input_ids": input_ids,
            "labels": labels,
            "seq_lengths": seq_lengths, # List[Tensor]
            "task_ids": task_ids,       # List[Tensor]
        }

        return res
if __name__ == "__main__":
    # 多进程必须在 main block 下运行
    multiprocessing.set_start_method("spawn", force=True) # 推荐在 CUDA 环境或复杂库中使用 spawn

    path = "/data2/public_data/qwen_mix_sft_128K" 
    data_args = PackedDataArguments(preprocessing_num_workers=64) # 设置为你机器的 CPU 核心数
    tokenizer = AutoTokenizer.from_pretrained("/data2/hf_models/Qwen3-4B", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    dataset = build_packed_dataset(
        paths=path,
        data_args=data_args,
        tokenizer=tokenizer
    )
    
    print(f"Dataset ready. Size: {len(dataset)}")
    # check one
    # print(dataset[0])