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
class DataArguments:
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
        "task_type": task_type,
    }

def _finalize_pack(tokenizer, input_ids, labels, task_ids, lengths, task_types):
    """打包收尾：Padding并转换为Tensor结构"""
    # curr_len = len(input_ids)
    # remainder = curr_len % 8
    # if remainder != 0:
    #     pad_len = 8 - remainder
    #     pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        
    #     input_ids.extend([pad_id] * pad_len)
    #     labels.extend([-100] * pad_len)
    
    seq_lengths = [0] + list(np.cumsum(lengths))
    
    return {
        "input_ids": input_ids,          # List[int]
        "labels": labels,                # List[int]
        "seq_lengths": seq_lengths,      # List[int]
        "task_ids": task_ids,            # List[int]
        "task_type": task_types,         # List[str]
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
        else:
            # Buffer 满了，finalize
            packed_item = _finalize_pack(tokenizer, buf_input_ids, buf_labels, buf_task_ids, buf_lengths, buf_task_types)
            local_packed_data.append(packed_item)
            
            # Reset buffer
            buf_input_ids = list(p_input_ids)
            buf_labels = list(processed["labels"])
            buf_task_ids = [processed["task_id"]]
            buf_lengths = [p_len]
            buf_task_types = [processed["task_type"]]

    # 处理最后一个 buffer
    if buf_input_ids:
        packed_item = _finalize_pack(tokenizer, buf_input_ids, buf_labels, buf_task_ids, buf_lengths, buf_task_types)
        local_packed_data.append(packed_item)
    
    return local_packed_data

# =========================================================
#  主 Dataset 类
# =========================================================

class PackedDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, max_seq_len=128*1024, cache_dir=None, num_proc=8):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.packed_data = None
        
        # 缓存逻辑
        self.cache_path = None
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            # 这里的后缀改为 .parquet
            cache_filename = f"packed_sft_len{len(raw_dataset)}_seq{max_seq_len}.parquet"
            self.cache_path = os.path.join(cache_dir, cache_filename)

        if self.cache_path and os.path.exists(self.cache_path):
            logger.info(f"🚀 发现缓存文件: {self.cache_path}")
            try:
                self.packed_data = load_dataset("parquet", data_files=self.cache_path, split="train")
                logger.info(f"✅ 成功加载 Parquet 缓存! 包含 {len(self.packed_data)} 条序列。")
                return 
            except Exception as e:
                logger.warning(f"⚠️ 加载缓存失败 ({e})，准备重新打包...")

        logger.info(f"开始多进程 Packing... 目标长度: {max_seq_len}, 进程数: {num_proc}")

        # 多进程处理，得到一个巨大的 List[Dict]
        packed_data_list = self._parallel_pack_dataset(raw_dataset, num_proc)

        logger.info("正在转换为 HuggingFace Dataset 对象...")
        self.packed_data = datasets.Dataset.from_list(packed_data_list)

        # 保存最终缓存
        if self.cache_path:
            logger.info(f"💾 正在保存 Parquet 到: {self.cache_path} ...")
            try:
                self.packed_data.to_parquet(self.cache_path) 
                logger.info("✅ Parquet 保存成功!")
            except Exception as e:
                logger.error(f"❌ 缓存保存失败: {e}")

    def _parallel_pack_dataset(self, raw_dataset, num_proc):
        total_size = len(raw_dataset)
        num_proc = min(num_proc, total_size)
        if num_proc < 1: num_proc = 1
        
        logger.info(f"Splitting dataset into {num_proc} chunks...")
        
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
        logger.info(f"所有子进程处理完毕，开始汇总数据...")
        
        results = []
        for f in tqdm(as_completed(futures), total=len(futures), desc="Waiting for workers"):
            try:
                res = f.result()
                results.extend(res)
            except Exception as e:
                logger.error(f"Worker failed with error: {e}")
                raise e
        
        logger.info(f"多进程 Packing 完成。原始: {total_size} -> Packed: {len(results)}")
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
            "task_type": item["task_type"] # 字符串列表保持原样
        }

# =========================================================
#  Utilities & Main
# =========================================================


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

    # 2. 检查并计算 length 字段 (如果原数据没有)
    if "length" not in raw.column_names:
        logger.info("Extracting 'length' from metadata for sorting...")
        
        # 这里的 int() 很重要：
        # 1. 你的 JSON 示例里 length 是字符串 ("length": "")
        # 2. 如果不转 int，排序会按字典序 ("10" 排在 "2" 前面)，导致打包效率变差
        raw = raw.map(
            lambda x: {"length": int(x["metadata"]["length"]) if x["metadata"]["length"] else 0},
            num_proc=data_args.preprocessing_num_workers,
            desc="Extracting lengths"
        )

    # 3. 按照 length 从小到大排序
    logger.info("📉 正在按 length 从小到大排序数据...")
    raw = raw.sort("length", reverse=False)

    max_len = data_args.per_device_max_tokens
    
    # 实例化并触发多进程处理
    return PackedDataset(
        raw, 
        tokenizer, 
        max_seq_len=max_len, # 根据需要调整
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
        
        # 1. 确保 input_ids 和 labels 是 Tensor
        input_ids_list = [item['input_ids'] for item in batch]
        labels_list = [item['labels'] for item in batch]
        
        # 如果 __getitem__ 没有转 Tensor，这里进行转换
        if not isinstance(input_ids_list[0], torch.Tensor):
            input_ids = torch.tensor(input_ids_list, dtype=torch.long)
            labels = torch.tensor(labels_list, dtype=torch.long)
        else:
            input_ids = torch.stack(input_ids_list, dim=0)
            labels = torch.stack(labels_list, dim=0)
        
        # 2. 处理变长字段 (seq_lengths, task_ids)
        # 它们是 List[Tensor] 或 List[List]，Collator 最终通常不 Stack 变长字段
        # 或者 Flatten 它们（取决于你的模型 FlashAttn 实现）
        
        seq_lengths = [item['seq_lengths'] for item in batch]
        task_ids = [item['task_ids'] for item in batch]
        
        # 确保内部也是 Tensor (针对变长部分)
        if not isinstance(seq_lengths[0], torch.Tensor):
             seq_lengths = [torch.tensor(s, dtype=torch.int32) for s in seq_lengths]
             
        if not isinstance(task_ids[0], torch.Tensor):
             task_ids = [torch.tensor(t, dtype=torch.long) for t in task_ids]

        task_types = [item['task_type'] for item in batch]
        
        res = {
            "input_ids": input_ids,     # [B, S], B == 1 
            "labels": labels,           # [B, S]
            "seq_lengths": seq_lengths, # List[Tensor] Tensor shape: [Bi + 1], Bi is the number of sub-samples in the seq.
            "task_ids": task_ids,       # List[Tensor] Tensor shape: [Bi]
            "task_type": task_types     # List[Tensor] Tensor shape: [Bi]
        }

        return res

if __name__ == "__main__":
    # 1. 多进程启动方式设置 (CUDA环境必备)
    multiprocessing.set_start_method("spawn", force=True) 

    # 2. 配置参数
    # 建议先用小数据或少量 worker 测试，跑通后再调大
    path = "/data2/public_data/qwen_mix_sft_128K" 
    data_args = DataArguments(
        preprocessing_num_workers=32, 
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

    # 6. 【验证环节 2】检查 Collator (模拟 DataLoader 行为)
    # 因为 Parquet 存的是 List，Dataset 输出的是 Tensor，Collator 需要能处理
    print("\n--- Collator Check ---")
    collator = PackedDataCollator(tokenizer=tokenizer)
    
    # 模拟一个 Batch
    batch_input = [dataset[1000]]
    # batch_output = collator(batch_input)
    breakpoint()
    print(f"Batch Input IDs Shape: {batch_input['input_ids'].shape}") # 应该是 [2, max_seq_len]
    print(f"Batch Labels Shape:    {batch_input['labels'].shape}")
    

    # breakpoint() # 如果需要手动调试解开此行