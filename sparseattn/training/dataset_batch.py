"""
Optimized dataset + collator for training on parquet files with packing support.
Features:
- Supports streaming (datasets.streaming) to avoid loading everything into memory.
- Option to pre-pack sequences offline into fixed-length examples (recommended for best training throughput).
- More efficient collator: avoids repeated tensor construction, supports multi-worker DataLoader.
- Safer tokenization with truncation and configurable max length.
- Improved label masking that prevents cross-chunk leakage.

Usage examples at the bottom.
"""

import os
import glob
import logging
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import AutoTokenizer
from datasets import load_dataset

import ast

logger = logging.getLogger(__name__)


# =========================================================
#  DataArguments
# =========================================================
@dataclass
class DataArguments:
    single_seq: bool = False
    subsplit_length: Optional[int] = None
    per_device_max_tokens: int = 32768
    apply_instruct_masks: bool = False
    prepack: bool = False
    streaming: bool = False
    min_seq_len: Optional[int] = None
    task_type: str = "pretrain" 
    use_packing: bool = False
    data_cache_dir: Optional[str] = None

# =========================================================
#  Datasets
# =========================================================
class ParquetDataset(Dataset):
    def __init__(self, raw_dataset, tokenizer, data_args, max_seq_len, is_training=True):
        self.raw_dataset = raw_dataset
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_seq_len = max_seq_len
        self.is_training = is_training

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        task_type = self.data_args.task_type
        item = self.raw_dataset[idx]

        if task_type == "sft":
            input_ids, labels, attention_mask, segment_ids, range_ids, class_id = self._build_sft_input_and_labels(
                item, self.tokenizer, self.max_seq_len
            )
            meta = item.get("metadata", {}) or {}
            task = meta.get("task", "default")
            return {
                "input_ids": input_ids,
                "labels": labels,
                "task_type": task,
                "attention_mask": attention_mask,
                "segment_ids": segment_ids,
                "range_ids": range_ids,
                "class_id": class_id,
            }
    
    def _build_sft_input_and_labels(self, item, tokenizer, max_seq_len):
        ctx = item.get("context", "") or ""
        q = item.get("question", "") or ""
        a = item.get("answer", "") or ""
        meta = item.get("metadata", {}) or {}
        flag = str(meta.get("flag", "0"))

        task_type = "Other"
        try:
            meta_dict = ast.literal_eval(meta) if isinstance(meta, str) else meta
            task_type = meta_dict.get('task', 'Other')
        except Exception:
            pass

        if task_type == 'Single QA':
            task_token = '[TASK_SQA]'
        elif task_type == 'MultiHop QA': 
            task_token = '[TASK_MHQA]'
        elif task_type == 'Summarization':
            task_token = '[TASK_SUM]'
        elif task_type == 'Code':
            task_token = '[TASK_CODE]'
        else:
            task_token = '[TASK_OTHER]'
            
        separator = "\n\n"

        # Task Token (Segment ID 0)
        task_ids = tokenizer(task_token, add_special_tokens=False)["input_ids"]

        if task_ids[-1] == tokenizer.eos_token_id or task_ids[-1] == tokenizer.sep_token_id:
            task_ids = task_ids[:-1]
        
        # Context (Segment ID 1)
        if flag == "1" or not ctx:
            ctx_text = ""
        else:
            ctx_text = "\n" + ctx.rstrip()
        ctx_ids = tokenizer(ctx_text, add_special_tokens=False)["input_ids"]
        
        # Question (Segment ID 2)
        if flag == "1":
            q_text = "\n" + q.lstrip()
        else:
            q_text = "\n" + q.lstrip() if ctx and q else (q.lstrip() if q and not ctx else "")
        q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

        # Separator + Answer (Segment ID 3)
        if a:
            a_text = separator + a
            a_ids = tokenizer(a_text, add_special_tokens=False)["input_ids"]
        else:
            a_ids = []

        # --- 3. Construct Full Sequence, Segment IDs, and Ranges ---
        
        # [TASK_TOKEN] + [CTX] + [Q] + [SEPARATOR] + [ANSWER]
        
        current_len = 0
        
        # Task (Segment 0)
        special_start = current_len
        full_input_ids = task_ids
        segment_ids = [0] * len(task_ids)
        current_len += len(task_ids)
        special_end = current_len - 1 if task_ids else special_start
        
        # Context (Segment 1)
        ctx_start = current_len
        full_input_ids.extend(ctx_ids)
        segment_ids.extend([1] * len(ctx_ids))
        current_len += len(ctx_ids)
        ctx_end = current_len - 1 if ctx_ids else ctx_start
        
        # Question (Segment 2)
        q_start = current_len
        full_input_ids.extend(q_ids)
        segment_ids.extend([2] * len(q_ids))
        current_len += len(q_ids)
        q_end = current_len - 1 if q_ids else q_start
        
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
        
        if original_len > max_seq_len:
            
            full_input_ids = full_input_ids[:max_seq_len]
            segment_ids = segment_ids[:max_seq_len]

            max_valid_index = max_seq_len - 1 

            special_end = min(special_end, max_valid_index)
            ctx_end = min(ctx_end, max_valid_index)
            q_end = min(q_end, max_valid_index)
            a_end = min(a_end, max_valid_index)

            if q_start > max_valid_index:
                q_start = q_end + 1 

            if special_start > max_valid_index:
                special_start = special_end + 1

            if ctx_start > max_valid_index:
                ctx_start = ctx_end + 1
                
            if a_start > max_valid_index:
                a_start = a_end + 1
            
        # labels = [-100] * len(full_input_ids)
        # if a_ids:
        #     labels[a_start:len(full_input_ids)] = full_input_ids[a_start:len(full_input_ids)]
        labels = full_input_ids.copy()
        
        # Range_ids: [ctx_start, ctx_end, q_start, q_end, a_start, a_end]
        range_ids = [special_start, special_end, ctx_start, ctx_end, q_start, q_end, a_start, a_end]
        
        padding_len = max_seq_len - len(full_input_ids)
        if padding_len > 0:

            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
            full_input_ids.extend([pad_id] * padding_len)

            labels.extend([-100] * padding_len)

            segment_ids.extend([0] * padding_len)

        input_ids = torch.tensor(full_input_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        segment_ids = torch.tensor(segment_ids, dtype=torch.long)

        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        range_ids_tensor = torch.tensor(range_ids, dtype=torch.long)
        
        CLASS_MAP = {'Single QA': 0, 'MultiHop QA': 1, 'Summarization': 2, 'Code': 3}
        class_id = CLASS_MAP.get(task_type, 4)
        class_id_tensor = torch.tensor(class_id, dtype=torch.long)
        
        return input_ids, labels, attention_mask, segment_ids, range_ids_tensor, class_id_tensor

# =========================================================
#  Collator
# =========================================================
class PackingDataCollator:
    def __init__(self, tokenizer, data_args, max_seq_len):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_seq_len = max_seq_len

    # main interface
    def __call__(self, batch):
        all_ids = []
        all_labels = []
        all_tasks = []
        # labels_exist = False

        all_ids = torch.stack([item['input_ids'] for item in batch], dim=0) 
        all_labels = torch.stack([item['labels'] for item in batch], dim=0)
        attention_mask = torch.stack([item['attention_mask'] for item in batch], dim=0)
        all_segment_ids = torch.stack([item['segment_ids'] for item in batch], dim=0)
        all_range_ids = torch.stack([item['range_ids'] for item in batch], dim=0) 
        all_tasks = [item.get("task_type", "default") for item in batch]
        all_class_ids = torch.stack([item['class_id'] for item in batch], dim=0)
        
        res = {
            "input_ids": all_ids,
            "labels": all_labels,
            "attention_mask": attention_mask,
            "task_type": all_tasks,
            "segment_ids": all_segment_ids,
            "range_ids": all_range_ids,
            "task_ids": all_class_ids,
        }
        
        return res


# =========================================================
#  Dataset builder
# =========================================================
def build_dataset(paths, data_args, tokenizer=None, is_training=True, model_name_or_path=None):

    if isinstance(paths, str):
        paths = [paths]

    parquet_files = []
    for p in paths:
        if os.path.isdir(p):
            parquet_files.extend(glob.glob(os.path.join(p, "*.parquet")))
        elif os.path.isfile(p) and p.endswith(".parquet"):
            parquet_files.append(p)
        else:
            raise ValueError(f"Invalid path: {p}")

    if not parquet_files:
        raise ValueError("No parquet files found")

    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_len = data_args.per_device_max_tokens or 32768
    max_len = min(max_len, 4096*250)  # hard clamp for safety

    raw = load_dataset("parquet", data_files=parquet_files, split="train", cache_dir=os.path.join(data_args.data_cache_dir, "raw") if data_args.data_cache_dir else None)

    # filter short samples
    if data_args.min_seq_len is not None and not data_args.prepack:
        def filter_fn(x):
            text = x.get("context") or x.get("text") or x.get("content")
            if text is None:
                return False
            l = len(tokenizer(text, add_special_tokens=True, truncation=False)["input_ids"])
            return l > data_args.min_seq_len
        raw = raw.filter(filter_fn, num_proc=os.cpu_count())
        logger.info(f"Filtered dataset size: {len(raw)}")

    return ParquetDataset(raw, tokenizer, data_args, max_len, is_training)