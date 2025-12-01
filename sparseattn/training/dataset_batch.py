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
    min_seq_len: Optional[int] = 1000
    task_type: str = "pretrain" 
    use_packing: bool = False
    data_cache_dir: Optional[str] = None


# =========================================================
#  Unified sample processor
# =========================================================
def process_sample(item, tokenizer, data_args, max_seq_len):
    """
    Unified task router.

    task_type:
      - "pretrain"
      - "sft"
      - future tasks: reward_model, dpo, preference, rlhf, etc.

    Output format must be consistent:
      {
        "input_ids": [...],              # token list
        "labels": [...],                 # optional
        "task_type": str,                # keep original
    }
    """

    task_type = data_args.task_type

    if task_type == "sft":
        input_ids, labels, attention_mask, segment_ids, range_ids, class_id = _build_sft_input_and_labels(item, tokenizer, data_args, max_seq_len)
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

    # -------- pretrain / others --------
    text = item.get("context") or item.get("text") or item.get("content")
    if text is None:
        raise KeyError("Item missing text field")

    tokenized = tokenizer(
        text, truncation=True, add_special_tokens=True
    )
    input_ids = tokenized["input_ids"]

    meta = item.get("metadata", {}) or {}
    task = meta.get("task", "default")

    # chunking (pretrain only)
    if (
        data_args.subsplit_length is not None
        and not data_args.single_seq
        and task_type != "sft"
    ):
        L = data_args.subsplit_length
        chunks = [
            input_ids[i : i + L]
            for i in range(0, len(input_ids), L)
            if len(input_ids[i : i + L]) > 0
        ]
        return {
            "input_ids_chunks": chunks,
            "task_type": task,
        }

    return {
        "input_ids": input_ids,
        "task_type": task,
    }


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
        self.all_class_ids = self._get_all_class_ids()
        
    def _get_all_class_ids(self):
        logger.info("Collecting all class IDs for stratified sampling...")

        CLASS_MAP = {
            'Single QA': 0,
            'MultiHop QA': 1,
            'Summarization': 2,
            'Code': 3
        }

        class_ids = []
        for idx in range(len(self.raw_dataset)):
            item = self.raw_dataset[idx]
            meta = item.get("metadata", {}) or {}
            if isinstance(meta, str):
                try:
                    meta = ast.literal_eval(meta)
                except Exception:
                    meta = {}

            task = meta.get("task", "Other")
            class_id = CLASS_MAP.get(task, 4)

            class_ids.append(class_id)

        return class_ids

    def get_class_indices(self):
        class_indices = {0: [], 1: [], 2: [], 3: []}
        for i, class_id in enumerate(self.all_class_ids):
            if class_id in class_indices:
                class_indices[class_id].append(i)
        return class_indices

    def __len__(self):
        return len(self.raw_dataset)

    def __getitem__(self, idx):
        return process_sample(
            self.raw_dataset[idx],
            self.tokenizer,
            self.data_args,
            self.max_seq_len,
        )


class StreamingParquetIterable(IterableDataset):
    def __init__(self, dataset_iterable, tokenizer, data_args, max_seq_len):
        self.dataset_iterable = dataset_iterable
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_seq_len = max_seq_len

    def __iter__(self):
        for item in self.dataset_iterable:
            yield process_sample(
                item,
                self.tokenizer,
                self.data_args,
                self.max_seq_len,
            )


# =========================================================
#  PrepackedDataset
# =========================================================
class PrepackedDataset(Dataset):
    def __init__(self, packed_input_ids, tokenizer, max_seq_len):
        self.packed_input_ids = packed_input_ids
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.packed_input_ids)

    def __getitem__(self, idx):
        seq = self.packed_input_ids[idx]
        if len(seq) > self.max_seq_len:
            seq = seq[: self.max_seq_len]
        return {"input_ids": seq}


# =========================================================
#  Collator
# =========================================================
class PackingDataCollator:
    def __init__(self, tokenizer, data_args, max_seq_len):
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.max_seq_len = max_seq_len

    # packing helper
    def _pack_sequences(self, seqs, labels=None):
        packed_ids = []
        packed_labels = []
        cur_ids = []
        cur_labels = []

        max_tokens = self.data_args.per_device_max_tokens or self.max_seq_len

        for idx, seq in enumerate(seqs):
            lab = labels[idx] if labels is not None else None

            if len(seq) > self.max_seq_len:
                seq = seq[: self.max_seq_len]
                if lab is not None:
                    lab = lab[: self.max_seq_len]

            pos = 0
            while pos < len(seq):
                remain = max_tokens - len(cur_ids)
                if remain <= 0:
                    packed_ids.append(cur_ids)
                    packed_labels.append(cur_labels)
                    cur_ids, cur_labels = [], []
                    remain = max_tokens

                take = min(len(seq) - pos, remain)
                chunk = seq[pos : pos + take]

                if lab is not None:
                    lab_chunk = lab[pos : pos + take]
                else:
                    lab_chunk = chunk.copy()
                    if not cur_ids:
                        lab_chunk[0] = -100

                if cur_ids:
                    lab_chunk[0] = -100

                cur_ids.extend(chunk)
                cur_labels.extend(lab_chunk)
                pos += take

                if len(cur_ids) == max_tokens:
                    packed_ids.append(cur_ids)
                    packed_labels.append(cur_labels)
                    cur_ids, cur_labels = [], []

        if cur_ids:
            packed_ids.append(cur_ids)
            packed_labels.append(cur_labels)

        return packed_ids, packed_labels

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
#  SFT builder
# =========================================================
# def _build_sft_input_and_labels(item, tokenizer, data_args, max_seq_len):
#     ctx = item.get("context", "") or ""
#     q = item.get("question", "") or ""
#     a = item.get("answer", "") or ""
#     meta = item.get("metadata", {}) or {}
#     flag = str(meta.get("flag", "0"))

#     if flag == "1":
#         prompt_text = q
#     else:
#         if ctx and q:
#             prompt_text = ctx.rstrip() + "\n" + q.lstrip()
#         else:
#             prompt_text = (ctx or q)

#     separator = "\n\n"
#     full = prompt_text + separator + a if a else prompt_text

#     ids = tokenizer(full, truncation=False, add_special_tokens=True)["input_ids"]
#     prompt_ids = tokenizer(prompt_text, add_special_tokens=True)["input_ids"]
#     prompt_len = len(prompt_ids)

#     labels = [-100] * len(ids)
#     if a:
#         sep_len = len(tokenizer(separator, add_special_tokens=False)["input_ids"])
#         start = prompt_len + sep_len
#         for i in range(start, len(ids)):
#             labels[i] = ids[i]

#     if len(ids) > max_seq_len:
#         ids = ids[:max_seq_len]
#         labels = labels[:max_seq_len]


#     return ids, labels
def _build_sft_input_and_labels(item, tokenizer, data_args, max_seq_len):
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
    task_ids = tokenizer(task_token, add_special_tokens=True)["input_ids"]

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
    full_input_ids = task_ids
    segment_ids = [0] * len(task_ids)
    current_len += len(task_ids)
    
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
    if tokenizer.eos_token_id is not None and (not full_input_ids or full_input_ids[-1] != tokenizer.eos_token_id):
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

        ctx_end = min(ctx_end, max_valid_index)
        q_end = min(q_end, max_valid_index)
        a_end = min(a_end, max_valid_index)

        if q_start > max_valid_index:
            q_start = q_end + 1 


        if ctx_start > max_valid_index:
            ctx_start = ctx_end + 1
            
        if a_start > max_valid_index:
            a_start = a_end + 1
        
    # labels = [-100] * len(full_input_ids)
    # if a_ids:
    #     labels[a_start:len(full_input_ids)] = full_input_ids[a_start:len(full_input_ids)]
    labels = full_input_ids.copy()
    
    # Range_ids: [ctx_start, ctx_end, q_start, q_end, a_start, a_end]
    range_ids = [ctx_start, ctx_end, q_start, q_end, a_start, a_end]
    
    padding_len = max_seq_len - len(full_input_ids)
    if padding_len > 0:

        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        full_input_ids.extend([pad_id] * padding_len)

        labels.extend([-100] * padding_len)

        segment_ids.extend([0] * padding_len)


    input_ids = torch.tensor(full_input_ids[:max_seq_len], dtype=torch.long)
    labels = torch.tensor(labels[:max_seq_len], dtype=torch.long)
    segment_ids = torch.tensor(segment_ids[:max_seq_len], dtype=torch.long)

    attention_mask = (input_ids != tokenizer.pad_token_id).long()

    range_ids_tensor = torch.tensor(range_ids, dtype=torch.long)
    
    CLASS_MAP = {'Single QA': 0, 'MultiHop QA': 1, 'Summarization': 2, 'Code': 3}
    class_id = CLASS_MAP.get(task_type, 4)
    class_id_tensor = torch.tensor(class_id, dtype=torch.long)
    
    return input_ids, labels, attention_mask, segment_ids, range_ids_tensor, class_id_tensor


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

    # streaming
    if data_args.streaming:
        ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
        return StreamingParquetIterable(ds, tokenizer, data_args, max_len)

    raw = load_dataset("parquet", data_files=parquet_files,split="train", cache_dir=os.path.join(data_args.data_cache_dir, "raw") if data_args.data_cache_dir else None)

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

    # prepack mode
    if data_args.prepack:
        all_ids = []
        for item in raw:
            r = process_sample(item, tokenizer, data_args, max_len)
            if "input_ids_chunks" in r:
                all_ids.extend(r["input_ids_chunks"])
            else:
                all_ids.append(r["input_ids"])

        collator = PackingDataCollator(tokenizer, data_args, max_len)
        packed, _ = collator._pack_sequences(all_ids)
        logger.info(f"Prepacked into {len(packed)} sequences.")
        return PrepackedDataset(packed, tokenizer, max_len)

    return ParquetDataset(raw, tokenizer, data_args, max_len, is_training)

from torch.utils.data.sampler import Sampler
import random
import math
import torch
import torch.distributed as dist
from typing import Dict, List, Iterator

class SamplerConditionError(ValueError):
    pass

class CustomDistributedStratifiedSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset,
        class_indices: Dict[int, List[int]],
        num_gpus: int = 8,
        required_per_class: int = 2,
        seed: int = 42,
    ):
        # -------- Distributed setup --------
        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = num_gpus

        self.seed = seed
        self.required_per_class = required_per_class
        self.num_classes = len(class_indices)
        self.class_indices = class_indices

        self.global_batch_size = self.num_classes * self.required_per_class

        if self.world_size != 8:
            raise SamplerConditionError(
                f"❌ Sampler Condition Failed: world_size={self.world_size} is not 8."
            )

        if self.global_batch_size != self.world_size:
            raise SamplerConditionError(
                f"❌ Sampler Condition Failed: global_batch_size={self.global_batch_size} (classes*{required_per_class}) "
                f"must equal world_size={self.world_size} for this specific sampler."
            )
        
        min_size = min(len(v) for v in class_indices.values())
        self.num_steps = min_size // required_per_class


        self.num_samples = self.num_steps
        if self.rank == 0:
            print(f"✅ Sampler Initialized: steps={self.num_steps}, samples per rank={self.num_samples}")

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        rng = random.Random(self.seed)

        # flatten all samples for other-class borrowing
        all_indices = []
        for cls, idxs in self.class_indices.items():
            all_indices.extend(idxs)

        final_rank_samples = []

        for step in range(self.num_steps):

            step_indices = []

            for cls, idx_list in self.class_indices.items():
                # shuffle index list
                rng.shuffle(idx_list)

                start = step * self.required_per_class
                end = start + self.required_per_class
                chunk = idx_list[start:end]

                if len(chunk) < self.required_per_class:
                    # need extra samples
                    need = self.required_per_class - len(chunk)

                    # borrow from other classes
                    candidates = [i for i in all_indices if i not in chunk]
                    rng.shuffle(candidates)

                    borrowed = candidates[:need]
                    chunk = chunk + borrowed

                step_indices.extend(chunk)

            # now step_indices has num_classes * required_per_class items
            if len(step_indices) != self.world_size:
                raise RuntimeError("step size mismatches world_size")

            rng.shuffle(step_indices)
            final_rank_samples.append(step_indices[self.rank])

        return iter(final_rank_samples)

