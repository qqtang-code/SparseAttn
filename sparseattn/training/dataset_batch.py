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
    task_type: str = "pretrain"   # <-- unified
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
        input_ids, labels, attention_mask = _build_sft_input_and_labels(item, tokenizer, data_args, max_seq_len)
        meta = item.get("metadata", {}) or {}
        task = meta.get("task", "default")
        return {
            "input_ids": input_ids,
            "labels": labels,
            "task_type": task,
            "attention_mask": attention_mask
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

        all_ids = torch.cat([item['input_ids'] for item in batch], dim=0)
        all_labels = torch.cat([item['labels'] for item in batch], dim=0)
        all_tasks = [item.get("task_type", "default") for item in batch]
        attention_mask = torch.cat([item['attention_mask'] for item in batch], dim=0)
        
        res = {
            "input_ids": all_ids,
            "labels": all_labels,
            "attention_mask": attention_mask,
            "task_type": all_tasks,
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
    elif task_type == 'MultiHop QA': # 注意检查数据里的具体拼写，如 "Multi-hop QA"
        task_token = '[TASK_MHQA]'
    elif task_type == 'Summarization':
        task_token = '[TASK_SUM]'
    elif task_type == 'Code':
        task_token = '[TASK_CODE]'
    else:
        task_token = '[TASK_OTHER]'
    # -------- 构造 full 文本 --------
    if flag == "1":
        prompt_content = q
    else:
        if ctx and q:
            prompt_content = ctx.rstrip() + "\n" + q.lstrip()
        else:
            prompt_content = (ctx or q)

    # Construct prompt_text
    prompt_text = task_token + "\n" + prompt_content.lstrip()
    
    separator = "\n\n"
    full_text = prompt_text + separator + a if a else prompt_text

    # -------- tokenize --------
    encoding = tokenizer(
        full_text,
        truncation=True,
        max_length=max_seq_len,
        padding="max_length",
        add_special_tokens=True,
        return_tensors="pt",
    )
    
    input_ids, attention_mask = encoding["input_ids"], encoding["attention_mask"]
    labels = torch.where(attention_mask == 1, input_ids, -100)

    # labels = [-100] * len(input_ids)

    # labels = [
    #     label if mask == 1 else -100
    #     for label, mask in zip(labels, attention_mask)
    # ]
    return input_ids, labels, attention_mask


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
    # if data_args.streaming:
    #     ds = load_dataset("parquet", data_files=parquet_files, split="train", streaming=True)
    #     return StreamingParquetIterable(ds, tokenizer, data_args, max_len)
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