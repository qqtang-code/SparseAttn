# coding=utf-8
# Copyright 2020-present the HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
The Trainer class, to easily train a 🤗 Transformers from scratch or finetune it on a new task.
"""

import time
from collections.abc import Mapping
from distutils.util import strtobool
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
from functools import partial
import math
import gc

from datasets import Dataset
import transformers
from transformers import Trainer as HFTrainer
from transformers.trainer import _get_fsdp_ckpt_kwargs
import inspect
# Integrations must be imported before ML frameworks:

import numpy as np
import torch
import torch.distributed as dist
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, SequentialSampler
from torch.optim.lr_scheduler import LambdaLR


from transformers import __version__
from transformers.trainer_callback import (
    PrinterCallback,
    TrainerCallback,
)
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    get_parameter_names,
)
from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length,
    seed_worker,
    PREFIX_CHECKPOINT_DIR,
)
from transformers.utils import (
    is_apex_available,
    is_sagemaker_mp_enabled,
)
from transformers.optimization import get_scheduler
import swanlab

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp  # type: ignore
    from smdistributed.modelparallel import __version__ as SMP_VERSION  # type: ignore

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")
else:
    IS_SAGEMAKER_MP_POST_1_10 = False


from transformers.trainer import logger
from streaming import StreamingDataLoader, StreamingDataset
import torch.distributed as dist
import datasets
import os
import json

from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import (
    enable_full_determinism,
    get_last_checkpoint,
    set_seed,
    find_executable_batch_size,
)
from transformers.trainer_pt_utils import reissue_pt_warnings
import warnings
import huggingface_hub.utils as hf_hub_utils
import glob

from accelerate.utils import load_fsdp_optimizer

# Name of the files used for checkpointing
TRAINING_ARGS_NAME = "training_args.bin"
TRAINER_STATE_NAME = "trainer_state.json"
OPTIMIZER_NAME = "optimizer.pt"
OPTIMIZER_NAME_BIN = "optimizer.bin"
SCHEDULER_NAME = "scheduler.pt"
SCALER_NAME = "scaler.pt"

from collections import defaultdict

class SparsityAccumulator:
    def __init__(self):
        # 新增：持久化存储，用于跨 Step 记忆各个任务的最后指标
        # 结构示例: {"Spa-Code sparsity": 0.5, "Spa-Code target_sparsity": 0.5, ...}
        self.last_known_metrics = {}
        self.reset()

    def reset(self):
        # 记录标量累积 (loss, reg_loss, etc.)
        self.scalars = defaultdict(float)
        self.scalar_counts = defaultdict(int)
        
        # 记录任务特定指标 (task_name -> {sum, count})
        self.task_metrics = defaultdict(lambda: {"sum_sparsity": 0.0, "sum_target": 0.0, "sum_z_loss": 0.0, "count": 0})
        
        # 注意：这里千万不要重置 self.last_known_metrics

    def add(self, scalars: dict, task_data: list):
        """
        纯本地累积，无分布式通信。
        scalars: {"loss": val, "reg_loss": val...}
        task_data: [(task_name, sparsity, target, z_loss), ...]
        """
        # 1. 累积全局标量
        for k, v in scalars.items():
            if v is not None:
                # 确保转为 python float，避免显存占用
                val = v.item() if isinstance(v, torch.Tensor) else v
                self.scalars[k] += val
                self.scalar_counts[k] += 1

        # 2. 累积任务特定指标
        for task_name, sp, tgt, z_loss in task_data:
            # 同样确保转为 float
            sp = sp.item() if isinstance(sp, torch.Tensor) else sp
            tgt = tgt.item() if isinstance(tgt, torch.Tensor) else tgt
            z_loss = z_loss.item() if isinstance(z_loss, torch.Tensor) else z_loss
            
            self.task_metrics[task_name]["sum_sparsity"] += sp
            self.task_metrics[task_name]["sum_target"] += tgt
            self.task_metrics[task_name]["sum_z_loss"] += z_loss
            self.task_metrics[task_name]["count"] += 1

    def compute_global_metrics(self, accelerator):
        """
        只在 Sync Step 调用一次。执行分布式聚合。
        """
        # --- 第一阶段：聚合标量 ---
        local_data = {
            "scalars": dict(self.scalars),
            "scalar_counts": dict(self.scalar_counts),
            "tasks": dict(self.task_metrics)
        }
        
        # 收集所有 GPU 的累积数据
        gathered_data = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered_data, local_data)
        
        # --- 第二阶段：合并所有 GPU 的数据 ---
        final_scalars = defaultdict(float)
        final_scalar_counts = defaultdict(int)
        final_tasks = defaultdict(lambda: {"sum_sparsity": 0.0, "sum_target": 0.0, "sum_z_loss": 0.0, "count": 0})
        
        for gpu_data in gathered_data:
            # 合并标量
            for k, v in gpu_data["scalars"].items():
                final_scalars[k] += v
                final_scalar_counts[k] += gpu_data["scalar_counts"][k]
            
            # 合并任务
            for t_name, t_stats in gpu_data["tasks"].items():
                final_tasks[t_name]["sum_sparsity"] += t_stats["sum_sparsity"]
                final_tasks[t_name]["sum_target"] += t_stats["sum_target"]
                final_tasks[t_name]["sum_z_loss"] += t_stats["sum_z_loss"]
                final_tasks[t_name]["count"] += t_stats["count"]

        # --- 第三阶段：计算当前 Step 的平均值 ---
        metrics = {}
        
        # 计算标量平均
        for k in final_scalars:
            if final_scalar_counts[k] > 0:
                metrics[k] = final_scalars[k] / final_scalar_counts[k]
                
        # 计算任务平均
        for t_name, t_stats in final_tasks.items():
            if t_stats["count"] > 0:
                cnt = t_stats["count"]
                # 生成 Key 名称
                k_sp = f"Spa-{t_name} sparsity"
                k_tgt = f"Spa-{t_name} target_sparsity"
                k_z = f"Spa-{t_name} log_z_loss"

                # 计算值
                v_sp = t_stats["sum_sparsity"] / cnt
                v_tgt = t_stats["sum_target"] / cnt
                v_z = t_stats["sum_z_loss"] / cnt
                
                # 放入 metrics
                metrics[k_sp] = v_sp
                metrics[k_tgt] = v_tgt
                metrics[k_z] = v_z

                # [核心逻辑] 更新持久化记忆
                self.last_known_metrics[k_sp] = v_sp
                self.last_known_metrics[k_tgt] = v_tgt
                self.last_known_metrics[k_z] = v_z

        # --- 第四阶段：前向填充 (Forward Filling) ---
        # 遍历记忆中的所有 Key，如果当前 metrics 里没有，就补上上次的值
        for k, v in self.last_known_metrics.items():
            if k not in metrics:
                metrics[k] = v
                
        return metrics

class LogCallback(TrainerCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.start_time = None
        self.last_log_time = None
        self.log_time_interval = 0
        self.is_training = False

        self.max_steps = -1
        self.first_step_of_run = 0

    def on_step_begin(self, args, state, control, **kwargs):
        if state.is_world_process_zero and self.last_log_time is None:
            self.log_time_interval = getattr(args, "log_time_interval", 0)
            if self.log_time_interval > 0:
                logger.info(
                    f"Using log_time_interval {self.log_time_interval} s. This will override logging_steps and logging_strategy."
                )
                args.logging_steps = 1
                args.logging_strategy = "steps"

            self.last_step = 0

            self.start_time = time.time()
            self.last_log_time = self.start_time
            self.max_steps = state.max_steps
            self.first_step_of_run = state.global_step

            self.last_tokens_seen = state.num_input_tokens_seen

    def on_log(self, args, state, control, logs=None, **kwargs):
        _ = logs.pop("total_flos", None)

        if state.is_world_process_zero:
            if self.is_training:
                current_time = time.time()
                time_diff = current_time - self.last_log_time
                force = logs.get("force", False)

                if (
                    time_diff > self.log_time_interval
                    or state.global_step >= self.max_steps - 1
                    or force
                ):
                    self.last_log_time = current_time
                    steps_completed = max(state.global_step, 1)

                    steps_since_first = max(
                        1, state.global_step - self.first_step_of_run
                    )
                    self.last_step = state.global_step

                    tokens_seen_since_last = (
                        state.num_input_tokens_seen - self.last_tokens_seen
                    ) // args.seq_parallel_size
                    self.last_tokens_seen = state.num_input_tokens_seen

                    remaining_steps = self.max_steps - steps_completed
                    pct_completed = (steps_completed / self.max_steps) * 100
                    time_since_start = current_time - self.start_time
                    remaining_time = (
                        time_since_start / steps_since_first
                    ) * remaining_steps

                    gpu_mem_free, _ = torch.cuda.mem_get_info(device=args.device)

                    update = {
                        "completed": f"{pct_completed:.2f}% ({steps_completed:_} / {self.max_steps:_})",
                        "remaining time": self.format_duration(remaining_time),
                        "throughput": f"{tokens_seen_since_last / time_diff:.2f}",
                        "gpu_mem_free": f"{gpu_mem_free / 1024 / 1024:.0f}MB",
                    }

                    logs_with_step = {**logs, **update}
                    if "step" not in logs_with_step:
                        logs_with_step["step"] = state.global_step

                    logger.info(str(logs_with_step))
            else:
                logger.info(str(logs))

    def on_train_begin(self, args, state, control, **kwargs):
        args.include_num_input_tokens_seen = True

    def on_step_end(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.is_training = True

    def on_prediction_step(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            self.is_training = False

    @staticmethod
    def format_duration(seconds):
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{int(hours)}:{int(minutes):02}:{int(seconds):02}"


import signal
from subprocess import call


class SIGUSR1Callback(transformers.TrainerCallback):
    def __init__(self, trainer) -> None:
        super().__init__()
        self.signal_received = False
        signal.signal(signal.SIGUSR1, self.handle_signal)
        # signal.signal(signal.SIGINT, self.handle_signal)
        logger.warning("Handler registered")
        self.trainer = trainer

    def handle_signal(self, signum, frame):
        self.signal_received = True
        logger.warning("Stop signal received...")

    def on_substep_end(self, args, state, control, **kwargs):
        if self.signal_received:
            self.trainer._save_checkpoint(
                self.trainer.model, None
            )  # Note that here _save_checkpoint does not actually use this, so we can just pass on any model
            # The reason we don't set should_save but instead directly save here
            # is that streaming may collapse after receiving the signal and it
            # would be too late to wait till the save function is called.
            # Same reason for why we handle the single in both on_substep_end
            # and on_step_end, even though ideally we want to do on_step_end.
            # control.should_save = True
            control.should_training_stop = True

    def on_step_end(self, args, state, control, **kwargs):
        if self.signal_received:
            self.trainer._save_checkpoint(self.trainer.model, None)
            # control.should_save = True
            control.should_training_stop = True

    def on_train_end(self, args, state, control, **kwargs):
        if self.signal_received:
            exit(0)


def min_lr_bound(
    current_step: int,
    wrapped_func: Callable[[float], float],
    min_lr_ratio: float,
    warmup_steps: int,
):
    if current_step < warmup_steps:
        return wrapped_func(current_step)
    return min_lr_ratio + wrapped_func(current_step) * (1.0 - min_lr_ratio)


# - Callbacks: transformers.trainer_callback.DefaultFlowCallback, transformers.integrations.WandbCallback, transformers.trainer_callback.ProgressCallback
class Trainer(HFTrainer):
    def __init__(self, 
        model=None, 
        args=None, 
        data_collator=None, 
        train_dataset=None, 
        eval_dataset=None, 
        tokenizer=None, 
        model_init=None, 
        compute_metrics=None, 
        callbacks=None, 
        optimizers=(None, None), 
        preprocess_logits_for_metrics=None,
        **kwargs):
        # 1. 安全地提取自定义参数，并不再传给父类
        self.log_loss = kwargs.pop("log_loss", False)

        # 2. 显式调用父类初始化，明确指定关键参数 (特别是 tokenizer)
        # 这样可以防止 log_loss 或其他参数被误识别为 processing_class
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            **kwargs
        )
        self.start_head_sparsity = args.start_head_sparsity
        self.end_head_sparsity = args.end_head_sparsity
        self.learning_rate = args.learning_rate
        self.mask_learning_rate = args.mask_learning_rate
        self.reg_learning_rate = args.reg_learning_rate
        self.warmup_type = args.warmup_type
        self.sparsity_warmup_ratio = args.sparsity_warmup_ratio
        self.disable_linear_regularization_term = args.disable_linear_regularization_term
        self.context_window_if_toggled = args.context_window_if_toggled
        self.freeze_non_mask_parameters = args.freeze_non_mask_parameters
        self.freeze_mask_parameters = args.freeze_mask_parameters
        self.num_sparsity_warmup_steps = math.ceil(self.sparsity_warmup_ratio * self.args.max_steps)
        self.task_sparsity_config = {
            "default": {"start": self.start_head_sparsity, "end": self.end_head_sparsity},
            "Code": {"start": 0, "end": 0.9},
            "In-Context Learning": {"start": 0, "end": 0.9},
            "MultiHop QA": {"start": 0, "end": 0.6},
            "Single QA": {"start": 0, "end": 0.6},
            "Summarization": {"start": 0, "end": 0.9},
        }
        self.reverse_class_map = {0: 'Single QA', 1: 'MultiHop QA', 2: 'Summarization', 3: 'Code', 4:'In-Context Learning'}
        self.use_softmax = args.use_softmax
        
        self.tau_max = 1.5  # 初始 tau
        self.tau_min = 1  # 最终/保持 tau

        self.tau_decay_steps = math.ceil(self.args.max_steps * 0.6)

        if not dist.is_initialized() or args.seq_parallel_size == dist.get_world_size():
            logger.info(f"Using world as sequence parallel group")
            self.seq_parallel_group = dist.group.WORLD
        else:
            logger.info(
                f"Initializing sequence parallel groups with size {args.seq_parallel_size}"
            )
            self.seq_parallel_group, _ = dist.new_subgroups(args.seq_parallel_size)

        try:
            self.remove_callback(PrinterCallback)
            self.add_callback(LogCallback)
            self.add_callback(SIGUSR1Callback(self))
        except ValueError:
            logger.warning("Couldn't remove PrinterCallback")
        
        self.sparsity_accumulator = SparsityAccumulator()

    def get_current_target_sparsity(self, global_step: int, tasks: List[str]) -> torch.Tensor:
        """
        Returns: torch.Tensor of shape [len(tasks)], device=cpu (will be moved later)
        """
        sparsities = []
        for task in tasks:
            cfg = self.task_sparsity_config.get(task, self.task_sparsity_config["default"])
            start_sp = cfg["start"]
            end_sp = cfg["end"]
            
            sp = end_sp
            sparsities.append(sp)

        return torch.tensor(sparsities, dtype=torch.float32)  # shape: [B]
    
    def get_current_tau(self, global_step: int) -> torch.Tensor:
        
        tau_max = self.tau_max
        tau_min = self.tau_min
        T_max = self.tau_decay_steps
        
        t = torch.tensor(global_step, dtype=torch.float32)
        T_max_tensor = torch.tensor(T_max, dtype=torch.float32)
        
        if global_step < T_max:
            cosine_factor = torch.cos(torch.pi * t / T_max_tensor) 

            current_tau = tau_min + 0.5 * (tau_max - tau_min) * (1 + cosine_factor)
            return current_tau.to(torch.float32)
        else:
            # 保持 0.2
            return torch.tensor(tau_min, dtype=torch.float32)
        
    def get_sequence_parallel_inputs(self, inputs):
        """
        Args:
            inputs: Dict from DataCollator.
                - input_ids: [1, S] -> 需要变成 [S] 然后切分
                - labels: [1, S] -> 需要变成 [S] 然后切分
                - seq_lengths: List[Tensor] (len=1, tensor=[Bi+1]) -> 取出变成 [Bi+1], 保持全局
        """
        seq_parallel_world_size = (dist.get_world_size(self.seq_parallel_group) if dist.is_initialized() else 1)
        input_ids = inputs["input_ids"]
        if len(input_ids.shape) == 2:
            input_ids = inputs["input_ids"].squeeze(0) 
            labels = inputs["labels"].squeeze(0)
            
            shifted_labels = labels.roll(-1, dims=-1)
            shifted_labels[..., -1] = -100
            
            global_seq_lengths = inputs["seq_lengths"][0] 
            task_ids = inputs["task_ids"][0]
            range_ids = inputs["range_ids"][0]
            task_type = [i[0] for i in inputs['task_type']]
            
            raw_position_ids = inputs.get("position_ids", None)
            if raw_position_ids is not None:
                raw_position_ids = raw_position_ids.squeeze(0)
        else:
            global_seq_lengths = inputs["seq_lengths"]
        if seq_parallel_world_size > 1:
            seq_parallel_world_size = dist.get_world_size(self.seq_parallel_group)
            seq_parallel_rank = dist.get_rank(self.seq_parallel_group)

            total_seq_len = input_ids.size(0)

            # Padding (保证被 world_size 整除)
            if total_seq_len % seq_parallel_world_size != 0:
                padding = seq_parallel_world_size - (total_seq_len % seq_parallel_world_size)
                padding_zeros = torch.full((padding,), 0, dtype=input_ids.dtype, device=input_ids.device)
                
                input_ids = torch.cat([input_ids, padding_zeros], dim=0)
                shifted_labels = torch.cat([shifted_labels, padding_zeros - 100], dim=0)

            input_ids_chunks = torch.tensor_split(input_ids, seq_parallel_world_size, dim=0)
            shifted_labels_chunks = torch.tensor_split(shifted_labels, seq_parallel_world_size, dim=0)

            inputs = {
                "input_ids": input_ids_chunks[seq_parallel_rank],    # Local chunk
                "shifted_labels": shifted_labels_chunks[seq_parallel_rank],  
                "seq_lengths": global_seq_lengths,                   # Global
                "seq_parallel_group": self.seq_parallel_group,
                "range_ids": range_ids,
                "task_type": task_type,
                "task_ids": task_ids,
            }
        else:
            inputs = {
                "input_ids": input_ids,             # Global [S]
                "labels": labels,                   # Global [S]
                "task_ids": task_ids,
                "range_ids": range_ids,
                "seq_lengths": global_seq_lengths,  # Global [Bi+1]
                "task_type": task_type,
                "seq_parallel_group": None
            }

        return inputs

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,
        return_output_and_metrics=False,
    ):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        inputs = self.get_sequence_parallel_inputs(inputs)
        tasks = inputs.get("task_type", ["default"] * inputs["range_ids"].size(0)) #[B]
        # tasks = ["default"] * inputs["input_ids"].size(0)
        target_sparsity = self.get_current_target_sparsity(self.state.global_step, tasks)
        target_sparsity = target_sparsity.to(model.device)  # [B]
        current_tau = self.get_current_tau(self.state.global_step)
        
        seq_lens_display = []
        if "seq_lengths" in inputs and inputs["seq_lengths"] is not None:
            # inputs["seq_lengths"] 是 cu_seqlens (e.g. [0, 512, 1024])
            # 需要转为 list 并计算差值得到每个 sequence 的实际长度
            cu_seqlens = inputs["seq_lengths"]
            if isinstance(cu_seqlens, torch.Tensor):
                cu_seqlens = cu_seqlens.tolist()
            
            if len(cu_seqlens) > 1:
                seq_lens_display = [cu_seqlens[i+1] - cu_seqlens[i] for i in range(len(cu_seqlens)-1)]
            else:
                seq_lens_display = ["Unknown"]
        else:
            # Fallback: 如果不是 Packing 模式，长度就是 input_ids 的长度
            seq_lens_display = [inputs["input_ids"].shape[0]]

        # 优化后的 Print
        print(f"[Step {self.state.global_step} / Rank {dist.get_rank()}] "
              f"Tasks: {tasks} | Lens: {seq_lens_display} "
              f"→ Tgt Spa: {[f'{s:.3f}' for s in target_sparsity.tolist()]}")
        
        outputs = model(**inputs, use_cache=False, target_sparsity=target_sparsity, current_tau=current_tau)

        lm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        reg_loss = outputs["sparsity_loss"] if isinstance(outputs, dict) else outputs[-2]
        
        head_entropy = outputs["head_entropy"]
        model_sparsity = outputs["model_sparsity"] # [B] 或者是标量
        out_target_sparsity = outputs["target_sparsity"] # [B]
        log_z_loss = outputs['log_z_loss'] # [B]
        task_ids = outputs['task_ids'] # [B]
        
        if getattr(self.args, "token_scaled_loss", False):
            seq_parallel_world_size = (
                dist.get_world_size(self.seq_parallel_group)
                if dist.is_initialized()
                else 1
            )
            if seq_parallel_world_size > 1:  # Sequence parallelism
                device_num_valid_tokens = (
                    (inputs["shifted_labels"] != -100).sum().float()
                )
            else:
                device_num_valid_tokens = (inputs["labels"] != -100).sum().float()

            avg_device_num_valid_tokens = torch.mean(
                self.accelerator.gather(device_num_valid_tokens)
            ).item()
            if not hasattr(self.state, "count_step_for_num_valid_tokens"):
                self.state.count_step_for_num_valid_tokens = 1
                self.state.avg_num_valid_tokens_per_device = avg_device_num_valid_tokens
            else:
                self.state.count_step_for_num_valid_tokens += 1
                steps = self.state.count_step_for_num_valid_tokens
                self.state.avg_num_valid_tokens_per_device = (
                    self.state.avg_num_valid_tokens_per_device * ((steps - 1) / steps)
                    + avg_device_num_valid_tokens / steps
                )  # moving avg
            lm_loss = lm_loss / self.state.avg_num_valid_tokens_per_device

        loss = lm_loss + reg_loss
       
        
        # A. 准备本地数据 (不进行 gather!)
        # 提取当前微步(Micro-Step)的标量值
        local_scalars = {
            "loss": loss.detach(),
            "lm_loss": lm_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "head_entropy": head_entropy.detach(),
            # 如果 model_sparsity 是向量，这里先取平均作为整体指标，详细的任务拆分在下面
            "model_sparsity(avg)": model_sparsity.detach().float().mean(), 
            "target_sparsity(avg)": out_target_sparsity.detach().float().mean(),
        }

        # B. 准备任务特定数据 (Zip 并不是耗时操作)
        # 注意：这里我们只处理本 GPU 上的数据，坚决不做 all_gather
        local_task_data = []
        if task_ids.numel() > 0: # 确保有数据
            # 假设这些 tensor 维度是一致的 [B]
            t_ids = task_ids.detach().cpu().tolist()
            t_sparsities = model_sparsity.detach().float().tolist()
            t_targets = out_target_sparsity.detach().float().tolist()
            t_z_losses = log_z_loss.detach().float().tolist()
            
            for i in range(len(t_ids)):
                t_name = self.reverse_class_map.get(t_ids[i], "Unknown")
                local_task_data.append((t_name, t_sparsities[i], t_targets[i], t_z_losses[i]))

        # C. 加入累积器 (极快，纯内存操作)
        self.sparsity_accumulator.add(local_scalars, local_task_data)

        # D. 判断是否是 Global Step 的最后一步 (Sync Gradients)
        # 只有在这一步，我们才进行昂贵的通信和日志打印
        if self.accelerator.gradient_state.sync_gradients:
            
            # 只有当需要记录日志时才计算 (step > 0 且满足配置)
            if getattr(self.args, "log_train_sparsity_metrics", True):
                
                # [Trigger Communication] 计算 Global Mean
                # 这一步会发生 Block，但每 48 个微步才发生一次，开销可忽略
                global_metrics = self.sparsity_accumulator.compute_global_metrics(self.accelerator)
                
                # 只有主进程负责打印
                if self.accelerator.is_main_process:
                    # 补充一些只需取最后一次值的指标
                    global_metrics["step"] = self.state.global_step
                    global_metrics["current_tau"] = current_tau.detach().item()
                    
                    # Lambda 诊断信息
                    lambda1 = outputs.get("lambda1", None) if isinstance(outputs, dict) else None
                    if lambda1 is not None:
                        global_metrics.update({
                            "lambda1 Single QA": lambda1[0].detach().item(),
                            "lambda2 MultiHop QA": lambda1[1].detach().item(),
                            "lambda3 Summarization": lambda1[2].detach().item(),
                            "lambda4 Code": lambda1[3].detach().item(),
                        })

                    # --- 打印文本日志 ---
                    logger.info(
                        f"@ {self.state.global_step} | "
                        f"Loss: {global_metrics.get('loss', 0):.4f} | "
                        f"LM: {global_metrics.get('lm_loss', 0):.4f} | "
                        f"Reg: {global_metrics.get('reg_loss', 0):.4f} | "
                        f"Spa(Avg): {global_metrics.get('model_sparsity(avg)', 0):.3f}"
                    )
                    
                    # 打印任务日志
                    for k in sorted(global_metrics.keys()):
                        if k.startswith("Spa-") and "sparsity" in k and "target" not in k:
                            task = k.split(" ")[0] # e.g. "Spa-Code"
                            logger.info(
                                f"Statistic -> {task[4:]} | " # remove "Spa-"
                                f"Spa: {global_metrics[k]:.3f} | "
                                f"Tgt: {global_metrics.get(f'{task} target_sparsity', 0):.3f} | "
                                f"Z-Loss: {global_metrics.get(f'{task} log_z_loss', 0):.3f}"
                            )

                    # --- 上传 SwanLab ---
                    try:
                        swanlab.log(global_metrics)
                        import json
                        logger.info(f"[Micro-Log] {json.dumps(global_metrics, ensure_ascii=False)}")
                    except Exception:
                        pass

            # E. [重置累积器] 准备迎接下一个 Global Step
            self.sparsity_accumulator.reset()

        # =================================================================

        if return_outputs:
            return (loss, outputs)
        else:
            return loss

    def create_optimizer(self):
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            # 获取需要 weight decay 的参数名（通常排除 bias 和 layer norm）
            decay_parameters = self.get_decay_parameter_names(opt_model)

            # 定义 5 个参数组
            optimizer_1_group = []  # QKV params, non-decay
            optimizer_2_group = []  # QKV params, decay
            optimizer_3_group = []  # log_alpha params
            optimizer_4_group = []  # sparsity_lambda params
            optimizer_5_group = []  # mask_allocator params
            
            optimized_parameters = []
            
            # 定义我们只想训练的特定层后缀
            target_submodules = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]

            for n, p in opt_model.named_parameters():
                # --- 1. 处理 Mask 相关参数 ---
                if ".mask_allocator." in n or "mask_allocator" in n or "log_alpha" in n or "sparsity_lambda" in n:
                    if self.freeze_mask_parameters:
                        p.requires_grad = False
                    else:
                        p.requires_grad = True
                        if "log_alpha" in n:
                            optimizer_3_group.append(p)
                        elif "sparsity_lambda" in n:
                            optimizer_4_group.append(p)
                        else:
                            optimizer_5_group.append(p)
                        optimized_parameters.append(n)
                    continue

                # --- 2. 处理主体网络参数 (仅保留 Q, K, V) ---
                is_qkv = any(target in n for target in target_submodules)

                if is_qkv and not self.freeze_non_mask_parameters:
                    p.requires_grad = True
                    if n in decay_parameters:
                        optimizer_2_group.append(p)
                    else:
                        optimizer_1_group.append(p)
                    optimized_parameters.append(n)
                else:
                    # 核心修复：必须显式设为 False，否则 FSDP 会报错
                    p.requires_grad = False

            optimizer_grouped_parameters = []

            if not self.freeze_non_mask_parameters:
                if optimizer_1_group:
                    optimizer_grouped_parameters.append({
                        "params": optimizer_1_group,
                        "weight_decay": 0.0,
                        "lr": self.learning_rate,
                    })
                if optimizer_2_group:
                    optimizer_grouped_parameters.append({
                        "params": optimizer_2_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.learning_rate,
                    })

            if not self.freeze_mask_parameters:
                if optimizer_3_group:
                    optimizer_grouped_parameters.append({
                        "params": optimizer_3_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.mask_learning_rate,
                    })
                if optimizer_4_group:
                    optimizer_grouped_parameters.append({
                        "params": optimizer_4_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.reg_learning_rate,
                        "maximize": True,
                    })
                if optimizer_5_group:
                    optimizer_grouped_parameters.append({
                        "params": optimizer_5_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.mask_learning_rate,
                    })

            logger.info(f"Optimizing {len(optimized_parameters)} parameters.")
            logger.info(f"Optimized parameters list: {optimized_parameters}")

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(self.args, opt_model)
            
            if not optimizer_grouped_parameters:
                raise ValueError("No parameters to optimize! Check your freeze logic.")

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes
                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        manager.register_module_override(module, "weight", {"optim_bits": 32})

        if is_sagemaker_mp_enabled():
            import smdistributed.modelparallel.torch as smp
            self.optimizer = smp.DistributedOptimizer(self.optimizer)

        return self.optimizer

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
        """

        self.lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        if self.args.min_lr_ratio != 0.0:
            if isinstance(self.lr_scheduler, LambdaLR):
                lr_lambdas = self.lr_scheduler.lr_lambdas
                new_lr_lambdas = [
                    lr_lambda
                    if lr_lambda is None
                    or isinstance(lr_lambda, partial)
                    and lr_lambda.func == min_lr_bound
                    else partial(
                        min_lr_bound,
                        wrapped_func=lr_lambda,
                        min_lr_ratio=self.args.min_lr_ratio,
                        warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    )
                    for lr_lambda in lr_lambdas
                ]

                self.lr_scheduler.lr_lambdas = new_lr_lambdas
            else:
                raise NotImplementedError("Only LambdaLR is supported for min_lr_ratio")

        return self.lr_scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None:
            return None

        if self.args.ordered:
            return SequentialSampler(self.train_dataset)

        return super()._get_train_sampler()

    def get_train_dataloader(self):    
        if self.train_dataloader is not None:
            return self.train_dataloader

        return super().get_train_dataloader()

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Perform an evaluation step on `model` using `inputs`.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to evaluate.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.
            prediction_loss_only (`bool`):
                Whether or not to return the loss only.
            ignore_keys (`Lst[str]`, *optional*):
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions.

        Return:
            Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
            logits and labels (each being optional).
        """
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raise ValueError(
                    "SageMaker Model Parallelism is not supported in BaseTrainer"
                )
            else:
                with self.compute_loss_context_manager():
                    loss, outputs, metrics = self.compute_loss(
                        model, inputs, return_output_and_metrics=True
                    )
                if loss is not None:
                    loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(
                        v for k, v in outputs.items() if k not in ignore_keys + ["loss"]
                    )
                else:
                    logits = outputs[1:]

        if prediction_loss_only:
            return (loss, None, None, metrics)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels, metrics)

    def compute_loss_context_manager(self):
        """
        A helper wrapper to group together context managers.
        """
        if self.args.cuda_empty_cache:
            gc.collect()
            torch.cuda.empty_cache()
        return self.autocast_smart_context_manager()

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        args = self.args

        prediction_loss_only = (
            prediction_loss_only
            if prediction_loss_only is not None
            else args.prediction_loss_only
        )

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=False)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        if has_length(dataloader):
            logger.info(f"  Num examples = {self.num_examples(dataloader)}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None
        metrics_host = None

        metrics_names = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        all_metrics = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            # Prediction step
            loss, logits, labels, metrics = self.prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )
            inputs_decode = (
                self._prepare_input(inputs["input_ids"])
                if args.include_inputs_for_metrics
                else None
            )

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = (
                    losses
                    if losses_host is None
                    else nested_concat(losses_host, losses, padding_index=-100)
                )
            if labels is not None:
                labels = self.accelerator.pad_across_processes(
                    labels, dim=1, pad_index=-100
                )
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(
                    inputs_decode, dim=1, pad_index=-100
                )
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if metrics is not None:
                if metrics_names is None:
                    metrics_names = list(metrics.keys())
                else:
                    assert metrics_names == list(metrics.keys()), (
                        "Metrics should have the same keys across batches"
                    )

                metrics = [
                    metric if metric.shape else metric.repeat(batch_size)
                    for metric in metrics.values()
                ]
                metrics = self.accelerator.pad_across_processes(
                    metrics, dim=1, pad_index=float("nan")
                )
                metrics = self.accelerator.gather_for_metrics(metrics)
                metrics_host = (
                    metrics
                    if metrics_host is None
                    else nested_concat(
                        metrics_host, metrics, padding_index=float("nan")
                    )
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(
                    logits, dim=1, pad_index=-100
                )
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = (
                    logits
                    if preds_host is None
                    else nested_concat(preds_host, logits, padding_index=-100)
                )

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = (
                    labels
                    if labels_host is None
                    else nested_concat(labels_host, labels, padding_index=-100)
                )

            self.control = self.callback_handler.on_prediction_step(
                args, self.state, self.control
            )

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = (
                        losses
                        if all_losses is None
                        else np.concatenate((all_losses, losses), axis=0)
                    )
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = (
                        logits
                        if all_preds is None
                        else nested_concat(all_preds, logits, padding_index=-100)
                    )
                if metrics_host is not None:
                    metrics = nested_numpify(metrics_host)
                    all_metrics = (
                        metrics
                        if all_metrics is None
                        else nested_concat(
                            all_metrics, metrics, padding_index=float("nan")
                        )
                    )
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(
                            all_inputs, inputs_decode, padding_index=-100
                        )
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels
                        if all_labels is None
                        else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = (
                    None,
                    None,
                    None,
                    None,
                )

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = (
                losses
                if all_losses is None
                else np.concatenate((all_losses, losses), axis=0)
            )
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = (
                logits
                if all_preds is None
                else nested_concat(all_preds, logits, padding_index=-100)
            )
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode
                if all_inputs is None
                else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if metrics_host is not None:
            metrics = nested_numpify(metrics_host)
            all_metrics = (
                metrics
                if all_metrics is None
                else nested_concat(all_metrics, metrics, padding_index=float("nan"))
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = (
                labels
                if all_labels is None
                else nested_concat(all_labels, labels, padding_index=-100)
            )

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif (
            isinstance(eval_dataset, IterableDatasetShard)
            and getattr(eval_dataset, "num_examples", 0) > 0
        ):
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
        # samplers has been rounded to a multiple of batch_size, so we truncate.
        if all_losses is not None:
            all_losses = all_losses[:num_samples]
        if all_preds is not None:
            all_preds = nested_truncate(all_preds, num_samples)
        if all_labels is not None:
            all_labels = nested_truncate(all_labels, num_samples)
        if all_inputs is not None:
            all_inputs = nested_truncate(all_inputs, num_samples)
        # if all_metrics is not None:
        #     all_metrics = nested_truncate(all_metrics, num_samples)

        # Metrics!
        if (
            self.compute_metrics is not None
            and all_preds is not None
            and all_labels is not None
        ):
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(
                        predictions=all_preds, label_ids=all_labels, inputs=all_inputs
                    )
                )
            else:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels)
                )
        else:
            metrics = {}

        if all_metrics is not None:
            for key, value in zip(metrics_names, all_metrics):
                valid = ~np.isnan(value)
                metrics[key] = value[valid].mean().item()
                metrics[f"{key}___samples"] = np.sum(valid).item()

        metrics["samples"] = num_samples

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = (
                self.jit_compilation_time
            )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(
            predictions=all_preds,
            label_ids=all_labels,
            metrics=metrics,
            num_samples=num_samples,
        )

    def evaluate(
        self,
        eval_dataset: Optional[Union[Dict[str, Dataset], Dataset]] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        if eval_dataset is None:
            eval_dataset = self.eval_dataset

        if isinstance(eval_dataset, dict):
            metrics = {}
            for key, dataset in eval_dataset.items():
                metrics.update(
                    super().evaluate(
                        dataset,
                        ignore_keys=ignore_keys,
                        metric_key_prefix=f"{metric_key_prefix}_{key}",
                    )
                )
        else:
            metrics = super().evaluate(
                eval_dataset,
                ignore_keys=ignore_keys,
                metric_key_prefix=metric_key_prefix,
            )

        return metrics


    def _save_checkpoint(self, model, trial, metrics=None):
        # A wrapper around the original _save_checkpoint function to save streaming dataset state
        # Save model checkpoint
        sig = inspect.signature(super()._save_checkpoint)
        if "metrics" in sig.parameters:
            super()._save_checkpoint(model, trial, metrics=metrics)
        else:
            super()._save_checkpoint(model, trial)

        # Get the path
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)

        # Save streaming dataset state
        if (
            isinstance(self.train_dataset, StreamingDataset)
            and self.state.is_world_process_zero
        ):
            num_samples = (
                self.state.global_step
                * self.args.train_batch_size
                * self.args.world_size
                * self.args.gradient_accumulation_steps
            )
            if self.train_dataset.replication is not None:
                num_samples = num_samples // self.train_dataset.replication
            dataset_state_dict = self.train_dataset.state_dict(num_samples, True)
            logger.warning(f"Save streaming dataset state: {dataset_state_dict}")
            json.dump(
                dataset_state_dict,
                open(os.path.join(output_dir, "streaming_dataset_state.json"), "w"),
            )

    def _load_optimizer_and_scheduler(self, checkpoint):
        # A wrapper around the original _load_optimizer_and_scheduler to resume dataloader

        # Call the original function
        # super()._load_optimizer_and_scheduler(checkpoint)
        # Below is copied from the original _load_optimizer_and_scheduler
        # But allow only loading optimizer if the scheduler does not exist

        """If optimizer and scheduler states exist, load them."""
        if checkpoint is None:
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else (
                os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
                or os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME_BIN))
                or (
                    os.path.isdir(checkpoint)
                    and any(
                        OPTIMIZER_NAME_BIN.split(".")[0] in folder_name
                        for folder_name in os.listdir(checkpoint)
                        if os.path.isdir(os.path.join(checkpoint, folder_name))
                    )
                )
            )
        )
        if checkpoint_file_exists:
            logger.warning(f"Load optimizer state from {checkpoint}")
            # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
            # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
            # likely to get OOM on CPU (since we load num_gpu times the optimizer state
            map_location = self.args.device if self.args.world_size > 1 else "cpu"
            if self.is_fsdp_enabled:
                load_fsdp_optimizer(
                    self.accelerator.state.fsdp_plugin,
                    self.accelerator,
                    self.optimizer,
                    self.model,
                    checkpoint,
                    **_get_fsdp_ckpt_kwargs(),
                )
            else:
                self.optimizer.load_state_dict(
                    torch.load(
                        os.path.join(checkpoint, OPTIMIZER_NAME),
                        map_location=map_location,
                    )
                )

        if os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            logger.warning(f"Load scheduler state from {checkpoint}")
            with warnings.catch_warnings(record=True) as caught_warnings:
                self.lr_scheduler.load_state_dict(
                    torch.load(os.path.join(checkpoint, SCHEDULER_NAME))
                )
            reissue_pt_warnings(caught_warnings)

        # Resume dataloader
        if checkpoint is not None and self.args.streaming_dataset:
            try:
                dataset_state_dict = json.load(
                    open(os.path.join(checkpoint, "streaming_dataset_state.json"))
                )
            except:
                logger.warning(
                    f"Failed to load streaming dataset state from {checkpoint}"
                )
                logger.warning(f"Fall back to the HF data skip")
                self.args.ignore_data_skip = False

                return

            # First, disable HF's data skip
            self.args.ignore_data_skip = True

            # Load the dataset state and reinit the dataloader
            logger.warning(
                "⚠️ Dataset does not support state_dict loading, skipping resume."
            )

    # Override the original train() to handle the case
    # when resuming from a checkpoint but no trainer_state is there
    # (e.g., continual training with optimizer states)
    def train(
        self,
        resume_from_checkpoint: Optional[Union[str, bool]] = None,
        trial: Union["optuna.Trial", Dict[str, Any]] = None,
        ignore_keys_for_eval: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Main training entry point.

        Args:
            resume_from_checkpoint (`str` or `bool`, *optional*):
                If a `str`, local path to a saved checkpoint as saved by a previous instance of [`Trainer`]. If a
                `bool` and equals `True`, load the last checkpoint in *args.output_dir* as saved by a previous instance
                of [`Trainer`]. If present, training will resume from the model/optimizer/scheduler states loaded here.
            trial (`optuna.Trial` or `Dict[str, Any]`, *optional*):
                The trial run or the hyperparameter dictionary for hyperparameter search.
            ignore_keys_for_eval (`List[str]`, *optional*)
                A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                gathering predictions for evaluation during the training.
            kwargs (`Dict[str, Any]`, *optional*):
                Additional keyword arguments used to hide deprecated arguments
        """
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None

        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        args = self.args

        self.is_in_train = True

        # Attach NEFTune hooks if necessary
        if self.neftune_noise_alpha is not None:
            self.model = self._activate_neftune(self.model)

        # do_train is not a reliable argument, as it might not be set and .train() still called, so
        # the following is a workaround:
        if (args.fp16_full_eval or args.bf16_full_eval) and not args.do_train:
            self._move_model_to_device(self.model, args.device)

        if "model_path" in kwargs:
            resume_from_checkpoint = kwargs.pop("model_path")
            warnings.warn(
                "`model_path` is deprecated and will be removed in a future version. Use `resume_from_checkpoint` "
                "instead.",
                FutureWarning,
            )
        if len(kwargs) > 0:
            raise TypeError(
                f"train() received got unexpected keyword arguments: {', '.join(list(kwargs.keys()))}."
            )
        # This might change the seed so needs to run first.
        self._hp_search_setup(trial)
        self._train_batch_size = self.args.train_batch_size

        # Model re-init
        model_reloaded = False
        if self.model_init is not None:
            # Seed must be set before instantiating the model when using model_init.
            enable_full_determinism(
                self.args.seed
            ) if self.args.full_determinism else set_seed(self.args.seed)
            self.model = self.call_model_init(trial)
            model_reloaded = True
            # Reinitializes optimizer and scheduler
            self.optimizer, self.lr_scheduler = None, None

        # Load potential model checkpoint
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(
                    f"No valid checkpoint found in output directory ({args.output_dir})"
                )

        if resume_from_checkpoint is not None:
            if (
                not is_sagemaker_mp_enabled()
                and not self.is_deepspeed_enabled
                and not self.is_fsdp_enabled
            ):
                self._load_from_checkpoint(resume_from_checkpoint)
            # In case of repeating the find_executable_batch_size, set `self._train_batch_size` properly
            if os.path.isfile(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)):
                state = TrainerState.load_from_json(
                    os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
                )
                if state.train_batch_size is not None:
                    self._train_batch_size = state.train_batch_size

        # If model was re-initialized, put it on the right device and update self.model_wrapped
        if model_reloaded:
            if self.place_model_on_device:
                self._move_model_to_device(self.model, args.device)
            self.model_wrapped = self.model

        inner_training_loop = find_executable_batch_size(
            self._inner_training_loop, self._train_batch_size, args.auto_find_batch_size
        )
        if args.push_to_hub:
            try:
                # Disable progress bars when uploading models during checkpoints to avoid polluting stdout
                hf_hub_utils.disable_progress_bars()
                return inner_training_loop(
                    args=args,
                    resume_from_checkpoint=resume_from_checkpoint,
                    trial=trial,
                    ignore_keys_for_eval=ignore_keys_for_eval,
                )
            finally:
                hf_hub_utils.enable_progress_bars()
        else:
            return inner_training_loop(
                args=args,
                resume_from_checkpoint=resume_from_checkpoint,
                trial=trial,
                ignore_keys_for_eval=ignore_keys_for_eval,
            )

    def _fsdp_qlora_plugin_updates(self):
        pass  # This messes with autowrap policy
    
    
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        batch_samples = []
        num_items_in_batch = None
        
        # i=0
        # while i<num_batches:
        #     temp = next(epoch_iterator)
        #     if temp['seq_lengths'][0,-1]%2 == 0:
        #         batch_samples.append(temp)
        #         i+=1
                
  
        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and (
                # num_items_in_batch is passed to model forward
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3757
                self.model_accepts_loss_kwargs
                # num_items_in_batch is passed to compute_loss_func
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3773
                or self.compute_loss_func is not None
                # num_items_in_batch is also verified if (self.model_accepts_loss_kwargs or self.compute_loss_func)
                # https://github.com/huggingface/transformers/blob/v4.49.0/src/transformers/trainer.py#L3790
            )
        )

        if count_num_items_in_batch:
            # For now we don't support object detection
            try:
                num_items_in_batch = sum([(batch["labels"].ne(-100)).sum() for batch in batch_samples])
            except (TypeError, AttributeError):
                pass

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                num_items_in_batch = self.accelerator.gather(num_items_in_batch).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        return batch_samples, num_items_in_batch
