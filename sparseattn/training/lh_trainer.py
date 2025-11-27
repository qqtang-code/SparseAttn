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

from .dataset_batch import StreamingParquetIterable

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
    def __init__(self, model, args, *more_args, **kwargs):
        self.log_loss = kwargs.pop("log_loss", False)

        super().__init__(model, args, *more_args, **kwargs)
        self.start_head_sparsity = args.start_head_sparsity
        self.end_head_sparsity = args.end_head_sparsity
        self.learning_rate = args.learning_rate
        self.mask_learning_rate = args.mask_learning_rate
        self.reg_learning_rate = args.reg_learning_rate
        self.warmup_type = args.warmup_type
        self.sparsity_warmup_ratio = args.sparsity_warmup_ratio
        self.disable_linear_regularization_term = (
            args.disable_linear_regularization_term
        )
        self.context_window_if_toggled = args.context_window_if_toggled

        self.freeze_non_mask_parameters = args.freeze_non_mask_parameters
        self.freeze_mask_parameters = args.freeze_mask_parameters

        self.num_sparsity_warmup_steps = math.ceil(
            self.sparsity_warmup_ratio * self.args.max_steps
        )
        
        # self.task_sparsity_config = {
        #     "default": {"start": self.start_head_sparsity, "end": self.end_head_sparsity},
        #     "Code": {"start": 0.1, "end": 0.7},
        #     "Math": {"start": 0.0, "end": 0.6},
        #     "MultiHop QA": {"start": 0.2, "end": 0.5},
        #     "Single QA": {"start": 0.1, "end": 0.7},
        #     "Summarization": {"start": 0.3, "end": 0.6},
        # }
        self.task_sparsity_config = {
            "default": {"start": self.start_head_sparsity, "end": self.end_head_sparsity},
            "Code": {"start": 0.0, "end": 0.4},
            "Math": {"start": 0.0, "end": 0.6},
            "MultiHop QA": {"start": 0.0, "end": 0.2},
            "Single QA": {"start": 0.0, "end": 0.2},
            "Summarization": {"start": 0.0, "end": 0.7},
        }


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

    def get_current_target_sparsity(self, global_step: int, tasks: List[str]) -> torch.Tensor:
        """
        Returns: torch.Tensor of shape [len(tasks)], device=cpu (will be moved later)
        """
        sparsities = []
        for task in tasks:
            cfg = self.task_sparsity_config.get(task, self.task_sparsity_config["default"])
            start_sp = cfg["start"]
            end_sp = cfg["end"]
            
            if global_step < self.num_sparsity_warmup_steps:
                if self.warmup_type == "linear":
                    sp = start_sp + (end_sp - start_sp) * global_step / self.num_sparsity_warmup_steps
                elif self.warmup_type == "logarithmic":
                    log_one_minus_start = math.log(1 - start_sp)
                    log_one_minus_end = math.log(1 - end_sp)
                    log_one_minus_sp = (
                        log_one_minus_start
                        + (log_one_minus_end - log_one_minus_start) * global_step / self.num_sparsity_warmup_steps
                    )
                    sp = 1 - math.exp(log_one_minus_sp)
                else:
                    raise ValueError(f"Unknown warmup_type: {self.warmup_type}")
            else:
                sp = end_sp
            sparsities.append(sp)
        
        return torch.tensor(sparsities, dtype=torch.float32)  # shape: [B]

    def get_sequence_parallel_inputs(self, inputs):
        seq_parallel_world_size = (
            dist.get_world_size(self.seq_parallel_group) if dist.is_initialized() else 1
        )

        if seq_parallel_world_size > 1:
            seq_parallel_rank = dist.get_rank(self.seq_parallel_group)

            input_ids = inputs["input_ids"]
            labels = inputs["labels"]

            shifted_labels = labels.roll(-1, dims=-1)
            shifted_labels[..., -1] = -100

            seq_lengths = inputs["seq_lengths"]

            # add right padding here to make equal sized chunks
            if input_ids.size(-1) % seq_parallel_world_size != 0:
                padding = seq_parallel_world_size - (
                    input_ids.size(-1) % seq_parallel_world_size
                )
                padding_zeros = torch.full(
                    input_ids.size()[:-1] + (padding,),
                    0,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, padding_zeros], dim=-1)
                shifted_labels = torch.cat(
                    [shifted_labels, padding_zeros - 100], dim=-1
                )
                seq_lengths[-1] += padding

            # select chunk of input_ids and labels
            input_ids_chunks = torch.tensor_split(
                input_ids, seq_parallel_world_size, dim=-1
            )
            shifted_labels_chunks = torch.tensor_split(
                shifted_labels, seq_parallel_world_size, dim=-1
            )

            inputs = {
                "input_ids": input_ids_chunks[seq_parallel_rank],
                "shifted_labels": shifted_labels_chunks[seq_parallel_rank],
                "seq_lengths": seq_lengths,
                "seq_parallel_group": self.seq_parallel_group,
            }

            max_seq_length = seq_lengths.max()
            max_tokens_per_device = seq_lengths.sum() // seq_parallel_world_size

            start_index = sum(
                chunk.size(-1) for chunk in input_ids_chunks[:seq_parallel_rank]
            )
            end_index = start_index + input_ids_chunks[seq_parallel_rank].size(-1)

            inputs["position_ids"] = torch.tensor([start_index]).to(input_ids.device)

            # max sequence length is smaller per device => no need for sequence parallelism
            if max_seq_length <= max_tokens_per_device:
                # take the seq length field and only retain seq lengths with indices that are valid for this rank
                seq_indices = seq_lengths.cumsum(-1)
                seq_indices = seq_indices[
                    (seq_indices < end_index) & (seq_indices >= start_index)
                ]

                start_index_tensor = torch.tensor(
                    [start_index], device=seq_indices.device
                )
                end_index_tensor = torch.tensor([end_index], device=seq_indices.device)

                seq_lengths = seq_indices.diff(
                    prepend=start_index_tensor, append=end_index_tensor
                )
                seq_lengths = seq_lengths[seq_lengths > 0]
                inputs["seq_lengths"] = seq_lengths
                inputs["seq_parallel_group"] = None

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
        tasks = inputs.get("task_type", ["default"] * inputs["input_ids"].size(0))
        # tasks = ["default"] * inputs["input_ids"].size(0)
        target_sparsity = self.get_current_target_sparsity(self.state.global_step, tasks)
        target_sparsity = target_sparsity.to(model.device)  # [B]
        
        inputs = self.get_sequence_parallel_inputs(inputs)

        unique_tasks = list(set(tasks))
        mean_spars = {t: target_sparsity[i].item() for i, t in enumerate(tasks[:3])}
        print(f"[Step {self.state.global_step}] Sample tasks: {tasks[:3]} → sparsity: {[f'{s:.3f}' for s in target_sparsity[:3].tolist()]}")
        attention_mask = inputs.get("attention_mask")
        valid_tokens = attention_mask.sum(dim=1)
        print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: "
            f"valid tokens per sample = {valid_tokens.tolist()}, total = {valid_tokens.sum().item()}")
        try:
            outputs = model(**inputs, use_cache=False, target_sparsity=target_sparsity)
        except Exception as e:
            attn_mask = inputs["attention_mask"]
            valid_tokens = attn_mask.sum(dim=1)
            print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: "
                f"valid tokens per sample = {valid_tokens.tolist()}, total = {valid_tokens.sum().item()}")
            print(f"[Warning] Error occurred on this batch, skipping. Error: {repr(e)}")
            print("-" * 80, flush=True)

            zero_loss = torch.tensor(0.0, requires_grad=True, device=model.device)
            if return_outputs:
                return (zero_loss, None)
            return zero_loss

        lm_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

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

        reg_loss = (
            outputs["sparsity_loss"] if isinstance(outputs, dict) else outputs[-1]
        )
        loss = lm_loss + 20 * reg_loss
        task = tasks[0]  # because batch size = 1
        model_sparsity = outputs["model_sparsity"]
        print(f"Rank {torch.distributed.get_rank() if torch.distributed.is_initialized() else 0}: "f"[Step {self.state.global_step}] Task={task} | model_sparsity={model_sparsity} ｜ reg_loss={reg_loss}")

        if self.log_loss and self.accelerator.is_main_process:
            model_sparsity = (
                outputs["model_sparsity"] if isinstance(outputs, dict) else outputs[-3]
            )
            logger.info(
                f"@ {self.state.global_step} | Loss: {loss} | LM Loss: {lm_loss} | "
                f"Reg Loss: {reg_loss} | Target Sparsity: {target_sparsity} | "
                f"Model Sparsity: {model_sparsity}"
            )
            # Fetch diagnostics if present
            exp_sparsity = (
                outputs.get("expected_model_sparsity", None)
                if isinstance(outputs, dict)
                else None
            )
            lambda1 = (
                outputs.get("lambda1", None) if isinstance(outputs, dict) else None
            )
            lambda2 = (
                outputs.get("lambda2", None) if isinstance(outputs, dict) else None
            )
            ez_mean = (
                outputs.get("expected_z_mean", None)
                if isinstance(outputs, dict)
                else None
            )
            ez_std = (
                outputs.get("expected_z_std", None)
                if isinstance(outputs, dict)
                else None
            )
            la_mean = (
                outputs.get("log_alpha_mean", None)
                if isinstance(outputs, dict)
                else None
            )
            la_std = (
                outputs.get("log_alpha_std", None)
                if isinstance(outputs, dict)
                else None
            )

            extra = []
            if lambda1 is not None and lambda2 is not None:
                extra.append(
                    f"Lambda1: {float(lambda1):.4f} Lambda2: {float(lambda2):.4f}"
                )

            logger.info(
                f"@ {self.state.global_step} | Loss: {loss} | LM: {lm_loss} | Reg: {reg_loss} | "
                f"Target: {target_sparsity} | Sparsity: {model_sparsity}"
                + (" | " + " | ".join(extra) if len(extra) else "")
            )

            if (
                not return_output_and_metrics
                and getattr(self.args, "log_train_sparsity_metrics", True)
                and self.state.global_step > 0
            ):
                train_metrics = {
                    "lm_loss": float(
                        lm_loss.detach().item()
                        if isinstance(lm_loss, torch.Tensor)
                        else lm_loss
                    ),
                    "reg_loss": float(
                        reg_loss.detach().item()
                        if isinstance(reg_loss, torch.Tensor)
                        else reg_loss
                    ),
                    "loss": float(
                        loss.detach().item() if isinstance(loss, torch.Tensor) else loss
                    ),
                    "target_sparsity": target_sparsity,
                    "model_sparsity": model_sparsity,
                    "step": self.state.global_step,
                }
                if isinstance(outputs, dict):
                    for k in [
                        "lambda1",
                        "lambda2",
                    ]:
                        v = outputs.get(k, None)
                        if v is not None:
                            if isinstance(v, torch.Tensor):
                                v = v.detach()
                                if v.numel() == 1:
                                    v = v.item()
                                else:
                                    v = float(v.mean().item())
                            train_metrics[k] = v
                # if isinstance(outputs, dict):
                #     if (
                #         "layerwise_model_sparsity" in outputs
                #         and outputs["layerwise_model_sparsity"] is not None
                #     ):
                #         lms = outputs["layerwise_model_sparsity"]
                #         if isinstance(lms, torch.Tensor):
                #             lms = lms.detach().cpu().float()
                #             for i, s in enumerate(lms):
                #                 train_metrics[f"layer_{i}/sparsity"] = float(s)

                #     if (
                #         "layerwise_target_sparsity" in outputs
                #         and outputs["layerwise_target_sparsity"] is not None
                #     ):
                #         lt = outputs["layerwise_target_sparsity"]
                #         if isinstance(lt, torch.Tensor):
                #             lt = lt.detach().cpu().float()
                #             for i, t in enumerate(lt):
                #                 train_metrics[f"layer_{i}/target"] = float(t)

                #     if (
                #         "layerwise_model_sparsity" in outputs
                #         and "layerwise_target" in outputs
                #     ):
                #         lms = outputs["layerwise_model_sparsity"]
                #         lt = outputs["layerwise_target"]
                #         if lms is not None and lt is not None:
                #             diff = (lms - lt).detach().cpu().float()
                #             for i, d in enumerate(diff):
                #                 train_metrics[f"layer_{i}/sparsity_diff"] = float(d)
                self.log(train_metrics)

        if return_output_and_metrics:
            # shifted_labels = inputs["labels"][:,1:].contiguous()
            # valid_mask = (shifted_labels != -100)
            # correct = (outputs.logits[:,:-1].argmax(-1) == shifted_labels).float()
            # correct[~valid_mask] = 0.0
            # acc = correct.sum(dim=-1) / valid_mask.float().sum(dim=-1)
            model_sparsity_val = (
                outputs["model_sparsity"] if isinstance(outputs, dict) else outputs[-3]
            )

            metrics = {
                "lm_loss": lm_loss.detach().item()
                if isinstance(lm_loss, torch.Tensor)
                else float(lm_loss),
                "reg_loss": reg_loss.detach().item()
                if isinstance(reg_loss, torch.Tensor)
                else float(reg_loss),
                "loss": loss.detach().item()
                if isinstance(loss, torch.Tensor)
                else float(loss),
                "target_sparsity": float(target_sparsity),
                "model_sparsity": float(model_sparsity_val),
                "step": self.state.global_step,
            }

            # Carry diagnostics into metrics if available
            if isinstance(outputs, dict):
                for k in [
                    "lambda1",
                    "lambda2",
                ]:
                    if k in outputs and outputs[k] is not None:
                        metrics[k] = outputs[k]

            self.log(metrics)

            return (loss, outputs, metrics)

        if return_outputs:
            return (loss, outputs)
        else:
            return loss

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        Updates: Also makes sure to separate the parameters into different groups for different learning rates.
        """
        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)

            optimizer_1_group = []  # params, non decay
            optimizer_2_group = []  # params, decay
            optimizer_3_group = []  # mask params, decay
            optimizer_4_group = []  # reg params, decay
            optimizer_5_group = []  # mask params, decay

            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if ".mask_allocator." in n or "mask_allocator" in n:
                    if self.freeze_mask_parameters:
                        p.requires_grad = False
                    else:
                        optimizer_5_group.append(p)
                elif "log_alpha" in n:
                    if self.freeze_mask_parameters:
                        p.requires_grad = False
                    else:
                        optimizer_3_group.append(p)
                elif "sparsity_lambda" in n:
                    if self.freeze_mask_parameters:
                        p.requires_grad = False
                    else:
                        optimizer_4_group.append(p)
                elif n in decay_parameters:
                    if self.freeze_non_mask_parameters:
                        p.requires_grad = False
                    else:
                        optimizer_2_group.append(p)
                else:
                    if self.freeze_non_mask_parameters:
                        p.requires_grad = False
                    else:
                        optimizer_1_group.append(p)
            optimizer_grouped_parameters = []
            if not self.freeze_non_mask_parameters:
                optimizer_grouped_parameters.append(
                    {
                        "params": optimizer_1_group,
                        "weight_decay": 0.0,
                        "lr": self.learning_rate,
                    }
                )
                optimizer_grouped_parameters.append(
                    {
                        "params": optimizer_2_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.learning_rate,
                    }
                )
            if not self.freeze_mask_parameters:
                optimizer_grouped_parameters.append(
                    {
                        "params": optimizer_3_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.mask_learning_rate,
                    }
                )
                optimizer_grouped_parameters.append(
                    {
                        "params": optimizer_4_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.reg_learning_rate,
                        "maximize": True,
                    }
                )
                optimizer_grouped_parameters.append(
                    {
                        "params": optimizer_5_group,
                        "weight_decay": self.args.weight_decay,
                        "lr": self.mask_learning_rate,
                    }
                )

            optimizer_cls, optimizer_kwargs = self.get_optimizer_cls_and_kwargs(
                self.args, opt_model
            )

            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs
            )

            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum(
                            {
                                p.data_ptr(): p.numel() for p in module.parameters()
                            }.values()
                        )
                        logger.info(f"skipped {module}: {skipped / 2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32}
                        )
                        logger.debug(f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped / 2**20}M params")

        if is_sagemaker_mp_enabled():
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

    # def get_train_dataloader(self):
    #     """
    #     Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    #     The plainest data loader is enough.
    #     """
    #     if not isinstance(self.train_dataset, StreamingParquetIterable):
    #         return super().get_train_dataloader()

    #     if not self.args.streaming_dataset:
    #         return super().get_train_dataloader()

    #     logger.warning("Use streaming dataloader for train")

    #     if self.train_dataset is None:
    #         raise ValueError("Trainer: training requires a train_dataset.")

    #     train_dataset = self.train_dataset
    #     data_collator = self.data_collator
    #     data_collator = self._get_collator_with_removed_columns(
    #         data_collator, description="training"
    #     )

    #     dataloader_params = {
    #         "batch_size": self._train_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #         "persistent_workers": self.args.dataloader_persistent_workers,
    #     }

    #     # Streaming is iterable so no need to set sampler etc.

    #     # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    #     self.train_dataloader = DataLoader(train_dataset, **dataloader_params)
    #     # This actually uses the dataset first dimension......

    #     return self.train_dataloader

    # def get_eval_dataloader(self, eval_dataset):
    #     """
    #     Because streaming handles the distributed data parallel by itself, we don't need special data loader.
    #     The plainest data loader is enough.
    #     """
    #     if not self.args.streaming_dataset:
    #         return super().get_eval_dataloader()

    #     logger.warning("Use streaming dataloader for val")

    #     if eval_dataset is None and self.eval_dataset is None:
    #         raise ValueError("Trainer: evaluation requires an eval_dataset.")
    #     eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset
    #     data_collator = self.data_collator
    #     data_collator = self._get_collator_with_removed_columns(
    #         data_collator, description="evaluation"
    #     )

    #     dataloader_params = {
    #         "batch_size": self.args.eval_batch_size,
    #         "collate_fn": data_collator,
    #         "num_workers": self.args.dataloader_num_workers,
    #         "pin_memory": self.args.dataloader_pin_memory,
    #         "persistent_workers": self.args.dataloader_persistent_workers,
    #     }

    #     # Streaming is iterable so no need to set sampler etc.

    #     # Instead of use accelerate to prepare the dataloader, we just return a plain dataloader
    #     return StreamingDataLoader(eval_dataset, **dataloader_params)

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

        if (
            isinstance(self.train_dataset, StreamingParquetIterable)
            and self.state.is_world_process_zero
        ):
            num_samples = (
                self.state.global_step
                * self.args.train_batch_size
                * self.args.world_size
                * self.args.gradient_accumulation_steps
            )
            dataset_state_dict = self.train_dataset.state_dict(num_samples)
            logger.warning(f"Save streaming dataset state: {dataset_state_dict}")
            with open(
                os.path.join(output_dir, "streaming_dataset_state.json"), "w"
            ) as f:
                json.dump(dataset_state_dict, f)

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
                f"Resume streaming dataset state from {checkpoint}: {dataset_state_dict}"
            )
            if hasattr(self.train_dataset, "load_state_dict"):
                self.train_dataset.load_state_dict(dataset_state_dict)
                logger.warning(
                    "✅ Successfully resumed StreamingParquetIterable state."
                )
            elif isinstance(self.train_dataset, PrepackedDataset):
                logger.info("PrepackedDataset detected — no streaming state to resume.")
            else:
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
