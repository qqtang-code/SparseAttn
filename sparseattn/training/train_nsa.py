import logging
import os
import sys
import torch
import datasets
import transformers
import functools

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

from .block_sparse_attention_triton.native_sparse_attention.module.llama_nsa import LlamaNSA
from .block_sparse_attention_triton.native_sparse_attention.module.qwen3_nsa import Qwen3NSA
import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, Qwen3ForCausalLM
from .modeling_nsa_llama import NSALlamaForCausalLM

from .lh_trainer_nsa import Trainer as NSATrainer

# from .dataset import build_dataset, DataCollator, DataArguments
from .dataset_batch import build_dataset, PackingDataCollator, DataArguments
from .dataset import logger as dataset_logger
from .script_arguments import ScriptArguments, TrainingArguments


from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from transformers.trainer_utils import get_last_checkpoint
import json

from csv import reader




logger = logging.getLogger(__name__)

def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of script_args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, DataArguments))

    script_args, training_args, data_args = parser.parse_args_into_dataclasses()
    # print("script_args", script_args)
    # print("data_args", data_args)
    # print("training_args", training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    dataset_logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Additional arguments {script_args}")
    logger.info(f"Data arguments {data_args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_name_or_path)
    #tokenizer = AutoTokenizer.from_pretrained(origin_model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # 或手动设置为其他 token
    if training_args.attention_type is not None and "nsa" in training_args.attention_type :
        if "Llama" in script_args.model_name_or_path:
            model = LlamaForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
                torch_dtype=torch.bfloat16,
                use_cache=False
            )
            config = model.config
            config._attn_implementation = "flash_attention_2"
            config.compress_type = "linear"#"avgpool","weightedpool"
            config.kernel_size = 32
            config.kernel_stride = 16
            config.block_size = 64
            config.topk = 128
            config.init_blocks = 1
            config.local_blocks = 2
            config.window_size = 512
            for i, layer in enumerate(model.model.layers):
                original_attn = layer.self_attn

                original_dtype = next(original_attn.parameters()).dtype
                device = next(original_attn.parameters()).device

                new_attn = LlamaNSA(config, layer_idx=original_attn.layer_idx)
                new_attn.load_state_dict(original_attn.state_dict(), strict=False)
                new_attn = new_attn.to(device).to(original_dtype)

                layer.self_attn = new_attn
        elif "Qwen3" in script_args.model_name_or_path:
            model = Qwen3ForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
                torch_dtype=torch.bfloat16,
                use_cache=False
            )
            config = model.config
            config._attn_implementation = "flash_attention_2"
            config.compress_type = "linear"#"avgpool","weightedpool"
            config.kernel_size = 32
            config.kernel_stride = 16
            config.block_size = 64
            config.topk = 128
            config.init_blocks = 1
            config.local_blocks = 2
            config.window_size = 512
            for i, layer in enumerate(model.model.layers):
                original_attn = layer.self_attn

                original_dtype = next(original_attn.parameters()).dtype
                device = next(original_attn.parameters()).device

                new_attn = Qwen3NSA(config, layer_idx=original_attn.layer_idx)
                new_attn.load_state_dict(original_attn.state_dict(), strict=False)
                new_attn = new_attn.to(device).to(original_dtype)

                layer.self_attn = new_attn

    if (
        script_args.tokenizer_name is not None
        and script_args.model_name_or_path != script_args.tokenizer_name
    ):
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Model: {model}")

    # Idk causes weird issues without this when doing multiple runs from different codebases

    if script_args.token_scaled_loss:
        if hasattr(model, "token_scaled_loss"):
            model.token_scaled_loss = True
            training_args.token_scaled_loss = True
        else:
            logger.warning("skipping token_scaled_loss -- model does not support it")

    # load_datasets
    if training_args.do_train:
        train_dataset = build_dataset(
            script_args.tokenized_mds_train,
            tokenizer=tokenizer,
            data_args=data_args,
            is_training=True,
        )

    if training_args.do_eval:
        eval_dataset = {
            x.split("/")[-1]: build_dataset(
                [x],
                tokenizer=tokenizer,
                data_args=data_args,
                training_args=training_args,
                is_training=False,
            )
            for x in script_args.tokenized_mds_validation
        }

    if training_args.do_predict:
        test_dataset = {
            x.split("/")[-1]: build_dataset(
                [x],
                tokenizer=tokenizer,
                data_args=data_args,
                training_args=training_args,
                is_training=False,
            )
            for x in script_args.tokenized_mds_test
        }

    # data_collator = DataCollator(tokenizer, data_args)
    data_collator = PackingDataCollator(tokenizer, data_args, max_seq_len=data_args.per_device_max_tokens)
    assert training_args.max_steps is not None, "max_steps must be set!"

    # Initialize our Trainer
    if training_args.attention_type is not None and "nsa" in training_args.attention_type :
        trainer = NSATrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            log_loss=script_args.should_log_loss,
        )


    if trainer.is_fsdp_enabled:
        # Identify which modules have "_fsdp_wrap" attribute set to True and wrap these
        def fsdp_policy_fn(module):
            return getattr(module, "_fsdp_wrap", False)

        auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=fsdp_policy_fn
        )
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Predict
    if training_args.do_predict:
        logger.info("*** Predict ***")
        predictions = trainer.predict(test_dataset=test_dataset)
        predictions = predictions.predictions
        predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)

        # Save predictions to output directory
        output_file = os.path.join(training_args.output_dir, "predictions.json")
        with open(output_file, "w") as f:
            json.dump(predictions, f, indent=2)


if __name__ == "__main__":
    main()
