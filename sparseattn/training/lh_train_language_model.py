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
import torch
from transformers import LlamaForCausalLM, AutoTokenizer

from .modeling_flash_llama import PawLlamaForCausalLM, PawLlamaConfig
from .modeling_flash_qwen import PawQwen3ForCausalLM, PawQwen3Config
from .modeling_flash_phi import PawPhi3ForCausalLM, PawPhi3Config
from .lh_trainer import Trainer
# from .lh_trainer_nsa import Trainer as NSATrainer

# from .dataset import build_dataset, DataCollator, DataArguments
from .dataset_batch import build_dataset, PackingDataCollator, DataArguments, CustomDistributedStratifiedSampler
from .dataset import logger as dataset_logger
from .script_arguments import ScriptArguments, TrainingArguments


from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy

from transformers.trainer_utils import get_last_checkpoint
import json

from csv import reader

# from fla.models.nsa import AutoModelForCausalLM as NSAAutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_masks_from_tsv_file(
    model,
    tsv_file,
    sparsity=None,
    threshold=None,
):
    f = reader(open(tsv_file, "r"), delimiter="\t")
    masks = [[float(x) for x in row] for row in f]
    if threshold is not None:
        masks = [[float(x > threshold) for x in row] for row in masks]
    # At this point, masks are in [0,1] -- relinearize to [-10, 10]
    masks = [[(2 * x - 1) * 10 for x in row] for row in masks]

    model.load_masks(masks)

    if sparsity is not None:
        model.round_masks_for_sparsity(sparsity)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of script_args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, DataArguments))
    script_args, training_args, data_args = parser.parse_args_into_dataclasses()
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
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name or script_args.model_name_or_path,
        cache_dir=script_args.cache_dir,
        use_fast=script_args.use_fast_tokenizer,
        revision=script_args.model_revision,
        use_auth_token=True if script_args.use_auth_token else None,
        enable_thinking=True if script_args.use_thinking else False,
    )
    # Some special tokens
    special_tokens_dict = {
        'additional_special_tokens': [
            '[TASK_SQA]',   # For Single QA
            '[TASK_MHQA]',  # For MultiHop QA
            '[TASK_SUM]',   # For Summarization
            '[TASK_CODE]',  # For Code
            '[TASK_OTHER]'  # Fallback
        ]
    }
    tokenizer.add_special_tokens(special_tokens_dict)
    
    # Determine model type and load appropriate config
    if "qwen" in script_args.model_name_or_path.lower():
        config = PawQwen3Config.from_pretrained(
            script_args.config_name or script_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            revision=script_args.model_revision,
            use_auth_token=True if script_args.use_auth_token else None,
            toggle_type=training_args.toggle_type,
            local_window_size=training_args.context_window_if_toggled,
            sink_size=training_args.sink_size,
            topk_k=training_args.topk_k,
            disable_linear_regularization_term=training_args.disable_linear_regularization_term,
            enable_ada_sparsity=training_args.enable_ada_sparsity,
            enable_layerwise_sparsity=training_args.enable_layerwise_sparsity,
            erank_analysis_path=training_args.erank_analysis_path
            if hasattr(training_args, "erank_analysis_path")
            else None,
        )
    elif "llama" in script_args.model_name_or_path.lower():
        config = PawLlamaConfig.from_pretrained(
            script_args.config_name or script_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            revision=script_args.model_revision,
            use_auth_token=True if script_args.use_auth_token else None,
            toggle_type=training_args.toggle_type,
            local_window_size=training_args.context_window_if_toggled,
            sink_size=training_args.sink_size,
            topk_k=training_args.topk_k,
            disable_linear_regularization_term=training_args.disable_linear_regularization_term,
            enable_ada_sparsity=training_args.enable_ada_sparsity,
            enable_layerwise_sparsity=training_args.enable_layerwise_sparsity,
            erank_analysis_path=training_args.erank_analysis_path
            if hasattr(training_args, "erank_analysis_path")
            else None,
        )
    elif "phi" in script_args.model_name_or_path.lower():
        config = PawPhi3Config.from_pretrained(
            script_args.config_name or script_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            revision=script_args.model_revision,
            use_auth_token=True if script_args.use_auth_token else None,
            toggle_type=training_args.toggle_type,
            local_window_size=training_args.context_window_if_toggled,
            sink_size=training_args.sink_size,
            topk_k=training_args.topk_k,
            disable_linear_regularization_term=training_args.disable_linear_regularization_term,
            enable_ada_sparsity=training_args.enable_ada_sparsity,
            enable_layerwise_sparsity=training_args.enable_layerwise_sparsity,
            erank_analysis_path=training_args.erank_analysis_path
            if hasattr(training_args, "erank_analysis_path")
            else None,
        )
    else:
        raise ValueError(
            f"Model name {script_args.model_name_or_path} does not contain. "
            "Please provide a valid model name."
        )
    if script_args.config_overrides:
        logger.info(f"Overriding config: {script_args.config_overrides}")
        config.update_from_string(script_args.config_overrides)
        logger.info(f"New config: {config}")

    if script_args.config_overrides_json:
        logger.info(f"Overriding config: {script_args.config_overrides_json}")
        config.update(json.loads(script_args.config_overrides_json))
        logger.info(f"New config: {config}")

    config.pad_token_id = 0

    if script_args.model_name_or_path:
        # Determine model type and load appropriate model
        if training_args.attention_type is not None and "nsa" in training_args.attention_type :
            model = LlamaForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
                torch_dtype=torch.bfloat16,
            )
            # config = model.config
            # config._attn_implementation = "flash_attention_2"
            # config.compress_type = "linear"#"avgpool","weightedpool"
            # config.kernel_size = 32
            # config.kernel_stride = 16
            # config.block_size = 64
            # config.topk = 8
            # config.init_blocks = 1
            # config.local_blocks = 2
            # config.window_size = 500
            # for i, layer in enumerate(model.model.layers):
            #     original_attn = layer.self_attn

            #     original_dtype = next(original_attn.parameters()).dtype
            #     device = next(original_attn.parameters()).device

            #     new_attn = LlamaNSA(config, layer_idx=original_attn.layer_idx)
            #     new_attn.load_state_dict(original_attn.state_dict(), strict=False)
            #     new_attn = new_attn.to(device).to(original_dtype)

            #     layer.self_attn = new_attn

        elif "qwen" in script_args.model_name_or_path.lower():
            model = PawQwen3ForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                config=config,
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
            )
        elif "llama" in script_args.model_name_or_path.lower():
            model = PawLlamaForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                config=config,
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
            )
        elif "phi" in script_args.model_name_or_path.lower():
            model = PawPhi3ForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                config=config,
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
            )
        else:
            raise ValueError(
                f"Model name {script_args.model_name_or_path} does not contain. "
                "Please provide a valid model name."
            )
    else:
        logger.warning(f"Initializing new PawLlamaForCausalLM from scratch")
        # Determine model type and initialize appropriate model
        if "qwen" in script_args.model_name_or_path.lower():
            model = PawQwen3ForCausalLM(config)
        elif "llama" in script_args.model_name_or_path.lower():
            model = PawLlamaForCausalLM(config)
        elif "phi" in script_args.model_name_or_path.lower():
            model = PawPhi3ForCausalLM(config)
        else:
            raise ValueError(
                f"Model name {script_args.model_name_or_path} does not contain. "
                "Please provide a valid model name."
            )
    # reset tokenizer 
    model.resize_token_embeddings(len(tokenizer))

    if hasattr(model, "reset_masks"):
        model.reset_masks()

    if training_args.stripe_init_width_1 is not None:
        # We should initialize with a striped pattern
        assert training_args.stripe_init_width_2 is not None, (
            "If stripe_init_width_1 is set, stripe_init_width_2 must be set as well"
        )
        logger.info(
            f"Initializing with a striped pattern: ({training_args.stripe_init_width_1}, {training_args.stripe_init_width_2})"
        )
        if not training_args.freeze_mask_parameters:
            logger.warning(
                "Stripe initialization without freezing mask parameters is not recommended"
            )

        if hasattr(model, "reset_masks_with_stripe_pattern"):
            model.reset_masks_with_stripe_pattern(
                training_args.stripe_init_width_1,
                training_args.stripe_init_width_2,
                start_with_keep=training_args.stripe_init_start_with_keep,
            )
        else:
            logger.warning("skipping stripe initialization -- model does not support it")
    elif training_args.load_masks_from is not None:
        logger.info(f"Loading masks from {training_args.load_masks_from}")
        if hasattr(model, "load_masks"):
            load_masks_from_tsv_file(
                model,
                training_args.load_masks_from,
                sparsity=training_args.load_masks_sparsity,
            )
        else:
            logger.warning("skipping loading masks -- model does not support it")

    if (
        script_args.tokenizer_name is not None
        and script_args.model_name_or_path != script_args.tokenizer_name
    ):
        model.resize_token_embeddings(len(tokenizer))

    logger.info(f"Model: {model}")

    # Idk causes weird issues without this when doing multiple runs from different codebases
    import streaming

    streaming.base.util.clean_stale_shared_memory()

    if script_args.token_scaled_loss:
        if hasattr(model, "token_scaled_loss"):
            model.token_scaled_loss = True
            training_args.token_scaled_loss = True
        else:
            logger.warning("skipping token_scaled_loss -- model does not support it")

    data_collator = PackingDataCollator(tokenizer, data_args, max_seq_len=data_args.per_device_max_tokens)
    assert training_args.max_steps is not None, "max_steps must be set!"
    
    # load_datasets
    if training_args.do_train:
        # train_dataset = build_dataset(
        #     script_args.tokenized_mds_train, training_args, data_args, is_training=True
        # )
        train_dataset = build_dataset(
            script_args.tokenized_mds_train,
            tokenizer=tokenizer,
            data_args=data_args,
            is_training=True,
        )
        
        class_indices = train_dataset.get_class_indices()
        logger.info(f"Using stratified sampling. Class distribution: {[len(v) for v in class_indices.values()]}")

        if torch.distributed.is_initialized():
            world_size = torch.distributed.get_world_size()
        else:
            world_size = training_args.n_gpu

        train_sampler = CustomDistributedStratifiedSampler(
            train_dataset,
            class_indices=class_indices,
            num_gpus=world_size,
            required_per_class=2
        )

        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=1,
            sampler=train_sampler,
            collate_fn=data_collator,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )
        
        


    if training_args.do_eval:
        # eval_dataset = {
        #     x.split("/")[-1]: build_dataset(
        #         [x], training_args, data_args, is_training=False
        #     )
        #     for x in script_args.tokenized_mds_validation
        # }
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
    

    # Initialize our Trainer
    if training_args.attention_type is not None and "nsa" in training_args.attention_type :
        # trainer = NSATrainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset if training_args.do_train else None,
        #     eval_dataset=eval_dataset if training_args.do_eval else None,
        #     tokenizer=tokenizer,
        #     data_collator=data_collator,
        #     log_loss=script_args.should_log_loss,
        # )
        pass
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            log_loss=script_args.should_log_loss,
        )
    if training_args.do_train:
        trainer.train_dataloader = train_dataloader
        logger.info("Successfully injected CustomDistributedStratifiedSampler into Trainer.")

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
