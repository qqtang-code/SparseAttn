#!/bin/bash

# Model and training configuration (hard-coded values from your defaults)
model="/data2/hf_models/Qwen3-4B"
bsz=16
seq=2
lr=1e-5
steps=125
save_steps=20
warmup=0.1
min_lr_ratio=1e-7
seq_parallel_size=1

# FSDP
# 0=Disable, 1=FULL_SHARD, 2=SHARD_GRAD_OP, 3=NO_SHARD, 4=HYBRID_SHARD, 5=HYBRID_SHARD_ZERO2
fsdp=5  # 1=FULL_SHARD
gc=1    # enable gradient checkpointing

# PruLong-specific
max_toks=4000
start_head_sparsity=0.0
end_head_sparsity=0.3
mask_learning_rate=1.0
reg_learning_rate=1.0
sparsity_warmup_ratio=0.8
disable_linear_reg_term=false
context_window_if_toggled=2048
freeze_weights=true
freeze_masks=false
warmup_type="linear"

# Streaming
toggle_type="xattn"
sink_size=128
topk_k=2048

enable_ada_sparsity=true
enable_contrastive_loss=true

# Layer-wise sparsity
enable_layerwise_sparsity=false
layerwise_sparsity_schedule="high-low-high"
layerwise_sparsity_min_ratio=0.5
layerwise_sparsity_max_ratio=1.0
layerwise_sparsity_power=1.0
layerwise_sparsity_weight=1.0

# Dataset
dataset="/data2/public_data/mix_sft_64k"
task_type="sft"
pooling_mode="first_token"

# Extra name and run name
extra_name="sft3_pretrain_64k_xattn_mlp_linear_first_token_10reg_nocontrast_64k_12.4"
if [[ "$freeze_weights" == "true" ]]; then
    extra_name="${extra_name}_wfrozen"
fi
if [[ "$freeze_masks" == "true" ]]; then
    extra_name="${extra_name}_mfrozen"
fi

run_name="masksonly_$(basename "$model")_bsz${bsz}_steps${steps}_lr${lr}_warmup${warmup}_sp${end_head_sparsity}_cw${context_window_if_toggled}_mlr${mask_learning_rate}_rlr${reg_learning_rate}${extra_name}"

out_dir="checkpoints/$run_name"
mkdir -p "$out_dir"
nvidia-smi

# GPU/node detection
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
fi
num_gpus=${NUM_GPUS_PER_NODE:-$num_gpus}

num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | wc -l)
num_nodes=${num_nodes:-1}
num_nodes=${NUM_NODES:-$num_nodes}

# Distributed launch
if [ "$num_nodes" -gt 1 ]; then
    master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
    master_addr=${MASTER_ADDR:-$master_addr}
    header="srun torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$master_addr:56321 \
        --nnodes=$num_nodes \
        --nproc-per-node=$num_gpus \
        -m training.lh_train_language_model"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)
    header="torchrun \
        --rdzv-backend=c10d \
        --rdzv-endpoint=localhost:$master_port \
        --nnodes=1 \
        --nproc-per-node=$num_gpus \
        -m training.lh_train_language_model"
fi

# Gradient accumulation fixed to 4 (as in original override)
accu=4

echo "num_nodes=${num_nodes} num_gpus=${num_gpus}"

# Environment
export OMP_NUM_THREADS="$num_gpus"
export SWANLAB_API_KEY="t0PmOeLpVom1LRBDAKHaA"
export SWANLAB_LOG_DIR="$out_dir"
export SWANLAB_MODE="cloud"
export TOKENIZERS_PARALLELISM=true
export LOGIT_BLOCK_SIZE=2048

# Training args
base_arguments=(
    --report_to tensorboard
    --do_train

    --model_name "$model"
    --tokenizer_name "$model"

    --run_name "$run_name"
    --output_dir "$out_dir"
    --config_overrides_json ""
    --gradient_accumulation_steps "$accu"
    --per_device_train_batch_size "$seq"
    --per_device_eval_batch_size "$seq"

    --bf16
    --learning_rate "$lr"
    --min_lr_ratio "$min_lr_ratio"
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --warmup_ratio "$warmup"
    --optim adamw_torch

    --logging_steps 1
    --log_level info

    --max_steps "$steps"
    --save_steps "$save_steps"
    --save_total_limit 3
    --dataloader_num_workers 1

    --data_cache_dir "data_cache/sft"

    --disable_tqdm true
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false
    --cuda_empty_cache

    # PruLong
    --per_device_max_tokens "$max_toks"
    --task_type "$task_type"
    --seq_parallel_size "$seq_parallel_size"
    --start_head_sparsity "$start_head_sparsity"
    --end_head_sparsity "$end_head_sparsity"
    --mask_learning_rate "$mask_learning_rate"
    --reg_learning_rate "$reg_learning_rate"
    --warmup_type "$warmup_type"
    --sparsity_warmup_ratio "$sparsity_warmup_ratio"
    --disable_linear_regularization_term "$disable_linear_reg_term"
    --context_window_if_toggled "$context_window_if_toggled"
    --freeze_non_mask_parameters "$freeze_weights"
    --freeze_mask_parameters "$freeze_masks"
    --should_log_loss true

    --tokenized_mds_train "$dataset"

    # Streaming
    --toggle_type "$toggle_type"
    --sink_size "$sink_size"
    --topk_k "$topk_k"

    --enable_ada_sparsity "$enable_ada_sparsity"

    --pooling_mode "$pooling_mode"

    # Layer-wise sparsity
    --enable_layerwise_sparsity "$enable_layerwise_sparsity"
    --layerwise_sparsity_schedule "$layerwise_sparsity_schedule"
    --layerwise_sparsity_min_ratio "$layerwise_sparsity_min_ratio"
    --layerwise_sparsity_max_ratio "$layerwise_sparsity_max_ratio"
    --layerwise_sparsity_power "$layerwise_sparsity_power"
    --layerwise_sparsity_weight "$layerwise_sparsity_weight"
    --erank_analysis_path "/"
)

# FSDP
if [ "$fsdp" -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY="$fsdp"
    base_arguments+=( --fsdp "auto_wrap" )
    export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
fi

# Gradient checkpointing
if [ "$gc" -ne 0 ]; then
    base_arguments+=( --gradient_checkpointing )
fi

base_arguments+=( "$@" )

echo "Command: ${header} ${base_arguments[*]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a "$out_dir/log.out" \
    && [ -f "$out_dir/config.json" ] && python -m training.save_prulong_masks --checkpoint "$out_dir" --out_path "$out_dir/masks_sp${end_head_sparsity}.tsv" --sparsity "$end_head_sparsity"