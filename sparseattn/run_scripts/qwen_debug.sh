# Model and training configuration
model=${MODEL:-"/data2/hf_models/Qwen3-4B"}
bsz=${BSZ:-1}
seq=${SEQ:-1}
lr=${LR:-1e-5}
steps=${STEPS:-266}
save_steps=${SAVE:-133}
save_total_limit=10
warmup=${WARMUP:-0.3}

overrides=${OVERRIDES:-""}
min_lr_ratio=${MIN_LR_RATIO:-1e-7}
seq_parallel_size=${SEQ_PARALLEL_SIZE:-1}

# FSDP configuration
# 0=Disable, 1=FULL_SHARD, 2=SHARD_GRAD_OP, 3=NO_SHARD, 4=HYBRID_SHARD, 5=HYBRID_SHARD_ZERO2
fsdp=${FSDP:-"5"}
gc=${GC:-"1"}

# PruLong-specific arguments
# max_toks=${MAX_TOKS:-65536}
max_toks=${MAX_TOKS:-32768}
# max_toks=${MAX_TOKS:-256}
start_head_sparsity=${START_HEAD_SPARSITY:-0.0}
end_head_sparsity=${END_HEAD_SPARSITY:-0.3}
mask_learning_rate=${MASK_LEARNING_RATE:-1e-3}
reg_learning_rate=${REG_LEARNING_RATE:-1e-3}
sparsity_warmup_ratio=${SPARSITY_WARMUP_RATIO:-0.0}
disable_linear_reg_term=${DISABLE_LINEAR_REG_TERM:-false}
# topk
context_window_if_toggled=${CONTEXT_WINDOW_IF_TOGGLED:-2048}
freeze_weights=${FREEZE_WEIGHTS:-true}
freeze_masks=${FREEZE_MASKS:-false}
warmup_type=${WARMUP_TYPE:-"linear"}

# Streaming configuration
toggle_type=${TOGGLE_TYPE:-"xattn"}
retrieval_mode=${RETRIEVAL_MODE:-"full"} # "full","xattn"
sink_size=${SINK_SIZE:-128}
topk_k=${TOPK_K:-2048}

enable_ada_sparsity=${ENABLE_ADA_SPARSITY:-true}

# Layer-wise sparsity configuration
enable_layerwise_sparsity=${ENABLE_LAYERWISE_SPARSITY:-false}
layerwise_sparsity_schedule=${LAYERWISE_SPARSITY_SCHEDULE:-"high-low-high"}
layerwise_sparsity_min_ratio=${LAYERWISE_SPARSITY_MIN_RATIO:-0.5}
layerwise_sparsity_max_ratio=${LAYERWISE_SPARSITY_MAX_RATIO:-1.0}
layerwise_sparsity_power=${LAYERWISE_SPARSITY_POWER:-1.0}
layerwise_sparsity_weight=${LAYERWISE_SPARSITY_WEIGHT:-1.0}
erank_analysis_path="/"

# Dataset configuration
dataset=${DATASET:-"/data2/public_data/for_debug_mix_sft_64k"}
dataset_cache_dir="/data2/public_data/data_cache"
# dataset=${DATASET:-"/data1/public_data/Pre_filter"}
task_type="sft" # pretrain or sft

pooling_mode="ctx_q" # first_token,mean_all,ctx,q,ctx_q
enable_contrastive_loss=false
use_task_emb_for_mask=false
enable_lambda_task=false
use_softmax=true

# Create run name
suffix=${SUFFIX:-"debug_longbench"}
extra_name="full_xattn_32k_qwen3-4b"
# extra_name="debug_12.5"
if [[ $freeze_weights == "true" ]]; then
    extra_name="${extra_name}_wfrozen"
fi
if [[ $freeze_masks == "true" ]]; then
    extra_name="${extra_name}_mfrozen"
fi

run_name="${suffix}steps${steps}_${extra_name}"

export CUDA_VISIBLE_DEVICES=5
out_dir="checkpoints/$run_name"
mkdir -p $out_dir
nvidia-smi

# Calculate GPU and node configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi
num_gpus=${NUM_GPUS_PER_NODE:-$num_gpus}

num_nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST" 2>/dev/null | wc -l)
if [ $num_nodes == 0 ]; then
    num_nodes=1
fi
num_nodes=${NUM_NODES:-$num_nodes}

# Setup distributed training
if [ $num_nodes -gt 1 ]; then
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

accu=$(($bsz / $seq / $num_gpus / $num_nodes))
# accu=1

echo "num_nodes=${num_nodes} master_addr=${master_addr} master_port=${master_port} num_gpus=${num_gpus}"

# Environment variables
export OMP_NUM_THREADS=$num_gpus
export SWANLAB_API_KEY="g5vUmp1WaDMSV9FNveypn"
export SWANLAB_LOG_DIR=$out_dir
export SWANLAB_MODE="cloud"
export TOKENIZERS_PARALLELISM=true
export LOGIT_BLOCK_SIZE=2048

# Training arguments
base_arguments=(
    --report_to swanlab
    --do_train

    --model_name $model
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --config_overrides_json "$overrides"
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size $seq

    --bf16
    --learning_rate $lr
    --min_lr_ratio $min_lr_ratio
    --lr_scheduler_type cosine
    --max_grad_norm 5.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --warmup_ratio $warmup
    --optim adamw_torch

    --logging_steps 1
    --log_level info

    --max_steps $steps
    --save_steps $save_steps
    --save_total_limit $save_total_limit
    --dataloader_num_workers 1

    --data_cache_dir $dataset_cache_dir

    --disable_tqdm true
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false

    --cuda_empty_cache

    # PruLong-specific arguments
    --per_device_max_tokens $max_toks
    --task_type $task_type
    --seq_parallel_size $seq_parallel_size
    --start_head_sparsity $start_head_sparsity
    --end_head_sparsity $end_head_sparsity
    --mask_learning_rate $mask_learning_rate
    --reg_learning_rate $reg_learning_rate
    --warmup_type $warmup_type
    --sparsity_warmup_ratio $sparsity_warmup_ratio
    --disable_linear_regularization_term $disable_linear_reg_term
    --context_window_if_toggled $context_window_if_toggled
    --freeze_non_mask_parameters $freeze_weights
    --freeze_mask_parameters $freeze_masks
    --should_log_loss true
    --save_total_limit 3

    --tokenized_mds_train $dataset

    # Streaming configuration
    --toggle_type $toggle_type
    --sink_size $sink_size
    --topk_k $topk_k

    --enable_ada_sparsity $enable_ada_sparsity

    --retrieval_mode $retrieval_mode

    --pooling_mode $pooling_mode
    --enable_contrastive_loss $enable_contrastive_loss
    --use_task_emb_for_mask $use_task_emb_for_mask
    --enable_lambda_task $enable_lambda_task
    --use_softmax $use_softmax

    # layer decay configuration
    --enable_layerwise_sparsity $enable_layerwise_sparsity
    --layerwise_sparsity_schedule $layerwise_sparsity_schedule
    --layerwise_sparsity_min_ratio $layerwise_sparsity_min_ratio
    --layerwise_sparsity_max_ratio $layerwise_sparsity_max_ratio
    --layerwise_sparsity_power $layerwise_sparsity_power
    --layerwise_sparsity_weight $layerwise_sparsity_weight
    --erank_analysis_path $erank_analysis_path
)

# FSDP configuration
if [ $fsdp -ne 0 ]; then
    export FSDP_SHARDING_STRATEGY=$fsdp
    base_arguments+=( --fsdp "auto_wrap" )
    export FSDP_STATE_DICT_TYPE="FULL_STATE_DICT"
fi

# Gradient checkpointing
if [ $gc -ne 0 ]; then
    base_arguments+=( --gradient_checkpointing )
fi

base_arguments+=( $@ )

echo "Command: ${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a $out_dir/log.out \
    && [ -f $out_dir/config.json ] && python -m training.save_prulong_masks --checkpoint $out_dir --out_path $out_dir/masks_sp${end_head_sparsity}.tsv --sparsity $end_head_sparsity 