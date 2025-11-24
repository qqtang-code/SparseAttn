# Model and training configuration
model=${MODEL:-"/data1/hf_model/Qwen3-4B"}

bsz=${BSZ:-8}
seq=${SEQ:-1}
lr=${LR:-1e-5}
steps=${STEPS:-1000}
save_steps=${SAVE:-500}
warmup=${WARMUP:-0.1}
suffix=${SUFFIX:-""}
overrides=${OVERRIDES:-""}
min_lr_ratio=${MIN_LR_RATIO:-0.01}
seq_parallel_size=${SEQ_PARALLEL_SIZE:-1}

# FSDP configuration
# 0=Disable, 1=FULL_SHARD, 2=SHARD_GRAD_OP, 3=NO_SHARD, 4=HYBRID_SHARD, 5=HYBRID_SHARD_ZERO2
fsdp=${FSDP:-"0"}
gc=${GC:-"1"}

# PruLong-specific arguments
# max_toks=${MAX_TOKS:-32768}
max_toks=${MAX_TOKS:-32768}


attn_type=${ATTN_TYPE:-"nsa"}

# Dataset configuration
# dataset=${DATASET:-"/data/lcm_lab/qqt/project/SparseAttn/sparseattn/data"}
dataset=${DATASET:-"/data1/public_data/Pre_filter"}

# Create run name
extra_name="update_compress_key_compress_value_gate_only_blocksize_128_topk_64"

run_name="nsa_qwen3_$(basename $model)_bsz${bsz}_steps${steps}_lr${lr}_warmup${warmup}_${extra_name}${suffix}"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
out_dir="checkpoints/$run_name"
mkdir -p $out_dir
nvidia-smi

# Calculate GPU and node configuration
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    num_gpus=$(nvidia-smi -L | wc -l)
else
    num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
fi

num_gpus=${NUM_GPUS:-$num_gpus}

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
    -m training.nsa_train"
else
    master_port=$(comm -23 <(seq 49152 65535 | sort) <(ss -Htan | awk '{print $4}' | cut -d':' -f2 | sort -u) | shuf | head -n 1)

    header="torchrun \
    --rdzv-backend=c10d \
    --rdzv-endpoint=localhost:$master_port \
    --nnodes=1 \
    --nproc-per-node=$num_gpus \
    -m training.nsa_train"
fi

# accu=$(($bsz / $seq / $num_gpus / $num_nodes))
accu=1

echo "num_nodes=${num_nodes} master_addr=${master_addr} master_port=${master_port} num_gpus=${num_gpus}"

# Environment variables
export OMP_NUM_THREADS=$num_gpus
export SWANLAB_API_KEY="t0PmOeLpVom1LRBDAKHaA"
export SWANLAB_LOG_DIR=$out_dir
export SWANLAB_MODE="cloud"
export TOKENIZERS_PARALLELISM=true
export LOGIT_BLOCK_SIZE=2048

# Training arguments
base_arguments=(
    --report_to swanlab
    --do_train

    --model_name_or_path $model
    --tokenizer_name $model

    --run_name $run_name
    --output_dir $out_dir
    --config_overrides_json "$overrides"
    --gradient_accumulation_steps $accu
    --per_device_train_batch_size $seq
    --per_device_eval_batch_size $seq
    --per_device_max_tokens $max_toks

    --bf16
    --learning_rate $lr
    --min_lr_ratio $min_lr_ratio
    --lr_scheduler_type cosine
    --max_grad_norm 1.0
    --adam_beta1 0.9
    --adam_beta2 0.95
    --weight_decay 0.1
    --warmup_ratio $warmup
    --optim adamw_torch

    --logging_steps 1
    --log_level info

    --max_steps $steps
    --save_steps $save_steps
    --dataloader_num_workers 1

    --disable_tqdm true
    --use_fast_tokenizer false
    --remove_unused_columns false
    --ddp_find_unused_parameters false

    --cuda_empty_cache

    --tokenized_mds_train $dataset

    --attention_type $attn_type
    --deepspeed "/data1/lcm_lab/yy/checkpoint/ds_config_stage2.json"
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


# 生成带时间戳的日志文件名，例如：log_20251110_143022.out
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="$out_dir/log_${timestamp}.out"

echo "Command: ${header} ${base_arguments[@]}"
${header} "${base_arguments[@]}" 2>&1 | tee -a "$log_file" \
  && [ -f "$out_dir/config.json" ]