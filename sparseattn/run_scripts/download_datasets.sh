#!/bin/bash

# 基础路径
BASE_DIR="/data2/public_data/"

# 数据集列表
datasets=(
    "LCM_group/qwen_mix_sft_128K6"
    "LCM_group/qwen_mix_sft_64K6"
    "LCM_group/qwen_mix_sft_32K6"
    "LCM_group/llama_mix_sft_128K6"
    "LCM_group/llama_mix_sft_64K6"
    "LCM_group/llama_mix_sft_32K6"
)

# 循环下载
for dataset in "${datasets[@]}"; do
    # 提取数据集名称 (例如 qwen_mix_sft_128K6)
    dataset_name=$(basename "$dataset")
    
    # 拼接目标路径
    target_dir="$BASE_DIR/$dataset_name"
    
    echo "⬇️  正在下载 $dataset 到 $target_dir ..."

    modelscope download --dataset "$dataset" --local_dir "$target_dir"
    
    echo "✅ $dataset_name 下载完成"
    echo "----------------------------------------"
done