#!/bin/bash

# --- 1. 初始化 Conda 环境 ---
# 自动获取 conda 安装路径并加载初始化脚本
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"

# --- 2. 切换到 sora 环境 ---
echo "正在切换到 sora 环境..."
conda activate sora

# 检查环境是否切换成功
if [ "$CONDA_DEFAULT_ENV" != "sora" ]; then
    echo "错误: 未能成功激活 sora 环境，请检查环境名是否正确。"
    exit 1
fi

# --- 3. 运行 Python 脚本 ---
echo "正在启动脚本: run_sparse.py"
python /data1/lcm_lab/sora/sparseattn_evaluation/scripts/run_sparse.py

echo "任务运行结束。"