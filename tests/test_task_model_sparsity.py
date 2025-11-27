import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# --- 1. 配置路径和目标稀疏度 ---
# 将此变量设置为您的日志文件路径
log_file_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/masksonly_Qwen3-4B_bsz16_steps1000_lr1e-5_warmup0.1_sp0.3_cw2048_mlr1.0_rlr1.0sft3_64k_pretrain_xattn_adarouter_20reg_nolinear_specialtoken_newtasksparsity_11.25_wfrozen/log.out"

# 您提供的目标稀疏度配置
TASK_SPARSITY_CONFIG = {
    "Code": {"start": 0.1, "end": 0.7},
    "Math": {"start": 0.0, "end": 0.6},
    "MultiHop QA": {"start": 0.2, "end": 0.5},
    "Single QA": {"start": 0.1, "end": 0.7},
    "Summarization": {"start": 0.3, "end": 0.6},
    "default": {"start": 0.3, "end": 0.6}, # 假设 default 配置
}

# 假设的训练参数
TOTAL_STEPS = 1000  # 假设总步数为 1000
WARMUP_RATIO = 0.8   # 假设前 80% 的步数进行 Warmup/线性递增

# 2. 定义正则表达式
# 匹配: Rank #: [Step #] Task=TASK_NAME | model_sparsity=SPARSITY_VALUE
regex = re.compile(r"^Rank \d+: \[Step (\d+)] Task=(.+) \| model_sparsity=([\d\.]+)")

def extract_and_process_log(file_path):
    """从日志文件中提取数据并进行聚合处理"""
    
    if not os.path.exists(file_path):
        print(f"错误：文件未找到在路径: {file_path}")
        return None

    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = regex.match(line)
                if match:
                    step = int(match.group(1))
                    task = match.group(2).strip()
                    sparsity = float(match.group(3))
                    data.append({'step': step, 'task': task, 'model_sparsity': sparsity})
    except Exception as e:
        print(f"读取或处理文件时发生错误: {e}")
        return None

    if not data:
        print("未从日志中提取到 Model Sparsity 数据。")
        return None
        
    df = pd.DataFrame(data)

    # 聚合：计算每个 (Step, Task) 组合的平均 model_sparsity
    df_agg = df.groupby(['step', 'task'])['model_sparsity'].mean().reset_index()
    return df_agg

def calculate_target_sparsity(steps, start, end, total_steps, warmup_ratio):
    """计算目标稀疏度随 Step 变化的曲线"""
    
    warmup_steps = total_steps * warmup_ratio
    target_sparsity = []
    
    for step in steps:
        if step <= warmup_steps:
            # 线性递增
            t = (step / warmup_steps)
            current_target = start + (end - start) * t
        else:
            # 保持 end 值
            current_target = end
        target_sparsity.append(current_target)
        
    return np.array(target_sparsity)

def plot_all_tasks_with_global_target(df_agg, config):
    """绘制所有任务的 Model Sparsity 和全局 Target Sparsity 曲线"""

    plt.figure(figsize=(14, 7))
    
    # 1. 计算全局 Target Sparsity (使用 default 配置)
    # 找出所有 Step 以绘制完整的 Target 曲线
    all_steps = np.arange(df_agg['step'].min(), TOTAL_STEPS + 1)
    
    # 使用 'default' 配置来计算全局 Target
    default_config = config.get('default', {"start": 0.0, "end": 0.7}) 
    target_curve = calculate_target_sparsity(
        all_steps, 
        default_config["start"], 
        default_config["end"], 
        TOTAL_STEPS, 
        WARMUP_RATIO
    )
    
    # 绘制全局 Target Sparsity 曲线
    plt.plot(all_steps, target_curve, color='red', linestyle='--', linewidth=2, 
             label=f'Global Target Sparsity (Start={default_config["start"]}, End={default_config["end"]})')

    # 2. 绘制所有任务的 Model Sparsity
    tasks = df_agg['task'].unique()
    for task in tasks:
        task_data = df_agg[df_agg['task'] == task]
        plt.plot(task_data['step'], task_data['model_sparsity'], 
                 marker='.', linestyle='-', alpha=0.8, label=f'{task} Model Sparsity')

    plt.title('All Tasks Model Sparsity vs. Global Target Sparsity', fontsize=16)
    plt.xlabel('Step', fontsize=14)
    plt.ylabel('Average Model Sparsity', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(title='Series', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    # 保存图片
    output_filename = os.path.basename(log_file_path).replace(".out", "_global_sparsity_trend.png")
    plt.savefig(output_filename)
    print(f"\n✅ 主图（所有任务与全局Target）已保存为: {output_filename}")
    plt.show()

def plot_individual_tasks(df_agg, config):
    """为每个任务绘制单独的 Model Sparsity 及其目标范围图"""
    
    tasks = df_agg['task'].unique()
    num_tasks = len(tasks)
    
    if num_tasks == 0:
        return

    # 设置子图布局：最多 2 列
    cols = 2
    rows = int(np.ceil(num_tasks / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    
    # 如果只有一个任务，axes 不是数组，需要特殊处理
    if num_tasks == 1:
        axes = np.array([axes])
        
    axes = axes.flatten()

    all_steps_min = df_agg['step'].min()
    all_steps_max = df_agg['step'].max()

    for i, task in enumerate(tasks):
        ax = axes[i]
        task_data = df_agg[df_agg['task'] == task]
        
        # 1. 绘制该任务的 Model Sparsity
        ax.plot(task_data['step'], task_data['model_sparsity'], 
                marker='o', linestyle='-', color='blue', label=f'{task} Model Sparsity')
        
        # 2. 绘制该任务的 Target Range (Start/End)
        current_config = config.get(task, config.get('default'))
        start = current_config["start"]
        end = current_config["end"]

        # 绘制 Target Range 区域 (Start/End)
        # 计算 Target Sparsity 曲线
        task_steps_range = np.arange(all_steps_min, all_steps_max + 1)
        task_target_curve = calculate_target_sparsity(
            task_steps_range, 
            start, 
            end, 
            TOTAL_STEPS, 
            WARMUP_RATIO
        )
        
        ax.plot(task_steps_range, task_target_curve, 
                color='red', linestyle='--', alpha=0.7, linewidth=2, 
                label=f'Target Sparsity (Start={start}, End={end})')

        ax.set_title(f'Task: {task}', fontsize=14)
        ax.set_xlabel('Step')
        ax.set_ylabel('Model Sparsity')
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.legend(loc='best')

    # 隐藏未使用的子图
    for j in range(num_tasks, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle('Individual Task Model Sparsity vs. Target Sparsity', fontsize=18, y=1.02)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    # 保存图片
    output_filename = os.path.basename(log_file_path).replace(".out", "_individual_sparsity_plots.png")
    plt.savefig(output_filename)
    print(f"✅ 子图（每个任务及其Target）已保存为: {output_filename}")
    plt.show()

# --- 3. 执行主流程 ---
if __name__ == "__main__":
    aggregated_data = extract_and_process_log(log_file_path)
    
    if aggregated_data is not None and not aggregated_data.empty:
        print("--- 提取和聚合的数据 (前几行) ---")
        print(aggregated_data.head())
        
        # 绘制主图：所有任务的 Model Sparsity 和全局 Target Sparsity
        plot_all_tasks_with_global_target(aggregated_data, TASK_SPARSITY_CONFIG)
        
        # 绘制子图：每个任务的 Model Sparsity 和该任务的 Target Sparsity 曲线
        plot_individual_tasks(aggregated_data, TASK_SPARSITY_CONFIG)