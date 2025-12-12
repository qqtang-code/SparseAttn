import re
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from typing import Dict, Any, List

# --- 1. 配置路径和目标稀疏度 ---
# 将此变量设置为您的日志文件路径
# 注意: 实际运行需要替换成可访问的路径，此处仅为示例
log_file_path = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/masksonly_Qwen3-4B_bsz16_steps125_lr1e-5_warmup0.1_sp0.3_cw2048_mlr1.0_rlr1.0sft3_pretrain_64k_xattn_mlp_new*2_nolinear_first_token_5reg_32k_11.30_wfrozen/log.out" # 假设日志内容已作为文件log.out
print(f"假设日志文件路径为: {log_file_path}")

# 您提供的目标稀疏度配置
TASK_SPARSITY_CONFIG: Dict[str, Dict[str, float]] = {
    "Code": {"start": 0.0, "end": 0.4},
    "Math": {"start": 0.0, "end": 0.6},
    "MultiHop QA": {"start": 0.0, "end": 0.2},
    "Single QA": {"start": 0.0, "end": 0.2},
    "Summarization": {"start": 0.0, "end": 0.7},
    "default": {"start": 0.0, "end": 0.7}, # 假设 default 配置
}

# 假设的训练参数
TOTAL_STEPS = 1000  # 假设总步数为 1000
WARMUP_RATIO = 0.8   # 假设前 80% 的步数进行 Warmup/线性递增

# 2. 定义更新后的正则表达式
# 新正则 1 (Rank信息): 匹配: Rank #: [Step #] Task=['TASK_NAME'] | model_sparsity=tensor([SPARSITY_VALUE...
# 提取 group 1: step, group 2: task, group 3: sparsity
regex_rank = re.compile(r"^Rank \d+: \[Step (\d+)] Task=\['(.+?)'\] \| model_sparsity=tensor\(\[([\d\.]+)")
# 新正则 2 (INFO信息): 匹配: [INFO|...] ... 'model_sparsity': SPARSITY, 'step': #
# 提取 group 5: model_sparsity, group 6: step (用于补充全局稀疏度信息，本例中不用于任务绘图，但可备用)
regex_info = re.compile(r"^\[INFO\|.+\] .* >> \{'lm_loss': ([\d\.]+), 'reg_loss': ([\d\.]+), 'loss': ([\d\.]+), 'target_sparsity': ([\d\.]+), 'model_sparsity': ([\d\.]+), 'step': (\d+)")


def extract_and_process_log(log_content: str) -> pd.DataFrame | None:
    """从日志内容中提取数据并进行聚合处理"""

    data: List[Dict[str, Any]] = []
    lines = log_content.split('\n')
    
    for line in lines:
        # 尝试匹配 Rank 稀疏度信息 (优先级最高，包含任务名称)
        match_rank = regex_rank.match(line)
        if match_rank:
            step = int(match_rank.group(1))
            task = match_rank.group(2).strip()
            sparsity = float(match_rank.group(3))
            data.append({'step': step, 'task': task, 'model_sparsity': sparsity, 'source': 'Rank'})
            continue # 优先使用 Rank 数据，跳过 INFO 匹配

    if not data:
        print("未从日志中提取到任何 Model Sparsity 数据。")
        return None
        
    df = pd.DataFrame(data)

    # 聚合：计算每个 (Step, Task) 组合的平均 model_sparsity
    df_agg = df.groupby(['step', 'task'])['model_sparsity'].mean().reset_index()
    return df_agg

# (原脚本中 calculate_target_sparsity, plot_all_tasks_with_global_target, plot_individual_tasks 保持不变)
def calculate_target_sparsity(steps: np.ndarray, start: float, end: float, total_steps: int, warmup_ratio: float) -> np.ndarray:
    """计算目标稀疏度随 Step 变化的曲线"""
    
    warmup_steps = total_steps * warmup_ratio
    target_sparsity = []
    
    for step in steps:
        if step <= warmup_steps:
            # 线性递增 (Warmup)
            t = (step / warmup_steps)
            current_target = start + (end - start) * t
        else:
            # 保持 end 值
            current_target = end
        target_sparsity.append(current_target)
        
    return np.array(target_sparsity)

def plot_all_tasks_with_global_target(df_agg: pd.DataFrame, config: Dict[str, Dict[str, float]]):
    """绘制所有任务的 Model Sparsity 和全局 Target Sparsity 曲线"""
    print("\n--- 正在生成主图 ---")
    plt.figure(figsize=(14, 7))
    
    # 1. 计算全局 Target Sparsity (使用 default 配置)
    all_steps = np.arange(df_agg['step'].min(), df_agg['step'].max() + 1)
    
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
    output_filename = "all_tasks_sparsity_trend.png"
    plt.savefig(output_filename)
    plt.close() # 关闭当前图表，避免在 Jupyter 环境中重复显示
    print(f"✅ 主图（所有任务与全局Target）已保存为: {output_filename}")


def plot_individual_tasks(df_agg: pd.DataFrame, config: Dict[str, Dict[str, float]]):
    """为每个任务绘制单独的 Model Sparsity 及其目标范围图"""
    print("\n--- 正在生成子图 ---")
    tasks = df_agg['task'].unique()
    num_tasks = len(tasks)
    
    if num_tasks == 0:
        return

    cols = 2
    rows = int(np.ceil(num_tasks / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 5 * rows))
    
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
        current_config = config.get(task, config.get('default', {"start": 0.0, "end": 0.7}))
        start = current_config["start"]
        end = current_config["end"]

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
                label=f'Target (Start={start}, End={end})')

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
    output_filename = "individual_tasks_sparsity_plots.png"
    plt.savefig(output_filename)
    plt.close()
    print(f"✅ 子图（每个任务及其Target）已保存为: {output_filename}")

# --- 3. 执行主流程 ---
if __name__ == "__main__":    
    if not os.path.exists(log_file_path):
        print(f"致命错误：文件未找到在路径: {log_file_path}")
        # 如果文件不存在，立即退出
        exit() 

    # 1. 读取文件内容
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            log_content = f.read()
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
        exit()

    # 2. 传递文件内容进行处理
    aggregated_data = extract_and_process_log(log_content) 
    
    if aggregated_data is not None and not aggregated_data.empty:
        print("--- 提取和聚合的数据 (前几行) ---")
        # 使用 to_markdown 确保在终端输出时格式清晰
        print(aggregated_data.head().to_markdown(index=False))
        
        # 绘制图片
        plot_all_tasks_with_global_target(aggregated_data, TASK_SPARSITY_CONFIG)
        plot_individual_tasks(aggregated_data, TASK_SPARSITY_CONFIG)
    else:
        # 如果 aggregated_data 为空或 None，则可能是正则匹配失败
        print("未从日志中提取到 Model Sparsity 数据，可能是文件内容为空或正则表达式需要调整。")
        # 可以添加一些调试信息，例如打印 log_content 的长度
        print(f"日志内容长度: {len(log_content)}")