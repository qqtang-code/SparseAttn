import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 任务的目标稀疏度配置
task_configs = {
    "Code": {"start": 0.0, "end": 0.4},
    "MultiHop QA": {"start": 0.0, "end": 0.2},
    "Single QA": {"start": 0.0, "end": 0.2},
    "Summarization": {"start": 0.0, "end": 0.7},
}

LOG_PATH="/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/masksonly_Qwen3-4B_bsz16_steps125_lr1e-5_warmup0.1_sp0.3_cw2048_mlr1.0_rlr1.0sft3_pretrain_64k_xattn_mlp_new*2_nolinear_first_token_5reg_32k_11.30_debug_wfrozen/log.out"

TOTAL_STEPS = 125
WARMUP_RATIO = 0.8
WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_RATIO)

# ================= 1. 数据提取 =================
def extract_log_data(log_text):
    # 正则表达式匹配：Rank X: [Step Y] Task=['Name'] | model_sparsity=tensor([Value]...
    pattern = r"Rank \d+: \[Step (\d+)\] Task=\['(.*?)'\] \| model_sparsity=tensor\(\[(.*?)\]"
    
    # 数据结构: data[step][task] = [sparsity1, sparsity2, ...]
    data = defaultdict(lambda: defaultdict(list))
    
    for line in log_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            step = int(match.group(1))
            task = match.group(2)
            sparsity = float(match.group(3))
            data[step][task].append(sparsity)
    
    # 计算每个Step每个Task的平均值
    avg_data = defaultdict(dict)
    for step, tasks in data.items():
        for task, values in tasks.items():
            avg_data[step][task] = np.mean(values)
            
    return avg_data

# ================= 2. 辅助函数 =================
def get_target_sparsity(step, task_name):
    """计算指定步数的各项任务 Target Sparsity (线性增长)"""
    cfg = task_configs.get(task_name)
    if not cfg:
        return 0.0
    
    if step >= WARMUP_STEPS:
        return cfg['end']
    
    # 线性增长公式
    progress = step / WARMUP_STEPS
    return cfg['start'] + (cfg['end'] - cfg['start']) * progress

# ================= 3. 绘图逻辑 =================
def plot_results(avg_data):
    # 获取所有出现的 Step 并排序
    extracted_steps = sorted(avg_data.keys())
    tasks = list(task_configs.keys())
    
    # 用于绘制 Target 曲线的 X 轴 (1 到 125)
    all_theoretical_steps = np.arange(1, TOTAL_STEPS + 1)

    # --- 图表 1: 所有任务的 Model Sparsity 变化 ---
    plt.figure(figsize=(12, 6))
    for task in tasks:
        # 提取该任务在每一步的实际数据
        x = []
        y = []
        for s in extracted_steps:
            if task in avg_data[s]:
                x.append(s)
                y.append(avg_data[s][task])
        
        if x:
            plt.plot(x, y, marker='o', label=task)
            
    plt.title('Actual Model Sparsity per Task (Averaged over Ranks)')
    plt.xlabel('Step')
    plt.ylabel('Model Sparsity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_sparsity_overview.png')
    print("生成图表: model_sparsity_overview.png")

    # --- 图表 2: 每个任务 Actual vs Target ---
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    axs = axs.flatten()
    
    for i, task in enumerate(tasks):
        ax = axs[i]
        
        # 1. 绘制 Target Sparsity (虚线，理论值)
        targets = [get_target_sparsity(s, task) for s in all_theoretical_steps]
        ax.plot(all_theoretical_steps, targets, linestyle='--', color='gray', label='Target Sparsity', alpha=0.6)
        
        # 2. 绘制 Actual Model Sparsity (实线/点，提取值)
        x = []
        y = []
        for s in extracted_steps:
            if task in avg_data[s]:
                x.append(s)
                y.append(avg_data[s][task])
        
        if x:
            ax.plot(x, y, marker='o', color='blue', linewidth=2, label='Model Sparsity')
            # # 在数据点旁标注数值
            # for sx, sy in zip(x, y):
            #     ax.text(sx, sy, f'{sy:.4f}', fontsize=9, ha='right', va='bottom')

        ax.set_title(f'Task: {task}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Sparsity')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('sparsity_comparison.png')
    print("生成图表: sparsity_comparison.png")

# ================= 主程序 =================
if __name__ == "__main__":
    # 解析数据
    # 注意：实际使用时，你可以取消下行注释来读取文件
    with open(LOG_PATH, 'r') as f: log_content = f.read()
    
    # 只要将你的log粘贴到 log_content 变量中即可 (如果上面没有读文件)
    # 为了演示，这里假设 log_content 已经被填充
    
    processed_data = extract_log_data(log_content)
    
    # 打印Step 1的提取结果示例
    print("Extraction Preview (Step 1):")
    if 1 in processed_data:
        for task, val in processed_data[1].items():
            print(f"  {task}: {val:.5f}")
            
    # 绘图
    plot_results(processed_data)