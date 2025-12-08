import re
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

# 任务的目标稀疏度配置
task_configs = {
    "Code": {"start": 0.0, "end": 0.4},
    "MultiHop QA": {"start": 0.0, "end": 0.2},
    "Single QA": {"start": 0.0, "end": 0.2},
    "Summarization": {"start": 0.0, "end": 0.5},
}

LOG_PATH = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/checkpoints/steps125_qwen_mix_sft_32K_xattn_mlp_linear_first_token_10reg_nolambda_abs*100_head_contrast_wfrozen/log.out"

TOTAL_STEPS = 125
WARMUP_RATIO = 0.8
WARMUP_STEPS = int(TOTAL_STEPS * WARMUP_RATIO)

# ================= 1. 数据提取 (已修改) =================
def extract_log_data(log_text):
    # 正则表达式说明：
    # group(1): Step
    # group(2): Task 列表内部字符串，如 "'MultiHop QA', 'Summarization'"
    # group(3): Sparsity 列表内部字符串，如 "0.3941, 0.4002"
    pattern = r"Rank \d+: \[Step (\d+)\] Task=\['(.*?)'\] \| model_sparsity=tensor\(\[(.*?)\]"
    
    # 数据结构: data[step][task] = [sparsity1, sparsity2, ...]
    data = defaultdict(lambda: defaultdict(list))
    
    for line in log_text.split('\n'):
        match = re.search(pattern, line)
        if match:
            try:
                step = int(match.group(1))
                task_str = match.group(2)
                sparsity_str = match.group(3)

                # 1. 解析任务列表：按逗号分割，去除首尾空格和引号
                tasks = [t.strip().strip("'").strip('"') for t in task_str.split(',')]
                
                # 2. 解析稀疏度列表：按逗号分割，转为 float
                sparsities = [float(s.strip()) for s in sparsity_str.split(',')]

                # 3. 一一对应并存储
                # 只有当任务数量和稀疏度数量一致时才记录，防止日志打印截断导致错误
                if len(tasks) == len(sparsities):
                    for t, s in zip(tasks, sparsities):
                        data[step][t].append(s)
            except ValueError:
                # 忽略解析错误的行
                continue
    
    # 计算每个Step每个Task的平均值
    avg_data = defaultdict(dict)
    for step, tasks_dict in data.items():
        for task, values in tasks_dict.items():
            avg_data[step][task] = np.mean(values)
            
    return avg_data

# ================= 2. 辅助函数 (保持不变) =================
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

# ================= 3. 绘图逻辑 (保持不变) =================
def plot_results(avg_data):
    # 获取所有出现的 Step 并排序
    extracted_steps = sorted(avg_data.keys())
    if not extracted_steps:
        print("未提取到任何数据，请检查日志格式或路径。")
        return

    tasks = list(task_configs.keys())
    
    # 用于绘制 Target 曲线的 X 轴
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
            plt.plot(x, y, marker='.', label=task) # 改用'.'减少密集点的大小
            
    plt.title('Actual Model Sparsity per Task (Averaged over Ranks)')
    plt.xlabel('Step')
    plt.ylabel('Model Sparsity')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('model_sparsity_overview_batch.pdf')
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
            ax.plot(x, y, marker='.', color='blue', linewidth=1, label='Model Sparsity')

        ax.set_title(f'Task: {task}')
        ax.set_xlabel('Step')
        ax.set_ylabel('Sparsity')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('sparsity_comparison_batch.pdf')
    print("生成图表: sparsity_comparison.png")

# ================= 主程序 =================
if __name__ == "__main__":
    # 读取文件
    try:
        with open(LOG_PATH, 'r') as f: 
            log_content = f.read()
            processed_data = extract_log_data(log_content)
            
            # 打印Step 0或1的提取结果示例，用于调试
            check_step = 0 if 0 in processed_data else 1
            print(f"Extraction Preview (Step {check_step}):")
            if check_step in processed_data:
                for task, val in processed_data[check_step].items():
                    print(f"  {task}: {val:.5f}")
            
            # 绘图
            plot_results(processed_data)
    except FileNotFoundError:
        print(f"错误: 找不到文件 {LOG_PATH}")