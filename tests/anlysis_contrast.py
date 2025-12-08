import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# 配置
DUMP_DIR = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/nolambda_abs"
SAVE_IMG_PATH = "training_dashboard.png"

def load_data():
    files = glob.glob(os.path.join(DUMP_DIR, "batch_*.pt"))
    # 按时间戳排序，确保时间轴正确
    files.sort(key=os.path.getmtime)
    
    steps = []
    losses = []
    off_diag_means = []
    diag_means = []
    latest_matrix = None
    latest_tasks = None

    print(f"Found {len(files)} batch dumps. Processing...")

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')
            
            # 提取 step (文件名格式: batch_stepX_timestamp.pt)
            # 如果文件名里包含 step，优先用文件名，因为此时 data['global_hidden'] 可能不包含 step 信息
            match = re.search(r'batch_step(\d+)_', os.path.basename(f))
            step_num = int(match.group(1)) if match else 0
            
            # 提取 Loss
            loss = data.get('loss_val', None)
            if loss is None: continue
            
            # 计算矩阵统计量
            unique_tasks = data['unique_tasks']
            prototypes = data['prototypes']
            
            if prototypes is None or len(unique_tasks) < 2:
                continue

            P = F.normalize(prototypes.float(), dim=1)
            sim_mat = torch.mm(P, P.t()).numpy()
            
            # 对角线均值 (Self-Similarity)
            d_mean = np.mean(np.diag(sim_mat))
            
            # 非对角线均值 (Cross-Task Similarity) -> 越低越好
            mask = ~np.eye(sim_mat.shape[0], dtype=bool)
            off_mean = np.mean(sim_mat[mask])
            
            steps.append(step_num)
            losses.append(loss)
            diag_means.append(d_mean)
            off_diag_means.append(off_mean)
            
            latest_matrix = sim_mat
            latest_tasks = unique_tasks.numpy()
            
        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue
            
    return steps, losses, diag_means, off_diag_means, latest_matrix, latest_tasks

def plot_dashboard():
    steps, losses, diag_means, off_diag_means, latest_matrix, latest_tasks = load_data()
    
    if not steps:
        print("No valid data found yet.")
        return

    # 创建画布
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2)

    # --- 图 1: Loss 曲线 ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(losses, color='tab:red', label='Contrastive Loss')
    ax1.set_title("Contrastive Loss (Lower is Better)")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # --- 图 2: 分离度曲线 (最重要) ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(off_diag_means, color='tab:blue', label='Cross-Task Similarity (Off-Diag)')
    ax2.plot(diag_means, color='tab:green', linestyle='--', label='Self-Similarity (Diag)')
    ax2.set_title("Task Separation Metric")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_xlabel("Micro-Batch Index (Time)")
    ax2.axhline(y=0.0, color='gray', linestyle=':', alpha=0.5) # 0线
    ax2.legend()
    
    # 标注状态
    current_sep = off_diag_means[-1]
    status = "BAD" if current_sep > 0.8 else ("OK" if current_sep > 0.3 else "GOOD")
    color = "red" if status == "BAD" else ("orange" if status == "OK" else "green")
    ax2.text(0.02, 0.05, f"Current Sep: {current_sep:.2f} ({status})", 
             transform=ax2.transAxes, color=color, weight='bold', fontsize=12)

    # --- 图 3: 最新矩阵热力图 ---
    ax3 = fig.add_subplot(gs[:, 1])
    if latest_matrix is not None:
        im = ax3.imshow(latest_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax3.set_title(f"Latest Prototype Similarity (Step {steps[-1]})")
        
        # 添加数值标签
        for i in range(len(latest_tasks)):
            for j in range(len(latest_tasks)):
                text = ax3.text(j, i, f"{latest_matrix[i, j]:.2f}",
                               ha="center", va="center", color="black", fontsize=9)
        
        ax3.set_xticks(np.arange(len(latest_tasks)))
        ax3.set_yticks(np.arange(len(latest_tasks)))
        ax3.set_xticklabels(latest_tasks)
        ax3.set_yticklabels(latest_tasks)
        plt.colorbar(im, ax=ax3)
    
    plt.tight_layout()
    plt.savefig(SAVE_IMG_PATH)
    print(f"✅ Dashboard saved to {SAVE_IMG_PATH}")
    print(f"   Latest Cross-Sim: {off_diag_means[-1]:.4f} (Goal: < 0.1)")

if __name__ == "__main__":
    plot_dashboard()