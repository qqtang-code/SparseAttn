import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import re

# 配置
DUMP_DIR = "/data1/lcm_lab/qqt/SparseAttn/sparseattn/nolambda_abs*100"
SAVE_IMG_PATH = "training_dashboard.png"

def load_data():
    files = glob.glob(os.path.join(DUMP_DIR, "batch_*.pt"))
    files.sort(key=os.path.getmtime)  # 时间排序

    steps = []
    losses = []
    off_diag_means = []
    diag_means = []
    latest_matrix = None
    latest_tasks = None

    # 新增 Hidden states 分析
    hidden_inter = []  # cross-task similarity
    hidden_intra = []  # within-task similarity

    print(f"Found {len(files)} batch dumps. Processing...")

    for f in files:
        try:
            data = torch.load(f, map_location='cpu')

            # 解析 step
            match = re.search(r'batch_step(\d+)_', os.path.basename(f))
            step_num = int(match.group(1)) if match else 0

            # Loss
            loss = data.get('loss_val', None)
            if loss is None:
                continue

            steps.append(step_num)
            losses.append(loss)

            # Prototype similarity
            unique_tasks = data.get('unique_tasks', None)
            prototypes = data.get('prototypes', None)

            if prototypes is not None and unique_tasks is not None and len(unique_tasks) >= 2:
                P = F.normalize(prototypes.float(), dim=1)
                sim_mat = torch.mm(P, P.t()).numpy()

                diag_means.append(np.mean(np.diag(sim_mat)))

                mask = ~np.eye(sim_mat.shape[0], dtype=bool)
                off_diag_means.append(np.mean(sim_mat[mask]))

                latest_matrix = sim_mat
                latest_tasks = unique_tasks.numpy()
            else:
                diag_means.append(None)
                off_diag_means.append(None)

            # ================================================================
            # Added: Hidden state cosine similarity (direct sample-based)
            # ================================================================
            global_hidden = data.get("global_hidden", None)
            global_task = data.get("global_task", None)

            if global_hidden is not None and global_task is not None:
                H = F.normalize(torch.tensor(global_hidden).float(), dim=1)  # [N, D]
                T = torch.tensor(global_task)

                inter_sims = []
                intra_sims = []

                task_list = torch.unique(T)

                for ti in task_list:
                    hi = H[T == ti]

                    # intra-task similarity
                    if hi.shape[0] > 1:
                        S = (hi @ hi.t()).numpy()
                        mask = ~np.eye(S.shape[0], dtype=bool)
                        intra_sims.append(S[mask].mean())

                    # inter-task similarity
                    for tj in task_list:
                        if tj <= ti:
                            continue
                        hj = H[T == tj]

                        if hi.shape[0] > 0 and hj.shape[0] > 0:
                            S = (hi @ hj.t()).numpy()
                            inter_sims.append(S.mean())

                hidden_inter.append(np.mean(inter_sims) if len(inter_sims) else None)
                hidden_intra.append(np.mean(intra_sims) if len(intra_sims) else None)

            else:
                hidden_inter.append(None)
                hidden_intra.append(None)

        except Exception as e:
            print(f"Error reading {f}: {e}")
            continue

    return (steps, losses, diag_means, off_diag_means,
            latest_matrix, latest_tasks, hidden_inter, hidden_intra)


def plot_dashboard():
    (steps, losses, diag_means, off_diag_means,
     latest_matrix, latest_tasks, hidden_inter, hidden_intra) = load_data()

    if not steps:
        print("No valid data found.")
        return

    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 2)

    # ---------------- LOSS
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(losses, color='tab:red', label='Contrastive Loss')
    ax1.set_title("Contrastive Loss")
    ax1.set_ylabel("Loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # ---------------- Prototype Similarity
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(off_diag_means, label='Cross-Task Prototype CosSim', color='tab:blue')
    ax2.plot(diag_means, label='Self-Similarity Prototype', color='tab:green', linestyle='--')
    ax2.set_title("Prototype-Level Task Separation")
    ax2.set_ylabel("Cosine Similarity")
    ax2.set_xlabel("Step Index")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # ---------------- New: Hidden-State Separation
    ax3 = fig.add_subplot(gs[0, 1])
    ax3.plot(hidden_inter, label="Inter-Task Hidden CosSim", color='tab:purple')
    ax3.plot(hidden_intra, label="Intra-Task Hidden CosSim", color='tab:gray')
    ax3.set_title("Hidden-State Task Separation (Direct Sample-Based)")
    ax3.set_ylabel("Cosine Similarity")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    if hidden_inter[-1] is not None and hidden_intra[-1] is not None:
        sep_ratio = hidden_inter[-1] / (hidden_intra[-1] + 1e-6)
        ax3.text(
            0.05, 0.05,
            f"Separation = {sep_ratio:.3f}",
            transform=ax3.transAxes,
            color="red" if sep_ratio > 0.8 else "green",
            fontsize=12
        )

    # ---------------- Latest Prototype Similarity Heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    if latest_matrix is not None:
        im = ax4.imshow(latest_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax4.set_title(f"Latest Prototype Similarity (Step {steps[-1]})")

        for i in range(len(latest_tasks)):
            for j in range(len(latest_tasks)):
                ax4.text(j, i, f"{latest_matrix[i, j]:.2f}",
                         ha="center", va="center", color="black", fontsize=9)

        ax4.set_xticks(np.arange(len(latest_tasks)))
        ax4.set_yticks(np.arange(len(latest_tasks)))
        ax4.set_xticklabels(latest_tasks)
        ax4.set_yticklabels(latest_tasks)
        plt.colorbar(im, ax=ax4)

    plt.tight_layout()
    plt.savefig(SAVE_IMG_PATH)
    print(f"✔ Dashboard saved → {SAVE_IMG_PATH}")

    # Final summary
    print("------ SUMMARY ------")
    if hidden_inter[-1] is not None:
        print(f"Final Inter-task Hidden CosSim: {hidden_inter[-1]:.4f}")
    if hidden_intra[-1] is not None:
        print(f"Final Intra-task Hidden CosSim: {hidden_intra[-1]:.4f}")


if __name__ == "__main__":
    plot_dashboard()
