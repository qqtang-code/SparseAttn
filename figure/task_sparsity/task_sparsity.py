import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 8,
    "axes.labelsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "lines.linewidth": 1.4,
})

top_p_sparse = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.1, 0.05]

tasks = [
    ("Single-QA", 41.66, [40.77, 40.82, 35.65, 32.34, 29.83, 27.85, 25.56, 23.07, 19.56, 18.89]),
    ("Multi-QA", 43.47, [44.26, 44.28, 40.73, 37.85, 34.11, 32.29, 28.09, 26.12, 24.44, 23.20]),
    ("Summarization", 24.41, [24.66, 24.31, 24.24, 24.30, 23.72, 23.68, 23.44, 23.37, 22.92, 22.82]),
    ("Few-Shot", 62.50, [63.59, 64.27, 65.53, 66.15, 65.59, 65.15, 65.42, 64.16, 58.62, 56.11]),
    ("Code", 15.10, [13.85, 12.42, 11.62, 10.84, 10.13, 8.93, 9.73, 9.36, 9.67, 8.98]),
]

fig, axes = plt.subplots(1, 5, figsize=(7.2, 1.8), sharex=True, sharey=True)

for ax, (task, full, sparse) in zip(axes, tasks):
    # 计算相对 Full 的变化
    delta_perf = [s - full for s in sparse]

    ax.plot(
        top_p_sparse,
        delta_perf,
        marker="o",
        markersize=3
    )

    # Full baseline → y = 0
    ax.axhline(
        y=0.0,
        linestyle="--",
        linewidth=1.6,
        color="black"
    )

    ax.set_title(task)
    ax.invert_xaxis()
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_xlabel("Top-P")

axes[0].set_ylabel("Δ Performance (vs. Full)")

plt.ylim(-25, 5)

plt.tight_layout(pad=0.35, w_pad=0.45)
plt.savefig("task_sparsity_delta.pdf")
plt.show()
