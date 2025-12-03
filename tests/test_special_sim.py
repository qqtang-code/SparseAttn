import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
import os

# --- 辅助函数：加载数据 ---
def load_all_logs(path):
    """加载单个文件或目录下所有文件中的日志数据"""
    all_data = []
    
    if os.path.isdir(path):
        file_list = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pt')]
        if not file_list:
            raise FileNotFoundError(f"No .pt files found in directory: {path}")
        print(f"✅ Found {len(file_list)} log files in directory: {path}")
        
        for file in file_list:
            try:
                # 假设每个文件内是一个 list of dicts
                data = torch.load(file)
                if isinstance(data, list):
                    all_data.extend(data)
                elif isinstance(data, dict):
                    # 如果单个文件只包含一个大字典
                    all_data.append(data)
                else:
                    print(f"⚠️ Warning: File {file} content format unexpected.")

            except RuntimeError as e:
                print(f"❌ Corrupted File: Skipping {file}. Error: {e}")
    
    elif os.path.isfile(path) and path.endswith('.pt'):
        try:
            data = torch.load(path)
            if isinstance(data, list):
                all_data.extend(data)
            elif isinstance(data, dict):
                 all_data.append(data)
            print(f"✅ Loaded single log file: {path}")
        except RuntimeError as e:
            raise RuntimeError(f"❌ Failed to load single file {path}. Error: {e}") from e
    
    else:
        raise FileNotFoundError(f"Log path not found or is invalid: {path}")

    if not all_data:
        raise ValueError("Loaded data is empty after checking all logs.")
    return all_data

def analyze_router_inputs(log_file_path, max_samples=2000, 
                          avg_heads=True, perplexity=30):
    
    print("\n==============================================")
    print("       AttentionRouter Input (x) Analysis       ")
    print("==============================================")
    
    try:
        data_chunks = load_all_logs(log_file_path)
    except Exception as e:
        print(f"❌ Error loading logs. Please check the path and file integrity.")
        print(f"Detail: {e}")
        return

    # 1. 提取数据
    all_x = []
    all_tasks = []
    
    ID_TO_TASK = {0: 'Single QA', 1: 'MultiHop QA', 2: 'Summarization', 3: 'Code', 4: 'Other'}
    
    # 统计信息初始化
    total_chunks_loaded = len(data_chunks)
    total_data_points = 0 

    total_count = 0
    for chunk in data_chunks:
        x_chunk = chunk['router_input'].numpy() # [B, H, D]
        t_chunk = chunk['task_ids'].numpy()     # [B]
        
        B, H, D = x_chunk.shape
        total_data_points += B
        
        if avg_heads:
            # 策略 A: 对 Heads 取平均 [B, D]
            X_data = x_chunk.mean(axis=1)
            tasks_data = [ID_TO_TASK.get(tid, 'Unknown') for tid in t_chunk]
        else:
            # 策略 B: Heads 展平 [B*H, D]
            X_data = x_chunk.reshape(B * H, D)
            tasks_data = np.repeat([ID_TO_TASK.get(tid, 'Unknown') for tid in t_chunk], H)
        
        if total_count + len(X_data) > max_samples:
            slice_idx = max_samples - total_count
            all_x.append(X_data[:slice_idx])
            all_tasks.extend(tasks_data[:slice_idx])
            total_count = max_samples
            break
        
        all_x.append(X_data)
        all_tasks.extend(tasks_data)
        total_count += len(X_data)

    # 拼接数据
    X = np.concatenate(all_x, axis=0) # [N, D]
    y = np.array(all_tasks)           # [N]
    
    num_samples = len(X)
    
    # --- 打印数据统计信息 ---
    print(f"\n--- Data Collection Summary ---")
    print(f"Total batches/chunks loaded: {total_chunks_loaded}")
    print(f"Total raw data points collected: {total_data_points}")
    print(f"Final samples used for analysis (N): {num_samples} (Max limit: {max_samples})")
    print(f"Feature Dimension (D): {X.shape[1]}")
    print(f"Data Mode: {'Averaged over Heads' if avg_heads else 'All Heads Flattened'}")
    
    # 任务分布
    df_temp = pd.DataFrame({'Task': y})
    print("\nTask Sample Distribution:")
    print(df_temp['Task'].value_counts())
    
    if num_samples < 5:
        print(f"\n❌ CRITICAL ERROR: Insufficient samples ({num_samples}). Visualization requires more data.")
        return
    
    print("\n--- Running t-SNE Analysis ---")

    # =========================================
    # 分析 1: t-SNE 可视化 (看聚类情况)
    # =========================================
    
    # 调整 perplexity 以避免报错：确保 perplexity < N/3
    tsne_perplexity = min(perplexity, num_samples // 3 - 1) 
    if tsne_perplexity < 1:
        tsne_perplexity = 5 
        if num_samples < 15:
             print(f"⚠️ Warning: Sample size is too small ({num_samples}). Setting perplexity to {tsne_perplexity}. Results may be unreliable.")

    print(f"t-SNE running with final perplexity: {tsne_perplexity}")
    
    try:
        tsne = TSNE(n_components=2, perplexity=tsne_perplexity, n_iter=1000, random_state=42)
        X_embedded = tsne.fit_transform(X)
    except ValueError as e:
        print(f"❌ t-SNE failed. Detail: {e}")
        print(f"HINT: Check if you have enough unique data points ({num_samples}) for perplexity={tsne_perplexity}.")
        return

    df_tsne = pd.DataFrame({
        'x_dim1': X_embedded[:, 0],
        'x_dim2': X_embedded[:, 1],
        'Task': y
    })

    plt.figure(figsize=(16, 7))
    
    plt.subplot(1, 2, 1)
    sns.scatterplot(data=df_tsne, x='x_dim1', y='x_dim2', hue='Task', palette='bright', alpha=0.7)
    head_mode_label = 'Averaged over Heads' if avg_heads else 'All Heads Flattened'
    plt.title(f't-SNE Visualization of Router Inputs (x)\n({head_mode_label})')
    plt.grid(True, alpha=0.3)
    
    # =========================================
    # 分析 2: 任务中心点余弦相似度 (看混淆程度)
    # =========================================
    print("\n--- Running Cosine Similarity Analysis ---")
    
    unique_tasks = sorted(list(set(y)))
    centroids = []
    valid_tasks = []
    
    for task in unique_tasks:
        indices = np.where(y == task)
        task_vectors = X[indices]
        if len(task_vectors) > 0:
            centroid = np.mean(task_vectors, axis=0)
            centroids.append(centroid)
            valid_tasks.append(task)
    
    if len(valid_tasks) > 1:
        centroids = np.array(centroids)
        sim_matrix = cosine_similarity(centroids)
        
        plt.subplot(1, 2, 2)
        sns.heatmap(sim_matrix, annot=True, fmt=".2f", cmap="coolwarm", 
                    xticklabels=valid_tasks, yticklabels=valid_tasks, vmin=0, vmax=1)
        plt.title('Cosine Similarity between Task Centroids')
        
        print("\n=== Diagnosis: Similarity Matrix ===")
        mask = np.eye(sim_matrix.shape[0], dtype=bool)
        off_diag_sim = sim_matrix[~mask]
        max_sim = off_diag_sim.max() if len(off_diag_sim) > 0 else 0
        
        print(f"Max inter-task similarity (Off-diagonal): {max_sim:.4f}")
        
        if max_sim > 0.95:
            print("❌ CRITICAL: Inputs are extremely similar. The 'first_token' is not differentiating tasks effectively.")
        elif max_sim > 0.8:
            print("⚠️ WARNING: Tasks are close. Requires strong MLP/high logit separation.")
        else:
            print("✅ PASS: Inputs are well separated. The foundation for routing is good.")
            
    else:
        print("\n=== Diagnosis: Similarity Matrix ===")
        print("Insufficient task diversity (less than 2 unique tasks) to calculate similarity matrix.")

    plt.tight_layout()
    plt.show()
    plt.savefig('router_analysis.png')
    print("\n--- Done ---")
    print("Results saved to router_analysis.png")

if __name__ == "__main__":
    analyze_router_inputs(
        log_file_path="/data1/lcm_lab/qqt/SparseAttn/sparseattn/router_logs_task_emb_new",
        max_samples=2000,
        avg_heads=True,
        perplexity=30
    )