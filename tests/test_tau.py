import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class AnalyzableAttentionRouter(nn.Module):
    def __init__(self, input_dim, d_feature=128):
        super().__init__()
        # 简化版 Router，只保留核心逻辑
        self.dim_mapping = nn.Sequential(
            nn.Linear(d_feature, d_feature),
            nn.SiLU(),
            nn.Linear(d_feature, 2)
        )
    
    def get_logits(self, x):
        logits = self.dim_mapping(x)
        logits = torch.tanh(logits) * 5.0 
        logits = torch.nan_to_num(logits, nan=0.0, posinf=5.0, neginf=-5.0)
        return logits

def run_training_analysis():
    input_dim = 128
    batch_size = 1000

    torch.manual_seed(42)
    model = AnalyzableAttentionRouter(input_dim)
    dummy_input = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        fixed_logits = model.get_logits(dummy_input) # [B, 2]

    # 定义 Tau 范围 (Log scale)
    tau_values = np.logspace(np.log10(0.1), np.log10(10.0), 100)
    
    results = {
        'tau': [],
        'mean_prob': [],
        'std_prob': [],
        'saturation_rate': [],
        'gradient_proxy': []
    }

    for tau in tau_values:
        t = torch.tensor(tau)
        
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(fixed_logits) + 1e-9) + 1e-9)

        y_soft = F.softmax((fixed_logits + gumbel_noise) / t, dim=-1)
        
        # 关注 Class 1 的概率
        probs = y_soft[:, 1]
        probs_np = probs.numpy()

        results['tau'].append(tau)

        results['mean_prob'].append(np.mean(probs_np))
        results['std_prob'].append(np.std(probs_np))
        
        # 2. 饱和率 (Saturation Rate): 有多少样本变成了 "硬决策" (>0.9 或 <0.1)
        # 这代表了 Router 是否真的在做“选择”
        is_saturated = (probs_np > 0.9) | (probs_np < 0.1)
        results['saturation_rate'].append(np.mean(is_saturated))
        
        # 3. 梯度流代理指标 (Gradient Proxy)
        # Softmax 的导数正比于 p * (1 - p)。
        # 如果 p 都是 0 或 1，梯度为 0 (Vanishing Gradient)。
        # 如果 p 都是 0.5，梯度最大。
        grad_proxy = probs_np * (1 - probs_np)
        results['gradient_proxy'].append(np.mean(grad_proxy))

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    ax = axes[0]
    ax.plot(results['tau'], results['mean_prob'], color='#d62728', linewidth=2, label='Mean Probability')
    ax.fill_between(results['tau'], 
                    np.array(results['mean_prob']) - np.array(results['std_prob']),
                    np.array(results['mean_prob']) + np.array(results['std_prob']),
                    color='#d62728', alpha=0.2, label='Variation (Gumbel Noise)')
    ax.set_xscale('log')
    ax.set_title('1. Training Output Stability', fontsize=14)
    ax.set_ylabel('Probability P(Route=1)')
    ax.set_xlabel('Tau')
    ax.legend()

    ax = axes[1]
    ax.plot(results['tau'], results['saturation_rate'], color='#2ca02c', linewidth=2)
    ax.set_xscale('log')
    ax.set_title('2. "Hardness" of Decisions (Saturation Rate)', fontsize=14)
    ax.set_ylabel('Proportion of Distinct Choices (>0.9 or <0.1)')
    ax.set_xlabel('Tau')
    ax.axvline(x=0.5, color='gray', linestyle='--', label='Typical Min Tau')
    ax.legend()
    
    # 图 3: 梯度流动性 (p * (1-p))
    # 太低意味着没梯度。
    ax = axes[2]
    ax.plot(results['tau'], results['gradient_proxy'], color='#1f77b4', linewidth=2)
    ax.set_xscale('log')
    ax.set_title('3. Estimated Gradient Flow (Variance Proxy)', fontsize=14)
    ax.set_ylabel('Mean p*(1-p) [Higher is more gradient]')
    ax.set_xlabel('Tau')
    
    plt.tight_layout()
    plt.savefig('8192_sequence_results.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    run_training_analysis()