import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import matplotlib.pyplot as plt

class InferenceRouter(nn.Module):
    def __init__(self, d_feature=128, num_kv=32, use_softmax=True):
        super().__init__()
        self.num_kv = num_kv
        self.use_softmax = use_softmax
        self.tau = 1.0
        
        self.cls_feat_extractor = nn.Sequential( 
            nn.Linear(d_feature, 4 * d_feature),
            nn.SiLU(),
            nn.Linear(4 * d_feature, d_feature),
        )

        if self.use_softmax: 
            self.cls_router_head_agnostic = nn.Sequential( 
                nn.Linear(d_feature, 4 * d_feature),
                nn.SiLU(),
                nn.Linear(4 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 2),
            )
        else:
            self.cls_router_head_agnostic = nn.Sequential( 
                nn.Linear(d_feature, 2 * d_feature),
                nn.SiLU(),
                nn.Linear(2 * d_feature, d_feature),
                nn.SiLU(),
                nn.Linear(d_feature, 1)
            )

    def forward(self, x, cu_seq_len=None):
        if cu_seq_len is not None:
            x_s, x_e = cu_seq_len[0], cu_seq_len[1]
            pooled_latent = x[x_s:x_e].mean(dim=0).unsqueeze(0) # [1, H, D]
        else:
            s_len = x.shape[1]
            if s_len > 200:
                target = torch.cat([x[:, :100, :], x[:, -100:, :]], dim=1).mean(dim=1)
            else:
                target = x.mean(dim=1)
            pooled_latent = target # [1, H, D]

        pooled_hidden_states = self.cls_feat_extractor(pooled_latent)
        binary_logits = self.cls_router_head_agnostic(pooled_hidden_states)

        if self.use_softmax:
            z_soft = F.softmax(binary_logits, dim=-1)
            decisions = z_soft[..., 1]
        else:
            decisions = torch.sigmoid(binary_logits / self.tau)
            
        return decisions
def run_timing_loop(model, inputs, device, test_iters=500):
    warmup_iters = 50
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = model(*inputs)
    
    if device == 'cuda':
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        starter.record()
        with torch.no_grad():
            for _ in range(test_iters):
                _ = model(*inputs)
        ender.record()
        torch.cuda.synchronize()
        total_time_ms = starter.elapsed_time(ender)
    else:
        start_time = time.time()
        with torch.no_grad():
            for _ in range(test_iters):
                _ = model(*inputs)
        total_time_ms = (time.time() - start_time) * 1000
    
    return total_time_ms / test_iters

def visualize_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seq_lengths = [512, 2048, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
    d_feat = 128
    num_heads = 32
    
    standard_times = []
    packed_times = []
    
    model = InferenceRouter(d_feature=d_feat, num_kv=num_heads).to(device).eval()

    print(f"Starting benchmark on {device}...")
    for s_len in seq_lengths:
        # 1. Standard Mode
        x = torch.randn(1, s_len, num_heads, d_feat, device=device)
        t_std = run_timing_loop(model, (x,), device)
        standard_times.append(t_std)
        
        # 2. Packed Mode
        # x_packed = x.squeeze(0)
        # cu_seq_len = torch.tensor([0, s_len], device=device, dtype=torch.int32)
        # t_packed = run_timing_loop(model, (x_packed, cu_seq_len), device)
        # packed_times.append(t_packed)
        
        print(f"SeqLen {s_len:5d}: Standard {t_std:.4f}ms")

    # --- 开始绘图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(seq_lengths, standard_times, marker='o', linestyle='-', linewidth=2, label='Standard (Concat First/Last 100)')
    # plt.plot(seq_lengths, packed_times, marker='s', linestyle='--', linewidth=2, label='Packed (Mean Full Seq)')
    
    plt.title(f'Router Inference Latency vs Sequence Length ({device.upper()})', fontsize=14)
    plt.xlabel('Sequence Length (Tokens)', fontsize=12)
    plt.ylabel('Average Latency (ms)', fontsize=12)
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.xscale('log', base=2)
    plt.legend()

    for i, txt in enumerate(standard_times):
        plt.annotate(f"{txt:.3f}", (seq_lengths[i], standard_times[i]), textcoords="offset points", xytext=(0,10), ha='center')

    plt.tight_layout()
    plt.show()
    plt.savefig(f"{device}_router_inference_latency.png")

if __name__ == "__main__":
    visualize_benchmark()