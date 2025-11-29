import torch
import torch.distributed as dist

def main():
    # 初始化分布式环境
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()  # 当前进程的序号（0~num_gpus-1）
    world_size = dist.get_world_size()  # 总进程数（即使用的 GPU 数）
    
    print(f"成功启动！rank={rank}, world_size={world_size}, GPU={torch.cuda.current_device()}")
    
    # 销毁分布式环境
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
