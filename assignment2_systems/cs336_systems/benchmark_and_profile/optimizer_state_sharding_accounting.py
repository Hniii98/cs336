import torch
from naive_ddp_benchmarking import make_data, make_model, get_device, setup, cleanup
from cs336_systems.configs import CONFIGS
from cs336_systems.optimizer_state_sharding import SharedOptimizer
from cs336_systems.fsdp import FSDP
from torch.multiprocessing.spawn import spawn
import torch.distributed as dist


def report_peak_mem_usage(phase: str, device):
    torch.cuda.synchronize(device)

    gib = 1024**3

    print(
        f"{phase}: "
        f"current={torch.cuda.memory_allocated(device) / gib:.3f} GiB, "
        f"peak={torch.cuda.max_memory_allocated(device) / gib:.3f} GiB, "
    )
    
    

def main(rank, world_size, use_shard=False, use_fsdp=False):
    backend = "nccl"
    model_size = "large"
    setup(rank, world_size, backend)
    cfg = CONFIGS[model_size]
    device = get_device(rank, backend)
    

    torch.cuda.reset_peak_memory_stats(device)
    model = make_model(cfg)
    model = model.to(device)

    if use_fsdp:
        model = FSDP(model, torch.float)
        
    if rank == 0:
        report_peak_mem_usage(f"After model initialization (use_shard={use_shard}, model_size={model_size})", device)

    x, labels = make_data(rank, world_size, backend, cfg)

    torch.cuda.reset_peak_memory_stats(device)
    out = model(x)
    loss_fn = torch.nn.CrossEntropyLoss()
    
    optimizer = None
    if use_shard:
        optimizer = SharedOptimizer(
                        model.parameters(), 
                        torch.optim.AdamW, 
                        lr=1e-4,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0.01,
                    )
    else:
        optimizer = torch.optim.AdamW(
                        model.parameters(),
                        lr=1e-4,
                        betas=(0.9, 0.999),
                        eps=1e-8,
                        weight_decay=0.01,
                    )

    loss = loss_fn(
        out.reshape(-1, cfg["vocab_size"]), 
        labels.reshape(-1)
    )

    loss.backward()

    if rank == 0:
        report_peak_mem_usage(f"Before optimizer step      (use_shard={use_shard}, model_size={model_size})", device)
    
    torch.cuda.reset_peak_memory_stats(device)

    if use_fsdp:
        model.finish_gradient_synchronization()
    optimizer.step()

    if rank == 0:
        report_peak_mem_usage(f"After optimizer step       (use_shard={use_shard}, model_size={model_size})", device)

    cleanup()

    


if __name__ == "__main__":
        spawn(fn=main, args=(2, False, True), nprocs=2, join=True)

    

    


    

    