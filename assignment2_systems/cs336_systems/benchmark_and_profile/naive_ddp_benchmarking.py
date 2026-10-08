import torch
import os
import cs336_basics
import torch.distributed as dist
from torch.multiprocessing.spawn import spawn
import time
import argparse
from cs336_systems.naive_ddp import NaiveDDP, FlatDDP, OverlapDDP
from cs336_systems.configs import CONFIGS

DDP_IMPLEMENTATIONS = {
    "naive": NaiveDDP,
    "flat": FlatDDP,
    "overlap": OverlapDDP,
}


def setup(rank, world_size, backend):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    if backend == "nccl":
        torch.cuda.set_device(rank)
    
    dist.init_process_group(
        backend=backend, 
        rank=rank, 
        world_size=world_size,
        device_id=torch.device("cuda", rank) if backend == "nccl" else None,
    )

def cleanup():
    dist.destroy_process_group()

def get_device(rank, backend):
    if backend == "nccl":
        return torch.device(f"cuda:{rank}")
    elif backend == "gloo":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
    

def make_data(rank, world_size, backend, config):
    local_batch_size = config["batch_size"] // world_size
    
    x = torch.randint(
        low=0,
        high=config["vocab_size"],
        size=(local_batch_size, config["context_length"]), # partial shard
        dtype=torch.long,
        device=get_device(rank, backend),
    )

    label = torch.randint(
        low=0,
        high=config["vocab_size"],
        size=(local_batch_size, config["context_length"]),
        dtype=torch.long,
        device=get_device(rank, backend),
    )

    return x, label

def make_model(config):
    model = cs336_basics.BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    )
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark custom DDP implementations")

    parser.add_argument(
        "--ddp",
        type=str,
        choices=["naive", "flat", "overlap"],
        default="naive",
        help="DDP implementation to benchmark",
    )

    parser.add_argument(
        "--backend",
        type=str,
        choices=["nccl", "gloo"],
        default="nccl",
    )

    parser.add_argument(
        "--world_size",
        type=int,
        default=2,
        help="Number of worker processes",
    )

    parser.add_argument(
        "--model_size",
        type=str,
        choices=["large", "xl"],
        default="xl",
    )

    return parser.parse_args()


def wrap_ddp_model(model, ddp_type):
    ddp_class = DDP_IMPLEMENTATIONS[ddp_type]
    return ddp_class(model)
    


def main(rank, world_size, backend, ddp_type, config):
    # Intial environment
    setup(rank, world_size,backend)

    # Construct model 
    model = make_model(config)
    model = model.to(get_device(rank, backend))

    # Wrap ddp and construct vars
    ddp_model = wrap_ddp_model(model,ddp_type)

    # We care timing rather than correctness, so use individual labels
    x, label = make_data(rank, world_size, backend, config)

    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    # Warm up
    out = ddp_model(x)
    loss = loss_fn(
        out.reshape(-1, config["vocab_size"]), 
        label.reshape(-1)
    )
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step()

    
    # Benchmark start
    if ddp_type != "overlap":
        # 第一次仅测试通信时间
        torch.cuda.synchronize()
        dist.barrier()

        optimizer.zero_grad()
        out = ddp_model(x)
        loss = loss_fn(
            out.reshape(-1, config["vocab_size"]), 
            label.reshape(-1)
        )

        loss.backward()
        # 测试通信时间
        torch.cuda.synchronize()

        comm_start = time.time()
        dist.barrier()
        state = ddp_model.all_reduce_gradients()
        torch.cuda.synchronize()
        dist.barrier()
        comm_end = time.time()
        comm_time = comm_end - comm_start

        ddp_model.average_gradients(state)
        optimizer.step()

    # 第二测试完整的的一步的工作时间，排除多余同步原语的用时。
    # OverlapDDP没有显式地通信，可以直接注释掉通信部分代码
    torch.cuda.synchronize()
    dist.barrier()
    total_start = time.time()

    optimizer.zero_grad()
    out = ddp_model(x)
    loss = loss_fn(
        out.reshape(-1, config["vocab_size"]), 
        label.reshape(-1)
    )
    loss.backward()
    ddp_model.finish_gradient_synchronization() # 封装的通信和更新
    optimizer.step()

    torch.cuda.synchronize()
    total_end = time.time()
    dist.barrier()

    if rank == 0:
        total_time = total_end - total_start
        print(f"Total elapsed time: {total_time:.4f}s")
        if ddp_type != "overlap":
            print(f"Time spent communicating: {comm_time:.4f}s")
            print(f"Communication ratio: {comm_time / total_time:.2%}")

    cleanup()

if __name__ == "__main__":
    args = parse_args()
    spawn(fn=main, args=(args.world_size, args.backend, args.ddp, CONFIGS[args.model_size]), nprocs=args.world_size, join=True)
