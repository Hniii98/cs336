import torch
import os
import cs336_basics
from torch import nn
from cs336_systems.naive_dpp import DDP
from cs336_systems.configs import CONFIGS
import torch.distributed as dist
import torch.multiprocessing as mp
import time

cf = CONFIGS["xl"]

def setup(rank, world_size, backend):
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "29500"

	if backend == "cuda":
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


def main(rank, world_size, backend):
	# Intial environment
	setup(rank, world_size,backend)

	# Construct model 
	model = cs336_basics.BasicsTransformerLM(
		vocab_size=cf["vocab_size"],
		context_length=cf["context_length"],
		d_model=cf["d_model"],
		num_layers=cf["num_layers"],
		num_heads=cf["num_heads"],
		d_ff=cf["d_ff"],
	)

	model = model.to(get_device(rank, backend))

	# Wrap ddp and construct vars
	ddp_model = DDP(model)

	# We care timing rather than correctness, so use individual labels

	local_batch_size = cf["batch_size"] // world_size
	
	x = torch.randint(
		low=0,
		high=cf["vocab_size"],
		size=(local_batch_size, cf["context_length"]), # partial shard
		dtype=torch.long,
		device=get_device(rank, backend),
	)

	label = torch.randint(
		low=0,
		high=cf["vocab_size"],
		size=(local_batch_size, cf["context_length"]),
		dtype=torch.long,
		device=get_device(rank, backend),
	)

	loss_fn = torch.nn.CrossEntropyLoss()

	optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

	# Warm up
	out = ddp_model(x)
	loss = loss_fn(
		out.reshape(-1, cf["vocab_size"]), 
		label.reshape(-1)
	)
	loss.backward()
	ddp_model.finish_gradient_synchronization()
	optimizer.step()

	# Benchmark start
	torch.cuda.synchronize()
	dist.barrier()

	total_start = time.time()
	
	optimizer.zero_grad()
	out = ddp_model(x)
	loss = loss_fn(
		out.reshape(-1, cf["vocab_size"]), 
		label.reshape(-1)
	)
	loss.backward()

	comm_start = time.time()

	ddp_model.all_reduce_gradients()

	torch.cuda.synchronize()
	dist.barrier()
	comm_end = time.time()

	ddp_model.average_gradients()

	optimizer.step()

	torch.cuda.synchronize()
	dist.barrier()
	total_end = time.time()

	if rank == 0:
		total_time = total_end - total_start
		comm_time = comm_end - comm_start
		print(f"Total elapsed time: {total_time:.4f}s")
		print(f"Time spent communicating: {comm_time:.4f}s")
		print(f"Communication ratio: {comm_time / total_time:.2%}")
	
	cleanup()

if __name__ == "__main__":
	backend = "nccl"
	num_processes = 2
	
	mp.spawn(fn=main, args=(num_processes, backend), nprocs=num_processes, join=True)