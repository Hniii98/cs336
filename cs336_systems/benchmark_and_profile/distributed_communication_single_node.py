import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import time

def get_device(rank, backend):
    if backend == "nccl":
        return torch.device(f"cuda:{rank}")
    elif backend == "gloo":
        return torch.device("cpu")
    else:
        raise ValueError(f"Unsupported backend: {backend}")
	
def setup(rank, world_size, backend):
	os.environ["MASTER_ADDR"] = "localhost"
	os.environ["MASTER_PORT"] = "29500"

	if backend == "nccl":
		torch.cuda.set_device(rank)
	
	dist.init_process_group(
		backend=backend, 
		rank=rank, 
		world_size=world_size,
		device_id=get_device(rank, "nccl") if backend == "nccl" else None,
	)

def cleanup():
	dist.destroy_process_group()



def distributed_fn(rank, world_size, data_size, backend):
	setup(rank, world_size, backend)
	size = (data_size * 1024 ** 2) // 4
	data = torch.randn((size,), dtype=torch.float32, device=get_device(rank, backend))

	torch.cuda.synchronize()
	dist.barrier()

	start = time.time()

	dist.all_reduce(data, async_op=False)

	torch.cuda.synchronize()
	dist.barrier()
	end = time.time()

	if rank == 0:
		print(
            f"{world_size} processes with {data_size} MB data in backend: {backend}"
            f" using {(end - start) * 1000:.3f} ms"
		)

	cleanup()
	

if __name__ == "__main__":
	data_size = [1, 10, 100, 1024]
	num_processes = [2, 4]
	backend = "nccl"
	for np in num_processes:
		for ds in data_size:
			mp.spawn(fn=distributed_fn, args=(np, ds, backend), nprocs=np, join=True)
			
