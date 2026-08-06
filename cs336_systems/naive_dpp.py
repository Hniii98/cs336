import torch
from torch import nn
import torch.distributed as dist


class DDP(nn.Module):
	def __init__(
		self,
		module: nn.Module,
	):	
		super().__init__()
		self.module = module

		self.local_rank = dist.get_rank()
		self.world_size = dist.get_world_size()

		for param in self.module.parameters():
			dist.broadcast(param.data, src=0)

	def forward(
		self,
		sharded: torch.Tensor,
	):
		out = self.module(sharded)
		return out
	
	def all_reduce_gradients(self):
		for param in self.module.parameters():
			if param.grad is not None:
				dist.all_reduce(param.grad)

	def average_gradients(self):
		with torch.no_grad():
			for param in self.module.parameters():
				if param.grad is not None:
					param.grad.div_(self.world_size)
	
	def finish_gradient_synchronization(self):
		self.all_reduce_gradients()
		self.average_gradients()
		
	


