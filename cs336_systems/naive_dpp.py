import torch
from torch import nn
import torch.distributed as dist

from torch._utils import (
    _flatten_dense_tensors,
    _unflatten_dense_tensors,
)


class NaiveDDP(nn.Module):
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


class FlatDDP(nn.Module):
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
		grads_view = [
			p.grad for p in self.module.parameters() if p.grad is not None
		]

		flatten = _flatten_dense_tensors(grads_view)

		dist.all_reduce(flatten)
		return flatten, grads_view

	def average_gradients(self):
		flattten, grads_view = self.all_reduce_gradients()

		grads_reduced = _unflatten_dense_tensors(flattten, grads_view)

		with torch.no_grad():
			for grad, grad_reduced in zip(grads_view, grads_reduced):
				grad.copy_(grad_reduced)
				grad.div_(self.world_size)

	
	def finish_gradient_synchronization(self):
		self.all_reduce_gradients()
		self.average_gradients()
		
	


