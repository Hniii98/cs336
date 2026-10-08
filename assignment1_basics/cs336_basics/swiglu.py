import torch
import torch.nn as nn
from .linear import Linear

class SiLU(nn.Module):
	def	__init__(self):
		super().__init__()

	def forward(
		self,
		x: torch.Tensor
	):
		return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
	def __init__(
		self,
		d_model: int,
		d_ff: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None
	):
		super().__init__()
		
		d_ff = ((d_ff + 32) // 64) * 64 # round to multiple of 64

		self.lin1 = Linear(out_features=d_ff, in_features=d_model, device=device, dtype=dtype)
		self.lin2 = Linear(out_features=d_model, in_features=d_ff, device=device, dtype=dtype)
		self.lin3 = Linear(out_features=d_ff, in_features=d_model, device=device, dtype=dtype)
		

	def forward(
		self,
		x: torch.Tensor
	):
		silu = SiLU()
		x1 = self.lin1(x)
		silu_out = silu(x1)

		x3 = self.lin3(x)

		gelu_out = silu_out * x3
		swiglu_out = self.lin2(gelu_out)

		return swiglu_out