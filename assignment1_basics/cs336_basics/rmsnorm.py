import torch
import 	torch.nn as nn	
from einops import reduce


class RMSNorm(nn.Module):
	def __init__(
		self,
		d_model: int,
		eps: float = 1e-5,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None
	):
		super().__init__()
		
		self.g = nn.Parameter(
			torch.ones((d_model,), device=device, dtype=dtype)
		)
		self.eps = eps
		self.d_model = d_model

	def forward(
		self,
		x: torch.Tensor
	):
		in_type = x.dtype
		x = x.to(torch.float32)

		rms = torch.sqrt(reduce(x**2, '... d_model -> ... 1', 'sum') / self.d_model + self.eps)
		result = x * self.g / rms

		return result.to(in_type)