import torch
import torch.nn as nn
from einops import einsum



class Linear(nn.Module):
	def __init__(
		self,
		in_features: int,
		out_features: int,
		device: torch.device | None =None,
		dtype: torch.dtype | None =None
	):
		super().__init__()

		std = (2/(in_features+out_features)) ** 0.5
		self.weight = nn.Parameter(
			nn.init.trunc_normal_(
				torch.empty((out_features, in_features), dtype=dtype, device=device),
				mean=0,
				std=std,
				a=-3*std,
				b=3*std
			)
		)
	
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		out = einsum(x, self.weight, "...  d_in, d_out d_in -> ... d_out")
		return out