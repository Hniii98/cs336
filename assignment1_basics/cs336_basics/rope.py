import torch

import torch.nn as nn
from einops import repeat, rearrange, einsum

class RotaryPositionalEmbedding(nn.Module):
	def __init__(
		self,
		theta: float,
		d_k: int,
		max_seq_len: int,
		device: torch.device | None = None 
	):
		super().__init__()
		assert d_k % 2 == 0
		k = torch.linspace(start=1, end=d_k/2, steps=d_k//2)
		exp = (2 * k - 2)	/ d_k
		theta_k =  (theta ** exp) ** -1    #  theta_k = 1 / theat ^(2k-2)/d
		token_pos = torch.linspace(start=0, end=max_seq_len-1, steps=max_seq_len)

		seq = repeat(theta_k, "d -> seq_len d", seq_len=max_seq_len)
		token_pos = rearrange(token_pos, "seq_len -> seq_len 1")
		seq_with_pos = seq * token_pos

		self.register_buffer("cos", torch.cos(seq_with_pos), persistent=False)
		self.register_buffer("sin", torch.sin(seq_with_pos), persistent=False)


	def forward(
		self,
		x: torch.Tensor,
		token_positions: torch.Tensor
	) -> torch.Tensor:
		cos = self.cos[token_positions]
		sin = self.sin[token_positions]
		
		x_0 = x[..., ::2]
		x_1 = x[..., 1::2]
		
		

		x_0_rot = x_0 * cos - x_1 * sin
		x_1_rot = x_1 * cos + x_0 * sin

		x = torch.stack((x_0_rot, x_1_rot), dim=-1).flatten(start_dim=-2)
		return x


		