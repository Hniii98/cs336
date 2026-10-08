import torch
from torch import nn

from .rmsnorm import RMSNorm
from .attention import MHSA
from .swiglu import SwiGLU
from .embedding import Embedding
from .linear import Linear
from .softmax import softmax

from jaxtyping import Float, Int
from torch import Tensor


class TransformerBlock(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		d_ff: int,
		theta: float,
		max_seq_len: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None
	):
		super().__init__()


		self.rmsnorm1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
		self.rmsnorm2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
		self.multi_head_attention = MHSA(
			d_model=d_model, 
			num_heads=num_heads, 
			max_seq_len=max_seq_len, 
			theta=theta, 
			device=device,
			dtype=dtype
		)
		self.swiglu = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
		

	def forward(
		self,
		in_features: Float[Tensor, "batch_size seq_len d_model"],
		token_positions: Int[Tensor, "seq_len"]
	) -> Float[Tensor, "batch_size seq_len d_model"]:
		
		# Attention block
		attn_output = self.multi_head_attention(self.rmsnorm1(in_features), token_positions)
		x = attn_output + in_features

		# FFN block
		ffn_output = self.swiglu(self.rmsnorm2(x))
		x = x + ffn_output

		return x

		
class TransformerLM(nn.Module):
	def __init__(
		self,
		vocab_size: int,
		num_layers: int,
		d_model: int,
		num_heads: int,
		d_ff: int,
		theta: float,
		max_seq_len: int,
		device: torch.device | None = None,
		dtype: torch.dtype | None = None
	):
		super().__init__()

		self.embed = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

		self.transformer_blocks = nn.ModuleList(
			[
				TransformerBlock(
					d_model=d_model, 
					num_heads = num_heads,
					d_ff=d_ff,
					theta=theta,
					max_seq_len=max_seq_len,
					device=device,
					dtype=dtype
				)
				for _ in range(num_layers)
			]
		)

		self.rmsnorm = RMSNorm(d_model=d_model, device=device, dtype=dtype)
		self.lin = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

	def forward(
		self,
		token_ids: Float[Tensor, "batch_size seq_len"],
		token_pos: Int[Tensor, "seq_len"]
	)-> Float[Tensor, "batch_size seq_len vocab_size"]:
		
		# Get embeding vecotr
		x = self.embed(token_ids)
		
		# Pass transformer block by #num_layers times with token positions
		for layer in self.transformer_blocks:
			x = layer(x, token_pos)
		
		# Norm the transformer blocks output
		x = self.rmsnorm(x)
		
		# Project d_modl to vocab_size
		x = self.lin(x)

		
		
		return x


			
