import torch

from torch import nn 
from jaxtyping import Bool, Float, Int
from torch import Tensor
from einops import einsum, rearrange

from .softmax import softmax
from .linear import Linear
from .rope import RotaryPositionalEmbedding


def scaled_dot_product_attention(
	Q: Float[Tensor, " ... queries d_k"],
    K: Float[Tensor, " ... keys d_k"],
    V: Float[Tensor, " ... keys d_v"],
    mask: Bool[Tensor, " ... queries keys"] | None = None
) -> Float[Tensor, "... quries d_v"]:
	d_k = Q.shape[-1]
	pre_softmax_logits = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
	pre_softmax_logits = pre_softmax_logits / (d_k ** 0.5)
	
	if mask is not None:
		mask = torch.where(mask, 0.0, float('-inf'))
		pre_softmax_logits = pre_softmax_logits + mask

	logits = softmax(pre_softmax_logits, -1)
	
	scores = einsum(logits, V, "... queries keys, ... keys d_v -> ... queries d_v")
	return scores
	
class MHSA(nn.Module):
	def __init__(
		self,
		d_model: int,
		num_heads: int,
		max_seq_len: int | None = None, # Max sequence length for RoPE
		theta: float | None = None, # theta for RoPE
		device: torch.device | None = None,
		dtype:  torch.dtype | None = None
	):
		super().__init__()
		self.q_proj = Linear(in_features=d_model, out_features=d_model, device=device)
		self.k_proj = Linear(in_features=d_model, out_features=d_model, device=device)
		self.v_proj = Linear(in_features=d_model, out_features=d_model, device=device)
		self.out_proj = Linear(in_features=d_model, out_features=d_model, device=device)

		self.num_heads = num_heads
		self.d_model = d_model

		if max_seq_len is not None:
			d_k = d_model // num_heads
			self.rope = RotaryPositionalEmbedding(theta=theta, d_k=d_k, max_seq_len=max_seq_len, device=device)



	def forward(
		self,
		in_features: Float[Tensor, " ... sequence_length d_model"],
		token_positions: Int[Tensor, "... sequence_length"] | None = None	
	)-> Float[Tensor, "... sequence_length d_model"]:
		seq_len = in_features.shape[-2]

		# Project x to Q, K, V
		Q = self.q_proj(in_features)
		K = self.k_proj(in_features)
		V = self.v_proj(in_features)

		# Rearrange to batch-like muti-head matrix
		Q_heads = rearrange(Q, "... seq_len (nheads dk) -> ... nheads seq_len dk", nheads=self.num_heads)
		K_heads = rearrange(K, "... seq_len (nheads dk) -> ... nheads seq_len dk", nheads=self.num_heads)
		V_heads = rearrange(V, "... seq_len (nheads dk) -> ... nheads seq_len dk", nheads=self.num_heads)

		# Add rope to queries and keys
		if token_positions is not None:
			Q_heads = self.rope(Q_heads, token_positions)
			K_heads = self.rope(K_heads, token_positions)

		# Create mask for masking future tokens, True for keeping and False for masking.
		mask = torch.tril(torch.ones(seq_len, seq_len), diagonal=0).bool()

		scores_head = scaled_dot_product_attention(Q=Q_heads, K=K_heads, V=V_heads, mask=mask)
		scores = rearrange(scores_head, "... nheads seq_len dv -> ... seq_len (nheads dv)", nheads=self.num_heads)

		out = self.out_proj(scores)
		return out
		



		
		

