import torch
import torch.nn as nn


class Embedding(nn.Module):
	def __init__(
		self,
		num_embeddings: int, # Size of the vocabulary
		embedding_dim: int, # Dimension of the embedding vectors, i.e., dmodel
		device: torch.device | None = None,
		dtype:  torch.dtype | None = None
	):
		super().__init__()
		self.embed_lut = nn.Parameter(
			nn.init.trunc_normal_(
				torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype),
				mean=0,
				std=1,
				a=-3,
				b=3
			)
		)

	def forward(
		self,
		token_ids: torch.Tensor
	) -> torch.Tensor:
		return self.embed_lut[token_ids]