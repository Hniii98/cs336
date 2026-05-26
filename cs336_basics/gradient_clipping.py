
from typing import Iterable
import torch

def gradient_clipping_(
	gradients: Iterable[torch.nn.Parameter],
	maximum_value: float,
	eps: float = 1e-6
):
	l2_norm = None

	for p in gradients:
		if p.grad is not None:
			if l2_norm is None: # lazy initialization
				l2_norm = torch.sum(p.grad ** 2)
			else:
				l2_norm += torch.sum(p.grad ** 2)
				
	
	l2_norm = torch.sqrt(l2_norm)
	
	if l2_norm.item() > maximum_value:
		scale = maximum_value / (l2_norm + eps)
		for p in gradients:
			if p.grad is not None:
				p.grad.mul_(scale)