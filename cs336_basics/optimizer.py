import torch
from collections.abc import Callable
from typing import Optional
import math

class AdamW(torch.optim.Optimizer):
	def __init__(
		self, 
		params, 
		weight_decay,
		lr=1e-3,
		betas=(0.9, 0.95),
		eps=1e-8
	):
		if lr < 0:
			raise ValueError(f"Invalid learning rate: {lr}")
		
		defaults = {
			"lr": lr, "weight_decay": weight_decay, "beta1": betas[0], 
			"beta2": betas[1], "eps": eps
		}
		super().__init__(params, defaults)

	def step(self, closure:Optional[Callable] = None):
		loss = None if closure is None else closure()

		for group in self.param_groups:
			lr = group["lr"]
			weight_decay = group["weight_decay"]
			beta1 = group["beta1"]
			beta2 = group["beta2"]
			eps = group["eps"]

			for p in group["params"]:
				if p.grad is None:
					continue

				state = self.state[p]
				t = state.get("t", 0)
				m = state.get("m", torch.zeros_like(p))
				v = state.get("v", torch.zeros_like(p))
				grad = p.grad.data

				
				p.data -= lr * weight_decay * p.data # Apply weight decay
				m = beta1 * m + (1 - beta1) * grad # Update the first moment estimate 
				v = beta2 * v + (1 - beta2) * (grad ** 2) # Update the seconde moment estimate
				
				alpha_t = lr * math.sqrt(1 - beta2 ** (t + 1)) / (1 - beta1 ** (t + 1))
				p.data -= alpha_t * m / (torch.sqrt(v) + eps) # Update the weight in-place

				state["t"] = t + 1 # Increment iteration number.
				state["m"] = m
				state["v"] = v
		return loss

				





