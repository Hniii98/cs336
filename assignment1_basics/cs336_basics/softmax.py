import torch

def softmax(
	x: torch.Tensor,
	dim: int
):
	"""
		softmax(x_i) = exp(x_i) / sum(exp(x_i))
	"""
	row_max = torch.max(x, dim=dim, keepdim=True).values
	x_shifted = x - row_max
	exp_shifted = torch.exp(x_shifted)
	row_sum = torch.sum(exp_shifted, dim=dim, keepdim=True)

	normalized_x = exp_shifted / row_sum

	return normalized_x



