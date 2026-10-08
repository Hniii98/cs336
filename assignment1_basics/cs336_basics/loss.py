import torch 

from jaxtyping import Float, Int
from torch import Tensor

from einops import reduce


def cross_entropy(
	predicted_logits: Float[Tensor, "... seq_len vocab_size"],
	targets: Int[Tensor, "... seq_len"]	

):
	pick_logits = predicted_logits.gather(dim=-1, index= targets.unsqueeze(-1))

	max_of_logits = reduce(predicted_logits, '... vocab_size -> ... 1', 'max')
	sum_of_exp_logits = reduce(torch.exp(predicted_logits - max_of_logits), '... vocab_size -> ... 1', 'sum')
	log_sum_exp  = torch.log(sum_of_exp_logits)

	neg_log_likelyhood = log_sum_exp - (pick_logits - max_of_logits)

	total_loss = reduce(neg_log_likelyhood, "... seq_len 1 -> 1", 'sum')
	num_tokens = torch.tensor(predicted_logits.shape[:-1]).prod().item()

	return total_loss / num_tokens




