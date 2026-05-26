import numpy.typing as npt
import torch
import numpy as np


def get_batch(
	x: npt.NDArray,
	batch_size: int,
	context_length: int,
	device: torch.device
) -> tuple[torch.tensor, torch.tensor]:
	max_valid_starting_index = len(x) - context_length - 1

	sampled_starting_indices = np.random.randint(0, max_valid_starting_index+1, size=batch_size)
	
	token_ids = []
	targets = []

	for start in sampled_starting_indices:
		end = start + context_length
		token_ids.append(x[start:end])
			# Shift right to get the corresponding targets, we make sure all end+1
			# is valid index in list.
		targets.append(x[start+1:end+1])
	# Transfer to contiguous memory.
	token_ids = np.array(token_ids)
	targets = np.array(targets)

	token_ids = torch.tensor(token_ids,  device=device)
	targets = torch.tensor(targets, device=device)

	return token_ids, targets