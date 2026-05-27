import torch
import os
import typing


def save_checkpoint(
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer,
	iteration: int,
	out = str | os.PathLike | typing.BinaryIO | typing.IO[bytes]
):
	model_dict = model.state_dict()
	optimizer_dict = optimizer.state_dict()

	composed_dict = {"model": model_dict, "optimizer": optimizer_dict, "iteration": iteration}

	torch.save(composed_dict, out)

def load_checkpoint(
	src,
	model: torch.nn.Module,
	optimizer: torch.optim.Optimizer
):
	composed_dict = torch.load(src)

	model.load_state_dict(composed_dict["model"])
	optimizer.load_state_dict(composed_dict["optimizer"])

	return composed_dict["iteration"]