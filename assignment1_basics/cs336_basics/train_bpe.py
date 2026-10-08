from .pretokenization import parallel_pre_tokenization
from .merge import merge_pairs
from .utils import AutoKeyDictWrapper, bytes_to_unicode
import os

def encode_bytes(token_bytes: bytes,
				 encoder: dict[int: str]) -> str:
	return "".join(encoder[b] for b in token_bytes)


def train_bpe(
	input_path: str,
	vocab_size: int,
	special_tokens: list[str],
	verbose: bool = False,
) -> tuple[ dict[int, bytes],
		   	list[tuple[bytes, bytes]]]:
	
	num_processor = os.cpu_count() # including logical core
	
	freq_map = parallel_pre_tokenization(input_path=input_path, 
									  	 num_processes=num_processor, 
									  	 special_tokens=special_tokens)
	
	initial_vocab = AutoKeyDictWrapper()
	
	bytes_chars = [bytes([n]) for n in range(256)] 
	bytes_special_tokens = [bytes(token.encode("utf-8")) for token in special_tokens]

	initial_vocab.add_list(bytes_special_tokens + bytes_chars)

	merges = merge_pairs(freq_map=freq_map, 
					  	 initial_vocab=initial_vocab.dict, 
						 vocab_size=vocab_size,
						 verbose=verbose)

	return initial_vocab.dict, merges
	
	
	