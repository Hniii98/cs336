import json
import regex as re	
from typing import Iterable, Iterator
from .utils import bytes_to_unicode, PAT




def _unicode_to_bytes (
		json_vocab: dict[str:int],
		json_merges: list[str],	
	) -> tuple[dict[int:bytes], list[tuple[bytes, bytes]]]:
		vocab = {}
		merges = []
		
		# bytes -> str
		remap_encoder = bytes_to_unicode()
		# str -> bytes
		remap_decoder = {v : k for k,v in remap_encoder.items()}

		for k, v in json_vocab.items():
			token = bytes([remap_decoder[c] for c in k])
			vocab[v] = token


		for s in json_merges:
			str1, str2 = s.split()
			token1 = bytes([remap_decoder[c] for c in str1])
			token2 = bytes([remap_decoder[c] for c in str2])

			merges.append((token1, token2))
		
		return vocab, merges




class Tokenizer:
	def __init__(
		self, 
		vocab: dict[int, bytes],
		merges: list[tuple[bytes, bytes]], 
		special_tokens: list[str] | None = None,
	):
		self.vocab = vocab
		self.merges = merges
		self.special_tokens = ["<|endoftext|>"] if special_tokens is None else special_tokens

		# Add user-defined special tokens
		if special_tokens is not None:
			for tok in special_tokens:
				if tok.encode("utf-8") not in self.vocab.values():
					vocab[len(vocab)] = tok.decode("utf-8")

		self.bytes_to_ids = {v : k for k, v in self.vocab.items()}
		self.merges_hash = set(self.merges)
	
		
	@classmethod
	def from_files(
		cls,
		vocab_filepath: str,
		merges_filepath: str,
		special_tokens: list[str] | None = None,
	):
		json_vocab = {}
		json_merges = []
		try:
			with open(vocab_filepath, "r") as f1:
				json_vocab = json.load(f1)
		except:
			raise

		try:
			with open(merges_filepath, "r") as f2:
				json_merges = [line.rstrip() for line in f2]
		except:
			raise
		
		vocab, merges = _unicode_to_bytes(json_vocab=json_vocab, json_merges=json_merges)

		return cls(vocab, merges, special_tokens)
	
	def _merges(
		self,
		tokens: list[int],
		pair: tuple[int, int],
		idx: int,
	) -> list[int]:
		"""
		Replace pair in tokens with new idx.
		"""
		newids = []

		i = 0
		while i < len(tokens):
			if i < len(tokens) -1 and pair[0] == tokens[i] and pair[1] == tokens[i+1]:
				newids.append(idx)
				i += 2
			else:
				newids.append(tokens[i])
				i += 1
		return newids
					
	def _get_pair_lut(
		self,
		tokens: list[int],
	) -> dict[tuple[int, int], int]:
		"""
		Get ids of tuple[int, int] if bytes(tuple(int, int)) exists in self.vocab.
		"""
		lut = {}
		for pair in zip(tokens, tokens[1:]):
			merged_bytes = self.vocab[pair[0]] + self.vocab[pair[1]]
			if merged_bytes in self.bytes_to_ids:
				lut[pair] = self.bytes_to_ids[merged_bytes]
		return lut
	
	def _get_raw_bytes_ids(
		self,
		raw_bytes: list[int],
	) -> list[int]:
		"""
		Fetch ids of list of raw bytes
		"""
		ids_of_raw_bytes = []
		for b in raw_bytes:
			ids_of_raw_bytes.append(self.bytes_to_ids[bytes([b])])
		return ids_of_raw_bytes
			
	def _merge_within_pretoken(
		self,
		pretoken: str
	) -> list[int]:
		"""
		Encode a given pretoken to ids by merging pair within list of raw bytes.
		"""
		raw_bytes = list(pretoken.encode("utf-8"))
		tokens = self._get_raw_bytes_ids(raw_bytes)
		while len(tokens) >= 2:
			lut = self._get_pair_lut(tokens)
			if not lut:
				break
			pair = min(lut, key=lambda p : lut[p])
			idx = lut[pair]
			tokens = self._merges(tokens, pair, idx)
		return tokens
		
	def _encode_part_of_text(
		self,
		part: str,
	) -> list[int]:
		"""
		Split a part str without special tokens into pretokens and encode 
		each pretoken to ids in sequence.
		"""
		ids = []
		pretokens = re.finditer(PAT, part)
		for match in pretokens:
			pretoken = match.group()
			ids_in_pretoken = self._merge_within_pretoken(pretoken)
			ids.extend(ids_in_pretoken)
		return ids
		

	def encode(
		self,
		text: str
	) -> list[int]:
		# 1. Split text by special token into parts.
		# 2. If part is special token ,return its ids directly, else pretokenize part.
		# 3. For each pretoken, find all pair exists in vocab and keep its ids in dict within pretoken.
		# 3. Select the minimum ids as top pair to merge.
		# 4. Repeat #2 and #3, When there is no more pair to merge, end it.
		special_tokens_sorted = sorted(self.special_tokens, key=len, reverse=True)
		pattern = "(" + "|".join(re.escape(t) for t in special_tokens_sorted) + ")"

		parts = re.split(pattern, text)
		ids = []
		for part in parts:
			if part in self.special_tokens:
				ids.append(self.bytes_to_ids[part.encode("utf-8")])
			elif not part : # empty string caused by continuous special token
				continue
			else:
				ids.extend(self._encode_part_of_text(part))
		return ids
	
	def encode_iterable(
			self,
			iterable: Iterable[str]
	) -> Iterator[int]:
		for text in iterable:
			yield from self.encode(text)

	def decode(
			self, 
			ids: list[int]
	) -> str:
		tokens = b"".join(self.vocab[id] for id in ids)
		text = tokens.decode("utf-8", errors="replace")

		return text	

	