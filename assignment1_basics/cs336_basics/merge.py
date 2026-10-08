from collections import Counter, defaultdict
from tqdm import tqdm



def count_and_index_successive_pairs (
	freq_map: dict[tuple[bytes, ...], int] 
) -> tuple[ Counter[tuple[bytes, bytes]], 
		   	dict[tuple[bytes, bytes], set[tuple[bytes, ...]]]]: 
	"""
	Counting successive pairs of each key in freq_map and 
	indexing pairs to it's original pre-tokens. 
	"""
	num_of_pair = Counter()
	index_pairs_to_pretokens = defaultdict(set)
	for k, v in freq_map.items():
		for pair in zip(k[:-1], k[1:]):
			num_of_pair[pair] += v
			index_pairs_to_pretokens[pair].add(k)
	return num_of_pair, index_pairs_to_pretokens
	
	

def merge_pairs(
	freq_map: dict[tuple[bytes, ...], int],
	initial_vocab: dict[int, bytes],
	vocab_size: int,
	verbose: bool = False,
) -> list[tuple[bytes, bytes]]:
	
# 1. Counting initial sucessive pairs of pre-tokens and indexing the pairs to its pre-tokens.
# 2. Pick the maximum counting pair.
# 3. Update counts of those that overlap with the merged pairs.
# 4. Add new pair to vocabulary.
# 5. If length of vocabulary of is less than setting, go to step 2, otherwish finish.
	assert len(initial_vocab) < vocab_size , "There is no size for merging new vocabulary"

	num_of_pairs, index_pairs_to_pretokens = count_and_index_successive_pairs(freq_map)
	merge_procedure_cache = { key : list(key) for key in freq_map}
	merges = []
	for step in tqdm(range(vocab_size - len(initial_vocab)), desc="Compute merges", disable=not verbose):
		pair_to_merge = max(num_of_pairs, key=lambda k: (num_of_pairs[k], k))
		raw_pretokens_list = index_pairs_to_pretokens[pair_to_merge]

		# Each pair may index to many pre-tokens.
		for ptok in raw_pretokens_list:
			
			
			# Look up the merging procedure with the current pre-token.
			procedure = merge_procedure_cache[ptok]

			if len(procedure) == 1: continue
			
			i = 0
			# for i, (token1, token2) in enumerate(zip(procedure[:-1], procedure[1:])):
			while i < len(procedure) - 1:
				token1, token2 = procedure[i], procedure[i+1]
				freq_of_ptok = freq_map[ptok]
				if (token1, token2) == pair_to_merge:
					"""
					Check whether left overlap and right overlap exist. If do, decrease the
					previous pair num according to the freq_of_ptok. Meanwhile, increase the
					num of new pair produced by merging token1 and token2. Furthermore, preserve 
					the indexing between new pair and current pretoken.
					
					"""
					if i > 0:
						num_of_pairs[(procedure[i-1], token1)] -= freq_of_ptok

						num_of_pairs[(procedure[i-1], token1+token2)] += freq_of_ptok
						index_pairs_to_pretokens[(procedure[i-1], token1+token2)].add(ptok)
					# Right overlap exits.
					if (i + 2) < len(procedure):
						num_of_pairs[(token2, procedure[i+2])] -= freq_of_ptok

						num_of_pairs[(token1+token2, procedure[i+2])] += freq_of_ptok
						index_pairs_to_pretokens[(token1+token2, procedure[i+2])].add(ptok)

					
					num_of_pairs[(token1, token2)] -= freq_map[ptok]

					# Update procedure cache of a pretoken after merging pair.	
					procedure[i] = token1 + token2
					procedure.pop(i+1)

				i += 1
		merges.append((pair_to_merge[0], pair_to_merge[1]))
		initial_vocab[len(initial_vocab)] = pair_to_merge[0]+pair_to_merge[1]
		
					
				
	return merges