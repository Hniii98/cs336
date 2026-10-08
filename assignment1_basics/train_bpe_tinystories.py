import cs336_basics as cs336
import json
import os

def main():
	vocab, merges = cs336.train_bpe(input_path="data/TinyStoriesV2-GPT4-train.txt", 
								    vocab_size=10000, 
									special_tokens=["<|endoftext|>"],
									verbose=True)
	
	os.makedirs("data/TinyStories", exist_ok=True)
	
	bytes_encoder = cs336.bytes_to_unicode()

	longest_token = max(vocab.values(), key=len)

	print(f"longest token is : {longest_token}")
	
	vocab_for_json = {
		cs336.encode_bytes(token_bytes, bytes_encoder): token_id
		for token_id, token_bytes in vocab.items()
	}


	with open("data/TinyStories/vocab.json", "w", encoding="utf-8") as f_vocab, \
		 open("data/TinyStories/merges.txt", "w", encoding="utf-8") as f_merges:
		
		json.dump(vocab_for_json, f_vocab, ensure_ascii=False, indent=4)
		
		for pair in merges:
			f_merges.write(f"{cs336.encode_bytes(pair[0], bytes_encoder)} {cs336.encode_bytes(pair[1], bytes_encoder)}\n")



if __name__ == "__main__":
	main()