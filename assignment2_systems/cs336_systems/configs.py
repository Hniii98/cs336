

CONFIGS = {
	"small": {
		"d_model": 768,
		"d_ff": 3072,
		"num_layers": 12,
		"num_heads":12,
		"vocab_size": 10000,
		"batch_size": 4,
		"context_length": 512,
	},

	"medium": {
		"d_model": 1024,
		"d_ff": 4096,
		"num_layers": 24,
		"num_heads": 16,
		"vocab_size": 10000,
		"batch_size": 4,
		"context_length": 512,
	},

	"large": {
		"d_model": 1280,
		"d_ff": 5120,
		"num_layers": 36,
		"num_heads": 20,
		"vocab_size": 10000,
		"batch_size": 4,
		"context_length": 512,
	},

	"xl": {
		"d_model": 2560,
		"d_ff": 10240,
		"num_layers": 32,
		"num_heads": 32,
		"vocab_size": 10000,
		"batch_size": 4,
		"context_length": 512,
	},

	"10B": {
		"d_model": 4608,
		"d_ff": 12288,
		"num_layers": 50,
		"num_heads": 36,
		"vocab_size": 10000,
		"batch_size": 4,
		"context_length": 512,
	},
}
