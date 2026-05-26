import importlib.metadata

from .train_bpe import train_bpe, encode_bytes
from .utils import bytes_to_unicode
from .tokenizer import Tokenizer
from .linear import Linear
from .embedding import Embedding
from .rmsnorm import RMSNorm
from .swiglu import SiLU, SwiGLU
from .rope import RotaryPositionalEmbedding
from .softmax import softmax
from .attention import scaled_dot_product_attention, MHSA
from .transformer import TransformerBlock, TransformerLM
from .loss import cross_entropy
from .optimizer import AdamW
from .lr_schedule import lr_cosine_schedule
from .gradient_clipping import gradient_clipping_


try:
    __version__ = importlib.metadata.version("cs336_basics")
except importlib.metadata.PackageNotFoundError:
    pass
