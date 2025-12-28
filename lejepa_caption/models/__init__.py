# LeJEPA Captioning Models
from .encoder import VisionEncoder, get_encoder
from .connector import CAbstractor
from .predictor import EmbeddingPredictor
from .captioner import LeJEPACaptioner, get_captioner
from .sigreg import SIGRegLoss

# Decoder - RECOMMENDED (uses lm_head for correct hidden_states → vocab projection)
from .decoder import LMHeadDecoder, HybridLMHeadDecoder, get_lmhead_decoder

# Decoder - DEPRECATED (uses wrong embedding space, kept for backward compatibility)
from .decoder import StatisticsNormalizer, FastNNDecoder, HybridDecoder, get_hybrid_decoder
