"""
Embedding Predictor for LeJEPA Captioning (VL-JEPA Style).

Predicts text embeddings from vision embeddings using cross-attention.
This is the core novelty: predict embeddings, not tokens.
"""

import torch
import torch.nn as nn
import math


class EmbeddingPredictor(nn.Module):
    """
    Predicts text embeddings from vision embeddings (VL-JEPA paradigm).
    
    Architecture:
        - Learnable query tokens (like DETR object queries)
        - Transformer decoder with cross-attention to vision
        - Output: predicted text embeddings in LLM space
    
    Key Insight:
        Instead of autoregressive token generation, we predict WHERE in 
        embedding space the caption should be. Decoding to text is optional.
    """
    
    def __init__(
        self,
        dim: int = 1024,
        num_layers: int = 4,
        num_heads: int = 8,
        max_len: int = 50,
        dropout: float = 0.1,
    ):
        """
        Args:
            dim: Embedding dimension (must match LLM dim, e.g., 1024 for Gemma3)
            num_layers: Number of transformer decoder layers
            num_heads: Number of attention heads
            max_len: Maximum caption length to predict
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim = dim
        self.max_len = max_len
        
        # Learnable query tokens - one per output position
        # These learn to "ask" for different parts of the caption
        self.query_tokens = nn.Parameter(
            torch.randn(1, max_len, dim) * 0.02
        )
        
        # Positional encoding for queries
        self.pos_embed = nn.Parameter(
            torch.randn(1, max_len, dim) * 0.02
        )
        
        # Transformer decoder layers
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Output projection (optional, can learn slight refinement)
        self.out_proj = nn.Linear(dim, dim)
        
        # Layer norm
        self.norm = nn.LayerNorm(dim)
        
        self._init_weights()
        
    def _init_weights(self):
        """Initialize with small weights for stable training."""
        # Identity-ish initialization for output projection
        nn.init.eye_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        
    def forward(
        self, 
        vision_embeds: torch.Tensor,
        num_tokens: int = None,
    ) -> torch.Tensor:
        """
        Predict text embeddings from vision.
        
        Args:
            vision_embeds: C-Abstractor output (B, 49, 1024)
            num_tokens: Number of tokens to predict (default: max_len)
            
        Returns:
            Predicted text embeddings (B, num_tokens, 1024)
        """
        B = vision_embeds.shape[0]
        num_tokens = num_tokens or self.max_len
        
        # Prepare queries with positional encoding
        queries = self.query_tokens[:, :num_tokens, :] + self.pos_embed[:, :num_tokens, :]
        queries = queries.expand(B, -1, -1)  # (B, num_tokens, dim)
        
        # Cross-attend to vision embeddings
        # Queries attend to vision, learning what visual content maps to what text
        predicted = self.decoder(
            tgt=queries,
            memory=vision_embeds,
        )
        
        # Project and normalize
        predicted = self.out_proj(predicted)
        predicted = self.norm(predicted)
        
        return predicted
    
    @property
    def num_parameters(self) -> int:
        """Total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Quick test
    print("Testing EmbeddingPredictor...")
    
    predictor = EmbeddingPredictor(dim=1024, num_layers=4, max_len=50)
    print(f"  Dim: {predictor.dim}")
    print(f"  Max length: {predictor.max_len}")
    print(f"  Parameters: {predictor.num_parameters / 1e6:.2f}M")
    
    # Test forward pass (simulating connector output)
    dummy_vision = torch.randn(2, 49, 1024)
    out = predictor(dummy_vision)
    print(f"  Forward test: {dummy_vision.shape} -> {out.shape}")
    
    # Test with custom length
    out_short = predictor(dummy_vision, num_tokens=20)
    print(f"  Short caption: {dummy_vision.shape} -> {out_short.shape}")
    
    # Verify outputs
    assert out.shape == (2, 50, 1024), f"Expected (2, 50, 1024), got {out.shape}"
    assert out_short.shape == (2, 20, 1024), f"Expected (2, 20, 1024), got {out_short.shape}"
    print("All tests passed!")
