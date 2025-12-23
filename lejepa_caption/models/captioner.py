"""
LeJEPA Edge Captioner - Full Pipeline.

Combines all components for end-to-end image captioning with VL-JEPA paradigm.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

from .encoder import VisionEncoder
from .connector import CAbstractor
from .predictor import EmbeddingPredictor


class LeJEPACaptioner(nn.Module):
    """
    Complete LeJEPA Edge Captioning Pipeline.
    
    Architecture:
        Image -> VisionEncoder -> CAbstractor -> EmbeddingPredictor -> [Text Embeddings]
                                                                           |
                                                         (Optional) LLM Decoder -> Caption
    
    The model predicts text embeddings (VL-JEPA paradigm), not tokens directly.
    An LLM decoder is used only when text output is explicitly needed.
    """
    
    def __init__(
        self,
        enc_dim: int = 192,
        llm_dim: int = 640,
        predictor_layers: int = 4,
        predictor_heads: int = 8,
        max_caption_len: int = 50,
        encoder_pretrained: bool = False,
    ):
        """
        Args:
            enc_dim: Encoder embedding dimension (192 for ViT-Tiny)
            llm_dim: LLM embedding dimension (640 for Gemma3-270M)
            predictor_layers: Number of predictor transformer layers
            predictor_heads: Number of attention heads in predictor
            max_caption_len: Maximum caption length to predict
            encoder_pretrained: Use ImageNet pretrained encoder
        """
        super().__init__()
        
        # Vision Encoder (ViT-Tiny)
        self.encoder = VisionEncoder(pretrained=encoder_pretrained)
        
        # Connector (C-Abstractor)
        self.connector = CAbstractor(enc_dim=enc_dim, llm_dim=llm_dim)
        
        # Embedding Predictor
        self.predictor = EmbeddingPredictor(
            dim=llm_dim,
            num_layers=predictor_layers,
            num_heads=predictor_heads,
            max_len=max_caption_len,
        )
        
        # Store config
        self.config = {
            'enc_dim': enc_dim,
            'llm_dim': llm_dim,
            'predictor_layers': predictor_layers,
            'max_caption_len': max_caption_len,
        }
        
    def forward(
        self, 
        images: torch.Tensor,
        num_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Predict text embeddings from images.
        
        Args:
            images: Input images (B, 3, 224, 224)
            num_tokens: Number of tokens to predict (default: max_caption_len)
            
        Returns:
            Predicted text embeddings (B, num_tokens, llm_dim)
        """
        # Encode image patches
        patch_embeds = self.encoder(images)  # (B, 196, 192)
        
        # Compress to LLM-compatible tokens
        vision_tokens = self.connector(patch_embeds)  # (B, 49, 640)

        # Predict text embeddings
        text_embeds = self.predictor(vision_tokens, num_tokens)  # (B, N, 640)
        
        return text_embeds
    
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get vision embeddings (for retrieval, etc.).
        
        Args:
            images: Input images (B, 3, 224, 224)
            
        Returns:
            Vision tokens (B, 49, 640)
        """
        patch_embeds = self.encoder(images)
        vision_tokens = self.connector(patch_embeds)
        return vision_tokens

    @property
    def num_parameters(self) -> Dict[str, int]:
        """Parameter count breakdown."""
        return {
            'encoder': sum(p.numel() for p in self.encoder.parameters()),
            'connector': sum(p.numel() for p in self.connector.parameters()),
            'predictor': sum(p.numel() for p in self.predictor.parameters()),
            'total': sum(p.numel() for p in self.parameters()),
        }
    
    def load_lejepa_encoder(self, checkpoint_path: str):
        """Load LeJEPA pre-trained encoder weights."""
        self.encoder.load_lejepa_weights(checkpoint_path)


def get_captioner(
    config: str = "tiny",
    encoder_pretrained: bool = False,
) -> LeJEPACaptioner:
    """
    Factory function for creating captioner.
    
    Args:
        config: Model size ('tiny', 'small', 'base')
        encoder_pretrained: Use ImageNet pretrained encoder
    
    Returns:
        Configured LeJEPACaptioner instance
    """
    configs = {
        'tiny': {'predictor_layers': 2, 'predictor_heads': 4},
        'small': {'predictor_layers': 4, 'predictor_heads': 8},
        'base': {'predictor_layers': 6, 'predictor_heads': 8},
    }
    
    cfg = configs.get(config, configs['small'])
    
    return LeJEPACaptioner(
        predictor_layers=cfg['predictor_layers'],
        predictor_heads=cfg['predictor_heads'],
        encoder_pretrained=encoder_pretrained,
    )


if __name__ == "__main__":
    print("Testing LeJEPACaptioner...")
    
    # Test default config
    model = LeJEPACaptioner()
    
    params = model.num_parameters
    print(f"  Encoder: {params['encoder'] / 1e6:.2f}M")
    print(f"  Connector: {params['connector'] / 1e6:.2f}M")
    print(f"  Predictor: {params['predictor'] / 1e6:.2f}M")
    print(f"  Total: {params['total'] / 1e6:.2f}M")
    
    # Test forward pass
    dummy = torch.randn(2, 3, 224, 224)
    out = model(dummy)
    print(f"  Forward: {dummy.shape} -> {out.shape}")
    
    # Test vision encoding
    vis = model.encode_vision(dummy)
    print(f"  Vision: {dummy.shape} -> {vis.shape}")
    
    # Verify outputs
    assert out.shape == (2, 50, 640), f"Expected (2, 50, 640), got {out.shape}"
    assert vis.shape == (2, 49, 640), f"Expected (2, 49, 640), got {vis.shape}"
    
    # Test tiny config
    tiny = get_captioner('tiny')
    print(f"  Tiny config: {tiny.num_parameters['total'] / 1e6:.2f}M total")
    
    print("All tests passed!")
