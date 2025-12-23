"""
Vision Encoder for LeJEPA Captioning.

Wraps a ViT-Tiny backbone compatible with LeJEPA pre-training (SIGReg).
Outputs patch-level features for downstream processing.
"""

import torch
import torch.nn as nn
import timm


class VisionEncoder(nn.Module):
    """
    ViT-Tiny encoder for edge-optimized image understanding.
    
    Architecture:
        - Backbone: vit_tiny_patch16_224 (from timm)
        - Output: (B, 196, 192) patch embeddings
        - Params: ~5.5M
        - Latency: ~32ms on CPU
    
    Compatible with:
        - LeJEPA SIGReg pre-training
        - Standard ImageNet pre-training
        - Random initialization
    """
    
    def __init__(
        self,
        model_name: str = "vit_tiny_patch16_224",
        pretrained: bool = False,
        drop_path_rate: float = 0.0,
    ):
        """
        Args:
            model_name: timm model identifier
            pretrained: Use ImageNet pretrained weights
            drop_path_rate: Stochastic depth rate (0.0 = disabled)
        """
        super().__init__()
        
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            drop_path_rate=drop_path_rate,
        )
        
        # Store config for later use
        self.embed_dim = self.backbone.embed_dim  # 192 for ViT-Tiny
        self.num_patches = (224 // 16) ** 2  # 196 for patch16, 224px
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract patch-level features.
        
        Args:
            x: Input images (B, 3, 224, 224)
            
        Returns:
            Patch embeddings (B, 196, 192)
        """
        # forward_features returns (B, N+1, D) with CLS token
        features = self.backbone.forward_features(x)
        
        # Remove CLS token if present (first token)
        if features.shape[1] == self.num_patches + 1:
            features = features[:, 1:, :]  # (B, 196, 192)
        
        return features
    
    def load_lejepa_weights(self, checkpoint_path: str):
        """
        Load weights from LeJEPA pre-training.
        
        Args:
            checkpoint_path: Path to LeJEPA checkpoint (.pt file)
        """
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        
        # Handle different checkpoint formats
        if 'model' in state_dict:
            state_dict = state_dict['model']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Filter to backbone weights only
        backbone_state = {
            k.replace('backbone.', ''): v 
            for k, v in state_dict.items() 
            if 'backbone' in k or 'patch_embed' in k or 'blocks' in k
        }
        
        self.backbone.load_state_dict(backbone_state, strict=False)
        print(f"Loaded LeJEPA weights from {checkpoint_path}")
    
    @property
    def output_shape(self) -> tuple:
        """Returns expected output shape: (num_patches, embed_dim)"""
        return (self.num_patches, self.embed_dim)


def get_encoder(
    pretrained: bool = False,
    lejepa_checkpoint: str = None,
) -> VisionEncoder:
    """
    Factory function for creating the vision encoder.
    
    Args:
        pretrained: Use ImageNet pretrained weights
        lejepa_checkpoint: Path to LeJEPA checkpoint (overrides pretrained)
    
    Returns:
        Configured VisionEncoder instance
    """
    encoder = VisionEncoder(pretrained=pretrained)
    
    if lejepa_checkpoint is not None:
        encoder.load_lejepa_weights(lejepa_checkpoint)
    
    return encoder


if __name__ == "__main__":
    # Quick test
    print("Testing VisionEncoder...")
    
    encoder = VisionEncoder(pretrained=False)
    print(f"  Embed dim: {encoder.embed_dim}")
    print(f"  Num patches: {encoder.num_patches}")
    print(f"  Output shape: {encoder.output_shape}")
    
    # Test forward pass
    dummy = torch.randn(2, 3, 224, 224)
    out = encoder(dummy)
    print(f"  Forward test: {dummy.shape} → {out.shape}")
    
    # Verify output
    assert out.shape == (2, 196, 192), f"Expected (2, 196, 192), got {out.shape}"
    print("All tests passed!")
