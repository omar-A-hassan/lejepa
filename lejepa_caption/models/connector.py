"""
C-Abstractor Connector for LeJEPA Captioning.

Compresses vision tokens via strided convolution for efficient LLM processing.
Reduces 196 tokens -> 49 tokens (4x compression).
"""

import torch
import torch.nn as nn


class CAbstractor(nn.Module):
    """
    Convolutional Abstractor (C-Abstractor).
    
    Converts encoder patch embeddings to LLM-compatible token embeddings
    with spatial downsampling for reduced context length.
    
    Architecture:
        - Input: (B, 196, 192) - 14x14 grid of 192-dim patches
        - Conv2d with stride 2: 14x14 -> 7x7
        - Output: (B, 49, llm_dim) - 7x7 grid of LLM-dim tokens
    
    Benefits:
        - 4x token compression (196 -> 49)
        - Faster LLM decoding
        - Preserves spatial structure (unlike mean pooling)
    """
    
    def __init__(
        self,
        enc_dim: int = 192,
        llm_dim: int = 1024,
        kernel_size: int = 3,
        stride: int = 2,
    ):
        """
        Args:
            enc_dim: Encoder embedding dimension (192 for ViT-Tiny)
            llm_dim: LLM embedding dimension (1024 for Gemma3-270M)
            kernel_size: Convolution kernel size
            stride: Downsampling stride (2 = 4x token reduction)
        """
        super().__init__()
        
        self.enc_dim = enc_dim
        self.llm_dim = llm_dim
        self.grid_size = 14  # sqrt(196) for 224px / 16 patch
        
        # Spatial convolution for downsampling
        self.conv = nn.Conv2d(
            in_channels=enc_dim,
            out_channels=llm_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
        )
        
        # Layer norm for stable training
        self.norm = nn.LayerNorm(llm_dim)
        
        # Calculate output grid size: floor((14 + 2*1 - 3) / 2 + 1) = 7
        self.out_grid_size = (self.grid_size + 2 * (kernel_size // 2) - kernel_size) // stride + 1
        self.num_out_tokens = self.out_grid_size ** 2  # 49
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transform encoder patches to LLM tokens.
        
        Args:
            x: Encoder output (B, 196, 192)
            
        Returns:
            LLM input tokens (B, 49, 1024)
        """
        B, N, D = x.shape
        H = W = self.grid_size
        
        # Reshape to spatial grid: (B, 196, 192) -> (B, 192, 14, 14)
        x = x.transpose(1, 2).reshape(B, D, H, W)
        
        # Apply strided convolution: (B, 192, 14, 14) -> (B, 1024, 7, 7)
        x = self.conv(x)
        
        # Flatten back to sequence: (B, 1024, 7, 7) -> (B, 49, 1024)
        x = x.flatten(2).transpose(1, 2)
        
        # Normalize
        x = self.norm(x)
        
        return x
    
    @property
    def output_shape(self) -> tuple:
        """Returns expected output shape: (num_tokens, llm_dim)"""
        return (self.num_out_tokens, self.llm_dim)


if __name__ == "__main__":
    # Quick test
    print("Testing CAbstractor...")
    
    connector = CAbstractor(enc_dim=192, llm_dim=1024)
    print(f"  Encoder dim: {connector.enc_dim}")
    print(f"  LLM dim: {connector.llm_dim}")
    print(f"  Output tokens: {connector.num_out_tokens}")
    print(f"  Output shape: {connector.output_shape}")
    
    # Test forward pass (simulating encoder output)
    dummy_encoder_out = torch.randn(2, 196, 192)
    out = connector(dummy_encoder_out)
    print(f"  Forward test: {dummy_encoder_out.shape} -> {out.shape}")
    
    # Verify output
    assert out.shape == (2, 49, 1024), f"Expected (2, 49, 1024), got {out.shape}"
    print("All tests passed!")
