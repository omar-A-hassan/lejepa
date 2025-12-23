"""
Test LeJEPA compatibility with our VisionEncoder.

Verifies that:
1. Encoder output shape works with LeJEPA's SIGReg loss
2. Projector head can be attached for self-supervised training
"""

import torch
import torch.nn as nn
import sys
sys.path.insert(0, '.')

from lejepa_caption.models.encoder import VisionEncoder

# Import LeJEPA's SIGReg
try:
    import lejepa
    from lejepa.univariate import EppsPulley
    from lejepa.multivariate import SlicingUnivariateTest
    HAS_LEJEPA = True
    print(" LeJEPA package found")
except ImportError:
    HAS_LEJEPA = False
    print("LeJEPA package not installed, testing basic compatibility only")


def test_basic_encoder():
    """Test basic encoder functionality."""
    print("\n=== Test 1: Basic Encoder ===")
    encoder = VisionEncoder(pretrained=False)
    
    x = torch.randn(4, 3, 224, 224)
    out = encoder(x)
    
    print(f"  Input: {x.shape}")
    print(f"  Output: {out.shape}")
    print(f"  Embed dim: {encoder.embed_dim}")
    
    assert out.shape == (4, 196, 192), f"Wrong shape: {out.shape}"
    print("Passed")


def test_with_projector():
    """Test encoder + projector (LeJEPA training setup)."""
    print("\n=== Test 2: Encoder + Projector ===")
    
    encoder = VisionEncoder(pretrained=False)
    
    # LeJEPA uses a projector to map embeddings to a lower-dim space
    projector = nn.Sequential(
        nn.Linear(192, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128),  # Project to 128-dim for SIGReg
    )
    
    x = torch.randn(4, 3, 224, 224)
    
    # Encoder output
    emb = encoder(x)  # (4, 196, 192)
    
    # Pool to get global representation (for LeJEPA)
    pooled = emb.mean(dim=1)  # (4, 192)
    
    # Project
    proj = projector(pooled)  # (4, 128)
    
    print(f"  Encoder output: {emb.shape}")
    print(f"  Pooled: {pooled.shape}")
    print(f"  Projected: {proj.shape}")
    print("Passed")


def test_with_sigreg():
    """Test full LeJEPA training loop (if lejepa package available)."""
    if not HAS_LEJEPA:
        print("\n=== Test 3: SIGReg (SKIPPED - lejepa not installed) ===")
        return
    
    print("\n=== Test 3: SIGReg Loss ===")
    
    encoder = VisionEncoder(pretrained=False)
    projector = nn.Sequential(
        nn.Linear(192, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 128),
    )
    
    # Create SIGReg loss
    univariate_test = EppsPulley(n_points=17)
    sigreg = SlicingUnivariateTest(
        univariate_test=univariate_test,
        num_slices=256
    )
    
    # Forward pass
    x = torch.randn(8, 3, 224, 224)
    emb = encoder(x)
    pooled = emb.mean(dim=1)
    proj = projector(pooled)
    
    # Compute SIGReg loss
    loss = sigreg(proj)
    
    print(f"  Projected shape: {proj.shape}")
    print(f"  SIGReg loss: {loss.item():.4f}")
    
    # Test backward
    loss.backward()
    print("Backward pass successful")


if __name__ == "__main__":
    print("=" * 50)
    print("LeJEPA Compatibility Test")
    print("=" * 50)
    
    test_basic_encoder()
    test_with_projector()
    test_with_sigreg()
    
    print("\n" + "=" * 50)
    print("All compatibility tests passed!")
    print("=" * 50)
