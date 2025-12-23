"""
End-to-end integration test for LeJEPA Captioner.

Tests the full pipeline from image to predicted embeddings.
"""

import torch
import sys
sys.path.insert(0, '.')

from lejepa_caption.models import LeJEPACaptioner, get_captioner


def test_full_pipeline():
    """Test complete forward pass."""
    print("=== Test 1: Full Pipeline ===")
    
    model = LeJEPACaptioner()
    model.eval()
    
    # Simulated batch of images
    images = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        # Full forward: image -> predicted text embeddings
        pred_embeds = model(images)
    
    print(f"  Input: {images.shape}")
    print(f"  Output: {pred_embeds.shape}")
    print(f"  Expected: (4, 50, 1024)")
    
    assert pred_embeds.shape == (4, 50, 1024)
    print("  PASSED")


def test_vision_encoding():
    """Test vision-only encoding (for retrieval)."""
    print("\n=== Test 2: Vision Encoding ===")
    
    model = LeJEPACaptioner()
    model.eval()
    
    images = torch.randn(4, 3, 224, 224)
    
    with torch.no_grad():
        vision_tokens = model.encode_vision(images)
    
    print(f"  Input: {images.shape}")
    print(f"  Vision tokens: {vision_tokens.shape}")
    print(f"  Expected: (4, 49, 1024)")
    
    assert vision_tokens.shape == (4, 49, 1024)
    print("  PASSED")


def test_variable_length():
    """Test predicting different caption lengths."""
    print("\n=== Test 3: Variable Length ===")
    
    model = LeJEPACaptioner()
    model.eval()
    
    images = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        short = model(images, num_tokens=10)
        medium = model(images, num_tokens=25)
        long = model(images, num_tokens=50)
    
    print(f"  Short (10): {short.shape}")
    print(f"  Medium (25): {medium.shape}")
    print(f"  Long (50): {long.shape}")
    
    assert short.shape == (2, 10, 1024)
    assert medium.shape == (2, 25, 1024)
    assert long.shape == (2, 50, 1024)
    print("  PASSED")


def test_model_configs():
    """Test different model sizes."""
    print("\n=== Test 4: Model Configs ===")
    
    for config in ['tiny', 'small', 'base']:
        model = get_captioner(config)
        params = model.num_parameters['total'] / 1e6
        print(f"  {config}: {params:.1f}M params")
    
    print("  PASSED")


def test_gradient_flow():
    """Test that gradients flow through the model."""
    print("\n=== Test 5: Gradient Flow ===")
    
    model = LeJEPACaptioner()
    model.train()
    
    images = torch.randn(2, 3, 224, 224)
    pred_embeds = model(images)
    
    # Simple MSE loss against random target
    target = torch.randn_like(pred_embeds)
    loss = torch.nn.functional.mse_loss(pred_embeds, target)
    
    loss.backward()
    
    # Check gradients exist
    has_grad = all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients: {'OK' if has_grad else 'MISSING'}")
    
    assert has_grad
    print("  PASSED")


if __name__ == "__main__":
    print("=" * 50)
    print("LeJEPA Captioner Integration Test")
    print("=" * 50)
    
    test_full_pipeline()
    test_vision_encoding()
    test_variable_length()
    test_model_configs()
    test_gradient_flow()
    
    print("\n" + "=" * 50)
    print("ALL INTEGRATION TESTS PASSED!")
    print("=" * 50)
