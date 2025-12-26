"""
Test suite for combined InfoNCE + MSE loss.

Run with: uv run --with pytest pytest tests/test_combined_loss.py -v -s
"""

import torch
import torch.nn.functional as F
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestCombinedLoss:
    """Test the combined InfoNCE + MSE loss implementation."""

    def compute_infonce_loss(self, pred_embeds, target_embeds, temperature=0.07, mse_alpha=0.0):
        """
        Standalone implementation matching train.py's infonce_loss.
        
        Args:
            pred_embeds: (B, L, D) predicted embeddings
            target_embeds: (B, L, D) target embeddings
            temperature: InfoNCE temperature
            mse_alpha: MSE weight (0 = pure InfoNCE)
        
        Returns:
            dict with loss components
        """
        B, L, D = pred_embeds.shape
        device = pred_embeds.device

        # Pool sequence dimension
        pred_pooled = pred_embeds.mean(dim=1)
        target_pooled = target_embeds.mean(dim=1)

        # L2 normalize
        pred_norm = F.normalize(pred_pooled, dim=-1, p=2)
        target_norm = F.normalize(target_pooled, dim=-1, p=2)

        # InfoNCE
        logits = (pred_norm @ target_norm.T) / temperature
        labels = torch.arange(B, device=device)
        loss_pt = F.cross_entropy(logits, labels)
        loss_tp = F.cross_entropy(logits.T, labels)
        infonce_loss = (loss_pt + loss_tp) / 2

        # MSE on normalized embeddings
        mse_loss = F.mse_loss(pred_norm, target_norm)

        # Combined
        total_loss = infonce_loss + mse_alpha * mse_loss

        # Metrics
        with torch.no_grad():
            alignment = (pred_norm * target_norm).sum(dim=-1).mean()
            uniformity = (pred_norm @ pred_norm.T).triu(diagonal=1).mean()

        return {
            'loss': total_loss,
            'infonce': infonce_loss.item(),
            'mse': mse_loss.item(),
            'alignment': alignment.item(),
            'uniformity': uniformity.item(),
        }

    def test_pure_infonce_when_alpha_zero(self):
        """Test that alpha=0 gives pure InfoNCE loss."""
        B, L, D = 4, 50, 640
        pred = torch.randn(B, L, D)
        target = torch.randn(B, L, D)

        result = self.compute_infonce_loss(pred, target, mse_alpha=0.0)

        # With alpha=0, total loss should equal infonce loss
        assert abs(result['loss'].item() - result['infonce']) < 1e-5, \
            f"With alpha=0, loss should equal infonce: {result['loss'].item()} vs {result['infonce']}"

        print(f"\nPure InfoNCE (alpha=0):")
        print(f"  Total loss: {result['loss'].item():.4f}")
        print(f"  InfoNCE: {result['infonce']:.4f}")
        print(f"  MSE: {result['mse']:.4f} (not added)")

    def test_combined_loss_with_alpha(self):
        """Test that alpha > 0 adds MSE component."""
        B, L, D = 4, 50, 640
        pred = torch.randn(B, L, D)
        target = torch.randn(B, L, D)
        
        alpha = 0.15

        result = self.compute_infonce_loss(pred, target, mse_alpha=alpha)

        expected_total = result['infonce'] + alpha * result['mse']
        actual_total = result['loss'].item()

        assert abs(actual_total - expected_total) < 1e-4, \
            f"Combined loss mismatch: {actual_total:.4f} vs {expected_total:.4f}"

        print(f"\nCombined Loss (alpha={alpha}):")
        print(f"  InfoNCE: {result['infonce']:.4f}")
        print(f"  MSE: {result['mse']:.4f}")
        print(f"  Total: {actual_total:.4f}")
        print(f"  Expected: {expected_total:.4f}")

    def test_mse_component_shape(self):
        """Test MSE is computed on pooled, normalized embeddings."""
        B, L, D = 4, 50, 640
        pred = torch.randn(B, L, D)
        target = torch.randn(B, L, D)

        # Pool and normalize manually
        pred_pooled = F.normalize(pred.mean(dim=1), dim=-1)
        target_pooled = F.normalize(target.mean(dim=1), dim=-1)

        mse_expected = F.mse_loss(pred_pooled, target_pooled).item()

        result = self.compute_infonce_loss(pred, target, mse_alpha=0.15)

        assert abs(result['mse'] - mse_expected) < 1e-5, \
            f"MSE mismatch: {result['mse']:.4f} vs {mse_expected:.4f}"

        print(f"\nMSE Component:")
        print(f"  Computed on (B, D) = ({B}, {D}) normalized embeddings")
        print(f"  MSE value: {result['mse']:.4f}")

    def test_alpha_sweep(self):
        """Test different alpha values."""
        B, L, D = 8, 50, 640
        pred = torch.randn(B, L, D)
        target = torch.randn(B, L, D)

        alphas = [0.0, 0.1, 0.15, 0.2, 0.3]
        
        print("\nAlpha Sweep:")
        results = []
        for alpha in alphas:
            result = self.compute_infonce_loss(pred, target, mse_alpha=alpha)
            results.append(result)
            print(f"  α={alpha:.2f}: loss={result['loss'].item():.4f}, "
                  f"infonce={result['infonce']:.4f}, mse={result['mse']:.4f}")

        # Higher alpha should give higher total loss (since MSE is positive)
        for i in range(len(alphas) - 1):
            assert results[i+1]['loss'].item() >= results[i]['loss'].item() - 1e-4, \
                f"Loss should increase with alpha: α={alphas[i]}->{alphas[i+1]}"

    def test_gradients_flow(self):
        """Test that gradients flow through combined loss."""
        B, L, D = 4, 50, 640
        pred = torch.randn(B, L, D, requires_grad=True)
        target = torch.randn(B, L, D)

        result = self.compute_infonce_loss(pred, target, mse_alpha=0.15)
        result['loss'].backward()

        assert pred.grad is not None, "No gradients computed"
        assert not torch.isnan(pred.grad).any(), "Gradients contain NaN"
        assert pred.grad.abs().sum() > 0, "All gradients are zero"

        print(f"\nGradient Flow Test:")
        print(f"  Gradient norm: {pred.grad.norm():.4f}")
        print("  Gradients flowing correctly")

    def test_perfect_alignment_low_loss(self):
        """Test that perfect alignment gives low loss."""
        B, L, D = 4, 50, 640
        pred = torch.randn(B, L, D)
        target = pred.clone()

        result = self.compute_infonce_loss(pred, target, mse_alpha=0.15)

        # Perfect alignment should give:
        # - InfoNCE close to 0
        # - MSE exactly 0
        # - Alignment close to 1
        assert result['infonce'] < 0.1, f"Perfect alignment: infonce too high {result['infonce']}"
        assert result['mse'] < 1e-5, f"Perfect alignment: mse should be ~0, got {result['mse']}"
        assert result['alignment'] > 0.99, f"Perfect alignment: alignment should be ~1, got {result['alignment']}"

        print(f"\nPerfect Alignment Test:")
        print(f"  InfoNCE: {result['infonce']:.6f}")
        print(f"  MSE: {result['mse']:.6f}")
        print(f"  Alignment: {result['alignment']:.4f}")

    def test_backward_compatibility_alpha_zero(self):
        """Test that alpha=0 is backward compatible with pure InfoNCE."""
        B, L, D = 8, 50, 640
        pred = torch.randn(B, L, D)
        target = torch.randn(B, L, D)

        # Pure InfoNCE (manual)
        pred_pooled = pred.mean(dim=1)
        target_pooled = target.mean(dim=1)
        pred_norm = F.normalize(pred_pooled, dim=-1)
        target_norm = F.normalize(target_pooled, dim=-1)
        
        logits = pred_norm @ target_norm.T / 0.07
        labels = torch.arange(B)
        pure_infonce = (F.cross_entropy(logits, labels) + 
                        F.cross_entropy(logits.T, labels)) / 2

        # Combined with alpha=0
        result = self.compute_infonce_loss(pred, target, mse_alpha=0.0)

        assert abs(pure_infonce.item() - result['loss'].item()) < 1e-5, \
            f"Backward compatibility failed: {pure_infonce.item()} vs {result['loss'].item()}"

        print(f"\nBackward Compatibility (alpha=0):")
        print(f"  Pure InfoNCE: {pure_infonce.item():.4f}")
        print(f"  Combined (α=0): {result['loss'].item():.4f}")
        print("  Backward compatible")


class TestCombinedLossWithTrainer:
    """Test combined loss integration with EmbeddingTrainer."""

    @pytest.fixture
    def mock_trainer_components(self):
        """Create mock components for trainer testing."""
        from lejepa_caption.models import get_captioner
        model = get_captioner(config="small", encoder_pretrained=False)
        return model

    def test_trainer_with_mse_alpha_zero(self, mock_trainer_components):
        """Test trainer initializes correctly with alpha=0."""
        model = mock_trainer_components
        
        # We can't fully test without LLM, but we can check the class
        from lejepa_caption.train.train import EmbeddingTrainer
        
        # Check EmbeddingTrainer accepts mse_alpha parameter
        import inspect
        sig = inspect.signature(EmbeddingTrainer.__init__)
        params = sig.parameters
        
        assert 'mse_alpha' in params, "EmbeddingTrainer should accept mse_alpha parameter"
        assert params['mse_alpha'].default == 0.0, "mse_alpha default should be 0.0"

        print("\nTrainer Integration:")
        print("  mse_alpha parameter exists")
        print("  Default value is 0.0 (backward compatible)")

    def test_train_with_loader_accepts_mse_alpha(self):
        """Test train_with_loader accepts mse_alpha parameter."""
        from lejepa_caption.train.train import train_with_loader
        import inspect
        
        sig = inspect.signature(train_with_loader)
        params = sig.parameters
        
        assert 'mse_alpha' in params, "train_with_loader should accept mse_alpha parameter"
        assert params['mse_alpha'].default == 0.0, "mse_alpha default should be 0.0"

        print("\ntrain_with_loader Integration:")
        print("  mse_alpha parameter exists")
        print("  Default value is 0.0")


class TestMSEOnNormalizedEmbeddings:
    """Test properties of MSE on normalized embeddings."""

    def test_mse_range_normalized(self):
        """Test MSE range for normalized embeddings."""
        B, D = 8, 640

        # Random normalized embeddings
        pred = F.normalize(torch.randn(B, D), dim=-1)
        target = F.normalize(torch.randn(B, D), dim=-1)

        mse = F.mse_loss(pred, target).item()

        # For normalized vectors, MSE = 2 * (1 - cos_sim) / D
        # Max MSE (opposite vectors) ≈ 4/D (very small for D=640)
        # But with D terms, total MSE can be ~2 for orthogonal vectors
        # Actually: MSE = mean((p - t)^2) = mean(p^2 - 2pt + t^2) = 2 - 2*mean(pt)
        # For orthogonal: MSE ≈ 2, for aligned: MSE ≈ 0, for opposite: MSE ≈ 4

        assert 0 <= mse <= 4, f"MSE should be in [0, 4], got {mse}"

        print(f"\nMSE on Normalized Embeddings:")
        print(f"  MSE value: {mse:.4f}")
        print(f"  Expected range: [0, 4]")
        print(f"  For random orthogonal: ~2.0")

    def test_mse_vs_cosine_relationship(self):
        """Test relationship between MSE and cosine similarity."""
        B, D = 8, 640

        pred = F.normalize(torch.randn(B, D), dim=-1)
        target = F.normalize(torch.randn(B, D), dim=-1)

        mse = F.mse_loss(pred, target).item()
        
        # For normalized unit vectors: ||p - t||^2 = 2 - 2*<p,t>
        # MSE = mean over all elements = mean(||p - t||^2) / D
        # So: MSE = (2 - 2*cos_sim) / D for each sample
        # But torch.mse_loss averages over ALL elements (B*D)
        
        # Compute expected MSE manually
        diff_sq = (pred - target).pow(2)  # (B, D)
        expected_mse = diff_sq.mean().item()  # Mean over all B*D elements
        
        assert abs(mse - expected_mse) < 1e-5, \
            f"MSE mismatch: {mse:.6f} vs {expected_mse:.6f}"
        
        # Also verify the relationship per sample
        cos_sim_per_sample = (pred * target).sum(dim=-1)  # (B,)
        squared_norm_per_sample = diff_sq.sum(dim=-1)  # (B,)
        expected_sq_norm = 2 - 2 * cos_sim_per_sample  # (B,)
        
        assert torch.allclose(squared_norm_per_sample, expected_sq_norm, atol=1e-4), \
            "||p-t||^2 = 2 - 2*<p,t> relationship doesn't hold"

        print(f"\nMSE vs Cosine Relationship:")
        print(f"  MSE (mean over B*D): {mse:.6f}")
        print(f"  Expected: {expected_mse:.6f}")
        print(f"  Mean cosine similarity: {cos_sim_per_sample.mean():.4f}")
        print("  ✓ Relationship holds: ||p-t||^2 = 2(1 - cos_sim)")


class TestGradCacheIntegration:
    """Test GradCache integration with train_with_loader."""
    
    def test_train_with_loader_accepts_gradcache_params(self):
        """Test that train_with_loader accepts GradCache parameters."""
        import inspect
        from lejepa_caption.train import train_with_loader
        
        sig = inspect.signature(train_with_loader)
        params = list(sig.parameters.keys())
        
        # Check GradCache parameters exist
        assert 'use_gradcache' in params, "Missing use_gradcache parameter"
        assert 'gradcache_accum_steps' in params, "Missing gradcache_accum_steps parameter"
        
        # Check defaults
        assert sig.parameters['use_gradcache'].default is False
        assert sig.parameters['gradcache_accum_steps'].default == 4
    
    def test_embedding_trainer_accepts_gradcache_params(self):
        """Test that EmbeddingTrainer accepts GradCache parameters."""
        import inspect
        from lejepa_caption.train.train import EmbeddingTrainer
        
        sig = inspect.signature(EmbeddingTrainer.__init__)
        params = list(sig.parameters.keys())
        
        assert 'use_gradcache' in params, "Missing use_gradcache parameter"
        assert 'gradcache_chunk_size' in params, "Missing gradcache_chunk_size parameter"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
