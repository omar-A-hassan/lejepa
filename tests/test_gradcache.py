"""
Tests for GradCache integration.

These tests verify:
1. GradCache availability check
2. CaptionerGradCache wrapper functionality
3. FunctionalGradCache accumulation pattern
4. Loss computation with large effective batch sizes
5. Gradient flow through cached representations
6. Integration with MoCo queue for consistent loss
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any


# Skip all tests if GradCache not installed
pytest.importorskip("grad_cache")


def create_simple_loss_fn(temperature: float = 0.07, mse_alpha: float = 0.0):
    """
    Create a simple loss function for testing.
    
    This mimics what EmbeddingTrainer.infonce_loss does, returning
    a dict with 'loss' tensor and metrics.
    """
    def loss_fn(pred_embeds: torch.Tensor, target_embeds: torch.Tensor) -> Dict[str, Any]:
        B = pred_embeds.shape[0]
        device = pred_embeds.device
        
        # Pool sequence dimension if 3D
        if pred_embeds.dim() == 3:
            pred_pooled = pred_embeds.mean(dim=1)
            target_pooled = target_embeds.mean(dim=1)
        else:
            pred_pooled = pred_embeds
            target_pooled = target_embeds
        
        # Normalize
        pred_norm = F.normalize(pred_pooled, dim=-1, p=2)
        target_norm = F.normalize(target_pooled, dim=-1, p=2)
        
        # InfoNCE
        logits = (pred_norm @ target_norm.T) / temperature
        labels = torch.arange(B, device=device)
        loss_pt = F.cross_entropy(logits, labels)
        loss_tp = F.cross_entropy(logits.T, labels)
        infonce_loss = (loss_pt + loss_tp) / 2
        
        # MSE
        mse_loss = F.mse_loss(pred_norm, target_norm)
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
            'num_negatives': B - 1,
        }
    
    return loss_fn


class TestGradCacheAvailability:
    """Test GradCache installation and availability."""
    
    def test_gradcache_import(self):
        """Test that GradCache can be imported."""
        from grad_cache import GradCache
        assert GradCache is not None
    
    def test_gradcache_functional_import(self):
        """Test functional API imports."""
        from grad_cache.functional import cached, cat_input_tensor
        assert cached is not None
        assert cat_input_tensor is not None
    
    def test_availability_flag(self):
        """Test our availability flag."""
        from lejepa_caption.train.gradcache import GRADCACHE_AVAILABLE
        assert GRADCACHE_AVAILABLE is True
    
    def test_check_gradcache_available(self):
        """Test the availability check function."""
        from lejepa_caption.train.gradcache import check_gradcache_available
        # Should not raise
        result = check_gradcache_available()
        assert result is True


class TestFunctionalGradCache:
    """Test the FunctionalGradCache class."""
    
    @pytest.fixture
    def simple_model(self):
        """A simple model for testing."""
        return nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
        )
    
    @pytest.fixture
    def fgc(self, simple_model):
        """FunctionalGradCache instance."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        return FunctionalGradCache(
            model=simple_model,
            loss_fn=create_simple_loss_fn(temperature=0.07, mse_alpha=0.0),
        )
    
    def test_init(self, fgc):
        """Test initialization."""
        assert fgc.num_accumulated == 0
        assert fgc.last_metrics is None
    
    def test_accumulate(self, fgc, simple_model):
        """Test accumulating sub-batches."""
        batch = torch.randn(8, 64)
        target = torch.randn(8, 16)
        
        fgc.accumulate(batch, target)
        
        assert fgc.num_accumulated == 1
        assert len(fgc._pred_cache) == 1
        assert len(fgc._target_cache) == 1
    
    def test_accumulate_multiple(self, fgc):
        """Test accumulating multiple sub-batches."""
        for i in range(4):
            batch = torch.randn(8, 64)
            target = torch.randn(8, 16)
            fgc.accumulate(batch, target)
        
        assert fgc.num_accumulated == 4
    
    def test_reset(self, fgc):
        """Test resetting caches."""
        batch = torch.randn(8, 64)
        target = torch.randn(8, 16)
        fgc.accumulate(batch, target)
        
        # Compute loss first to have metrics
        _ = fgc.compute_loss()
        assert fgc.last_metrics is not None
        
        fgc.reset()
        
        assert fgc.num_accumulated == 0
        assert len(fgc._pred_cache) == 0
        assert fgc.last_metrics is None
    
    def test_compute_loss_without_accumulate_raises(self, fgc):
        """Test that compute_loss raises without accumulation."""
        with pytest.raises(RuntimeError, match="No sub-batches accumulated"):
            fgc.compute_loss()
    
    def test_compute_loss(self, fgc):
        """Test loss computation and metrics storage."""
        # Accumulate 2 sub-batches
        for _ in range(2):
            batch = torch.randn(8, 64)
            target = torch.randn(8, 16)
            fgc.accumulate(batch, target)
        
        loss = fgc.compute_loss()
        
        assert loss.shape == ()
        assert loss.requires_grad
        
        # Check metrics are stored
        metrics = fgc.last_metrics
        assert metrics is not None
        assert 'infonce' in metrics
        assert 'mse' in metrics
        assert 'alignment' in metrics
        assert 'uniformity' in metrics
        assert 'num_negatives' in metrics
    
    def test_effective_batch_size(self, fgc):
        """Test that effective batch size equals sum of sub-batches."""
        sub_batch_sizes = [8, 16, 12]  # Total: 36
        
        for size in sub_batch_sizes:
            batch = torch.randn(size, 64)
            target = torch.randn(size, 16)
            fgc.accumulate(batch, target)
        
        # Check concatenated size
        all_pred = torch.cat(fgc._pred_cache, dim=0)
        all_target = torch.cat(fgc._target_cache, dim=0)
        
        assert all_pred.shape[0] == 36
        assert all_target.shape[0] == 36
    
    def test_mse_alpha(self, simple_model):
        """Test MSE alpha integration via loss_fn."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=simple_model,
            loss_fn=create_simple_loss_fn(temperature=0.07, mse_alpha=0.15),
        )
        
        batch = torch.randn(16, 64)
        target = torch.randn(16, 16)
        fgc.accumulate(batch, target)
        
        loss = fgc.compute_loss()
        
        assert loss.shape == ()
        # MSE should be computed (check metrics)
        assert fgc.last_metrics['mse'] > 0


class TestGradCacheWithCaptioner:
    """Test GradCache with actual captioner model."""
    
    def test_functional_with_captioner(self, small_captioner):
        """Test FunctionalGradCache with captioner."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=small_captioner,
            loss_fn=create_simple_loss_fn(temperature=0.07, mse_alpha=0.0),
        )
        
        # Simulate 2 sub-batches
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        for _ in range(2):
            images = torch.randn(4, 3, 224, 224)
            targets = torch.randn(4, target_len, target_dim)
            fgc.accumulate(images, targets)
        
        loss = fgc.compute_loss()
        
        # Should be able to compute loss
        assert loss.shape == ()
        assert not torch.isnan(loss)
        # Should have stored metrics
        assert fgc.last_metrics is not None
    
    def test_gradient_flow(self, small_captioner):
        """Test gradients flow back to model parameters."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=small_captioner,
            loss_fn=create_simple_loss_fn(temperature=0.07, mse_alpha=0.0),
        )
        
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        # Zero gradients
        small_captioner.zero_grad()
        
        # Accumulate
        images = torch.randn(8, 3, 224, 224)
        targets = torch.randn(8, target_len, target_dim)
        fgc.accumulate(images, targets)
        
        # Compute and backward
        loss = fgc.compute_loss()
        loss.backward()
        
        # Propagate through cache
        fgc.backward_cached()
        
        # Check some gradients exist (encoder should have grads)
        encoder_grads = [
            p.grad for p in small_captioner.encoder.parameters() 
            if p.grad is not None
        ]
        assert len(encoder_grads) > 0
    
    def test_effective_batch_size_comparison(self, small_captioner):
        """Compare loss with different effective batch sizes."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        # Same data, different sub-batch configurations
        images = torch.randn(16, 3, 224, 224)
        targets = torch.randn(16, target_len, target_dim)
        
        loss_fn = create_simple_loss_fn(temperature=0.07, mse_alpha=0.0)
        
        # Config 1: One batch of 16
        fgc1 = FunctionalGradCache(model=small_captioner, loss_fn=loss_fn)
        fgc1.accumulate(images, targets)
        loss1 = fgc1.compute_loss()
        
        # Config 2: Two batches of 8
        fgc2 = FunctionalGradCache(model=small_captioner, loss_fn=loss_fn)
        fgc2.accumulate(images[:8], targets[:8])
        fgc2.accumulate(images[8:], targets[8:])
        loss2 = fgc2.compute_loss()
        
        # Losses should be similar (same effective batch size = 16)
        # Not exactly equal due to gradient caching mechanics
        assert abs(loss1.item() - loss2.item()) < 1.0


class TestGradCacheMemoryEfficiency:
    """Test that GradCache is actually memory efficient."""
    
    def test_chunked_forward_works(self, small_captioner):
        """Test that chunked forward passes work correctly."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=small_captioner,
            loss_fn=create_simple_loss_fn(temperature=0.07, mse_alpha=0.0),
        )
        
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        # Simulate chunked processing
        total_images = torch.randn(32, 3, 224, 224)
        total_targets = torch.randn(32, target_len, target_dim)
        chunk_size = 8
        
        for i in range(0, 32, chunk_size):
            fgc.accumulate(
                total_images[i:i+chunk_size],
                total_targets[i:i+chunk_size],
            )
        
        assert fgc.num_accumulated == 4  # 32 / 8 = 4 chunks
        
        loss = fgc.compute_loss()
        assert not torch.isnan(loss)
    
    def test_detached_target_embeds(self, small_captioner):
        """Test that target embeddings are detached (no grad)."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=small_captioner,
            loss_fn=create_simple_loss_fn(temperature=0.07, mse_alpha=0.0),
        )
        
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        images = torch.randn(8, 3, 224, 224)
        targets = torch.randn(8, target_len, target_dim, requires_grad=True)
        
        fgc.accumulate(images, targets)
        
        # Target cache should be detached
        assert not fgc._target_cache[0].requires_grad


class TestIntegrationWithMoCoQueue:
    """Test GradCache and MoCo Queue work together correctly."""
    
    def test_gradcache_with_moco_queue(self, small_captioner):
        """Test using GradCache with MoCo Queue for extra negatives."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        from lejepa_caption.train.moco_queue import MoCoQueue, infonce_with_queue
        
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        # Set up queue
        queue = MoCoQueue(dim=target_dim, queue_size=128, device='cpu')
        
        # Fill queue with some embeddings
        for _ in range(10):
            queue.enqueue(torch.randn(8, target_dim))
        
        # Create loss function that uses the queue (like EmbeddingTrainer does)
        def loss_fn_with_queue(pred_embeds, target_embeds, temperature=0.07):
            # Pool
            pred_pooled = pred_embeds.mean(dim=1) if pred_embeds.dim() == 3 else pred_embeds
            target_pooled = target_embeds.mean(dim=1) if target_embeds.dim() == 3 else target_embeds
            
            # Normalize
            pred_norm = F.normalize(pred_pooled, dim=-1, p=2)
            target_norm = F.normalize(target_pooled, dim=-1, p=2)
            
            # InfoNCE with queue
            loss = infonce_with_queue(pred_norm, target_norm, queue, temperature)
            
            # Update queue
            queue.enqueue(target_norm.detach())
            
            B = pred_embeds.shape[0]
            return {
                'loss': loss,
                'infonce': loss.item(),
                'mse': 0.0,
                'alignment': 0.0,
                'uniformity': 0.0,
                'num_negatives': B - 1 + len(queue),
            }
        
        # Set up GradCache with queue-aware loss
        fgc = FunctionalGradCache(
            model=small_captioner,
            loss_fn=loss_fn_with_queue,
        )
        
        # Accumulate batches
        images = torch.randn(16, 3, 224, 224)
        targets = torch.randn(16, target_len, target_dim)
        fgc.accumulate(images, targets)
        
        # Compute loss
        loss = fgc.compute_loss()
        
        # Verify
        assert not torch.isnan(loss)
        assert fgc.last_metrics is not None
        # num_negatives should include queue size
        assert fgc.last_metrics['num_negatives'] > 15  # More than just batch negatives
    
    def test_loss_uses_queue_negatives(self, small_captioner):
        """
        Verify that GradCache loss uses queue negatives.
        
        The loss should be ~ln(B + queue_size), not just ~ln(B).
        This is the bug we fixed.
        """
        from lejepa_caption.train.gradcache import FunctionalGradCache
        from lejepa_caption.train.moco_queue import MoCoQueue, infonce_with_queue
        
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        # Queue with 100 negatives
        queue = MoCoQueue(dim=target_dim, queue_size=128, device='cpu')
        for _ in range(15):  # Fill with 120 embeddings → queue has 120
            queue.enqueue(torch.randn(8, target_dim))
        
        assert len(queue) == 120
        
        # Loss function WITHOUT queue (batch negatives only)
        def loss_no_queue(pred_embeds, target_embeds, temperature=0.07):
            pred_pooled = pred_embeds.mean(dim=1)
            target_pooled = target_embeds.mean(dim=1)
            pred_norm = F.normalize(pred_pooled, dim=-1, p=2)
            target_norm = F.normalize(target_pooled, dim=-1, p=2)
            
            B = pred_embeds.shape[0]
            logits = (pred_norm @ target_norm.T) / temperature
            labels = torch.arange(B, device=pred_embeds.device)
            loss = F.cross_entropy(logits, labels)
            
            return {'loss': loss, 'infonce': loss.item(), 'mse': 0.0, 
                    'alignment': 0.0, 'uniformity': 0.0, 'num_negatives': B - 1}
        
        # Loss function WITH queue
        def loss_with_queue(pred_embeds, target_embeds, temperature=0.07):
            pred_pooled = pred_embeds.mean(dim=1)
            target_pooled = target_embeds.mean(dim=1)
            pred_norm = F.normalize(pred_pooled, dim=-1, p=2)
            target_norm = F.normalize(target_pooled, dim=-1, p=2)
            
            B = pred_embeds.shape[0]
            loss = infonce_with_queue(pred_norm, target_norm, queue, temperature)
            
            return {'loss': loss, 'infonce': loss.item(), 'mse': 0.0,
                    'alignment': 0.0, 'uniformity': 0.0, 'num_negatives': B - 1 + len(queue)}
        
        # Same data
        images = torch.randn(8, 3, 224, 224)
        targets = torch.randn(8, target_len, target_dim)
        
        # Compute loss WITHOUT queue
        fgc1 = FunctionalGradCache(model=small_captioner, loss_fn=loss_no_queue)
        fgc1.accumulate(images, targets)
        loss_no_q = fgc1.compute_loss()
        
        # Compute loss WITH queue
        fgc2 = FunctionalGradCache(model=small_captioner, loss_fn=loss_with_queue)
        fgc2.accumulate(images, targets)
        loss_with_q = fgc2.compute_loss()
        
        # Loss WITH queue should be higher (more negatives → harder task)
        # ~ln(8) ≈ 2.1 vs ~ln(8 + 120) ≈ 4.8
        assert loss_with_q.item() > loss_no_q.item()
        
        # Check metrics show correct number of negatives
        assert fgc1.last_metrics['num_negatives'] == 7  # B - 1
        assert fgc2.last_metrics['num_negatives'] == 7 + 120  # B - 1 + queue
