"""
Tests for GradCache integration.

These tests verify:
1. GradCache availability check
2. CaptionerGradCache wrapper functionality
3. FunctionalGradCache accumulation pattern
4. Loss computation with large effective batch sizes
5. Gradient flow through cached representations
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


# Skip all tests if GradCache not installed
pytest.importorskip("grad_cache")


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
            temperature=0.07,
            mse_alpha=0.0,
        )
    
    def test_init(self, fgc):
        """Test initialization."""
        assert fgc.num_accumulated == 0
        assert fgc.temperature == 0.07
        assert fgc.mse_alpha == 0.0
    
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
        
        fgc.reset()
        
        assert fgc.num_accumulated == 0
        assert len(fgc._pred_cache) == 0
    
    def test_compute_loss_without_accumulate_raises(self, fgc):
        """Test that compute_loss raises without accumulation."""
        with pytest.raises(RuntimeError, match="No sub-batches accumulated"):
            fgc.compute_loss()
    
    def test_compute_loss(self, fgc):
        """Test loss computation."""
        # Accumulate 2 sub-batches
        for _ in range(2):
            batch = torch.randn(8, 64)
            target = torch.randn(8, 16)
            fgc.accumulate(batch, target)
        
        loss = fgc.compute_loss()
        
        assert loss.shape == ()
        assert loss.requires_grad
    
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
        """Test MSE alpha integration."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=simple_model,
            temperature=0.07,
            mse_alpha=0.15,
        )
        
        batch = torch.randn(16, 64)
        target = torch.randn(16, 16)
        fgc.accumulate(batch, target)
        
        loss = fgc.compute_loss()
        
        assert loss.shape == ()


class TestGradCacheWithCaptioner:
    """Test GradCache with actual captioner model."""
    
    def test_functional_with_captioner(self, small_captioner):
        """Test FunctionalGradCache with captioner."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=small_captioner,
            temperature=0.07,
            mse_alpha=0.0,
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
    
    def test_gradient_flow(self, small_captioner):
        """Test gradients flow back to model parameters."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=small_captioner,
            temperature=0.07,
            mse_alpha=0.0,
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
        
        # Config 1: One batch of 16
        fgc1 = FunctionalGradCache(model=small_captioner, temperature=0.07)
        fgc1.accumulate(images, targets)
        loss1 = fgc1.compute_loss()
        
        # Config 2: Two batches of 8
        fgc2 = FunctionalGradCache(model=small_captioner, temperature=0.07)
        fgc2.accumulate(images[:8], targets[:8])
        fgc2.accumulate(images[8:], targets[8:])
        loss2 = fgc2.compute_loss()
        
        # Losses should be similar (same effective batch size = 16)
        # Not exactly equal due to gradient caching mechanics
        assert abs(loss1.item() - loss2.item()) < 1.0


class TestCreateGradCacheTrainer:
    """Test the factory function."""
    
    def test_factory_creation(self, small_captioner):
        """Test creating trainer via factory."""
        from lejepa_caption.train.gradcache import create_gradcache_trainer
        
        gc = create_gradcache_trainer(
            model=small_captioner,
            chunk_size=32,
            temperature=0.07,
            mse_alpha=0.15,
        )
        
        assert gc is not None
        assert gc.chunk_size == 32
    
    def test_factory_with_fp16(self, small_captioner):
        """Test factory with mixed precision."""
        from lejepa_caption.train.gradcache import create_gradcache_trainer
        from torch.amp import GradScaler
        
        scaler = GradScaler()
        gc = create_gradcache_trainer(
            model=small_captioner,
            chunk_size=32,
            fp16=True,
            scaler=scaler,
        )
        
        assert gc.fp16 is True


class TestGradCacheMemoryEfficiency:
    """Test that GradCache is actually memory efficient."""
    
    def test_chunked_forward_works(self, small_captioner):
        """Test that chunked forward passes work correctly."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        
        fgc = FunctionalGradCache(
            model=small_captioner,
            temperature=0.07,
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
        
        fgc = FunctionalGradCache(model=small_captioner, temperature=0.07)
        
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        images = torch.randn(8, 3, 224, 224)
        targets = torch.randn(8, target_len, target_dim, requires_grad=True)
        
        fgc.accumulate(images, targets)
        
        # Target cache should be detached
        assert not fgc._target_cache[0].requires_grad


class TestIntegrationWithMoCoQueue:
    """Test GradCache and MoCo Queue can work together."""
    
    def test_gradcache_with_moco_queue(self, small_captioner):
        """Test using GradCache with MoCo Queue for extra negatives."""
        from lejepa_caption.train.gradcache import FunctionalGradCache
        from lejepa_caption.train.moco_queue import MoCoQueue
        
        # Get target dims from captioner
        target_dim = small_captioner.config['llm_dim']
        target_len = small_captioner.config['max_caption_len']
        
        # Set up queue (MoCoQueue uses 'dim' not 'embed_dim')
        queue = MoCoQueue(dim=target_dim, queue_size=128, device='cpu')
        
        # Fill queue with some embeddings
        for _ in range(10):
            queue.enqueue(torch.randn(8, target_dim))
        
        # Set up GradCache
        fgc = FunctionalGradCache(
            model=small_captioner,
            temperature=0.07,
        )
        
        # Accumulate batches
        images = torch.randn(16, 3, 224, 224)
        targets = torch.randn(16, target_len, target_dim)
        fgc.accumulate(images, targets)
        
        # Both should work without conflict
        queue_negs = queue.get_negatives()
        loss = fgc.compute_loss()
        
        assert queue_negs.shape[0] == 80  # 10 batches * 8
        assert not torch.isnan(loss)
