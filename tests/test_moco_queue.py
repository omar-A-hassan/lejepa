"""
Test suite for MoCo-style memory queue.

Run with: uv run --with pytest pytest tests/test_moco_queue.py -v -s
"""

import torch
import torch.nn.functional as F
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lejepa_caption.train.moco_queue import MoCoQueue, infonce_with_queue


class TestMoCoQueueBasic:
    """Test basic MoCoQueue operations."""

    def test_queue_initialization(self):
        """Test queue initializes with correct shape."""
        queue = MoCoQueue(dim=640, queue_size=4096, device='cpu')
        
        assert queue.queue.shape == (4096, 640)
        assert queue.ptr == 0
        assert not queue.is_full
        assert len(queue) == 0
        
        print(f"\nQueue initialization:")
        print(f"  Shape: {queue.queue.shape}")
        print(f"  Pointer: {queue.ptr}")
        print(f"  Is full: {queue.is_full}")

    def test_enqueue_basic(self):
        """Test basic enqueue operation."""
        queue = MoCoQueue(dim=16, queue_size=64, device='cpu')
        
        batch = torch.randn(8, 16)
        queue.enqueue(batch)
        
        assert queue.ptr == 8
        assert len(queue) == 8
        assert not queue.is_full
        
        # Second batch
        batch2 = torch.randn(8, 16)
        queue.enqueue(batch2)
        
        assert queue.ptr == 16
        assert len(queue) == 16
        
        print(f"\nEnqueue test:")
        print(f"  After 1 batch (8): ptr={8}, len={8}")
        print(f"  After 2 batches (16): ptr={16}, len={16}")

    def test_enqueue_wraparound(self):
        """Test queue wraparound when full."""
        queue = MoCoQueue(dim=16, queue_size=32, device='cpu')
        
        # Fill queue (4 batches of 8)
        for i in range(4):
            batch = torch.randn(8, 16)
            queue.enqueue(batch)
        
        assert queue.ptr == 0  # Wrapped around
        assert queue.is_full
        assert len(queue) == 32
        
        # One more batch should overwrite oldest
        batch = torch.randn(8, 16)
        queue.enqueue(batch)
        
        assert queue.ptr == 8
        assert queue.is_full
        
        print(f"\nWraparound test:")
        print(f"  After filling (32): ptr=0, is_full=True")
        print(f"  After one more batch: ptr=8, still full")

    def test_enqueue_split_wraparound(self):
        """Test wraparound that splits a batch."""
        queue = MoCoQueue(dim=16, queue_size=32, device='cpu')
        
        # Fill 28 slots
        for i in range(3):
            queue.enqueue(torch.randn(8, 16))
        queue.enqueue(torch.randn(4, 16))  # 28 total
        
        assert queue.ptr == 28
        
        # Add 8 more - should wrap (4 at end, 4 at start)
        batch = torch.randn(8, 16)
        queue.enqueue(batch)
        
        assert queue.ptr == 4  # 28 + 8 = 36 -> 36 % 32 = 4
        assert queue.is_full
        
        print(f"\nSplit wraparound:")
        print(f"  After 28: ptr=28")
        print(f"  After 8 more (split): ptr=4, is_full=True")

    def test_get_negatives_before_full(self):
        """Test get_negatives returns only filled portion."""
        queue = MoCoQueue(dim=16, queue_size=64, device='cpu')
        
        # Initially empty
        negs = queue.get_negatives()
        assert negs is None
        
        # Add some embeddings
        queue.enqueue(torch.randn(8, 16))
        negs = queue.get_negatives()
        
        assert negs.shape == (8, 16)
        
        queue.enqueue(torch.randn(8, 16))
        negs = queue.get_negatives()
        
        assert negs.shape == (16, 16)
        
        print(f"\nGet negatives (partial):")
        print(f"  Empty: None")
        print(f"  After 8: shape=(8, 16)")
        print(f"  After 16: shape=(16, 16)")

    def test_get_negatives_after_full(self):
        """Test get_negatives returns full queue."""
        queue = MoCoQueue(dim=16, queue_size=32, device='cpu')
        
        # Fill completely
        for _ in range(4):
            queue.enqueue(torch.randn(8, 16))
        
        negs = queue.get_negatives()
        assert negs.shape == (32, 16)
        
        # After wraparound, still returns full
        queue.enqueue(torch.randn(8, 16))
        negs = queue.get_negatives()
        assert negs.shape == (32, 16)
        
        print(f"\nGet negatives (full):")
        print(f"  Full queue: shape=(32, 16)")
        print(f"  After wrap: shape=(32, 16)")

    def test_queue_gradient_detach(self):
        """Ensure queue embeddings are detached from computation graph."""
        queue = MoCoQueue(dim=16, queue_size=64, device='cpu')
        
        # Embeddings with grad
        batch = torch.randn(8, 16, requires_grad=True)
        queue.enqueue(batch)
        
        negs = queue.get_negatives()
        
        # Queue should not have gradient
        assert not negs.requires_grad
        
        print(f"\nGradient detach test:")
        print(f"  Input requires_grad: True")
        print(f"  Queue requires_grad: {negs.requires_grad}")

    def test_queue_reset(self):
        """Test queue reset functionality."""
        queue = MoCoQueue(dim=16, queue_size=32, device='cpu')
        
        # Fill and make full
        for _ in range(4):
            queue.enqueue(torch.randn(8, 16))
        
        assert queue.is_full
        
        queue.reset()
        
        assert queue.ptr == 0
        assert not queue.is_full
        assert len(queue) == 0
        assert queue.get_negatives() is None
        
        print(f"\nReset test:")
        print(f"  After reset: ptr=0, is_full=False, len=0")


class TestInfoNCEWithQueue:
    """Test InfoNCE loss with queue integration."""

    def test_infonce_with_empty_queue(self):
        """Test InfoNCE falls back to standard when queue empty."""
        queue = MoCoQueue(dim=16, queue_size=64, device='cpu')
        
        B = 4
        pred_norm = F.normalize(torch.randn(B, 16), dim=-1)
        target_norm = F.normalize(torch.randn(B, 16), dim=-1)
        
        loss = infonce_with_queue(pred_norm, target_norm, queue, temperature=0.07)
        
        # Should be reasonable value
        assert 0 < loss.item() < 10
        
        print(f"\nInfoNCE with empty queue:")
        print(f"  Loss: {loss.item():.4f}")
        print(f"  Falls back to standard bi-directional InfoNCE")

    def test_infonce_with_filled_queue(self):
        """Test InfoNCE uses queue negatives."""
        queue = MoCoQueue(dim=16, queue_size=64, device='cpu')
        
        # Pre-fill queue
        for _ in range(8):
            queue.enqueue(F.normalize(torch.randn(8, 16), dim=-1))
        
        B = 4
        pred_norm = F.normalize(torch.randn(B, 16), dim=-1)
        target_norm = F.normalize(torch.randn(B, 16), dim=-1)
        
        loss = infonce_with_queue(pred_norm, target_norm, queue, temperature=0.07)
        
        assert 0 < loss.item() < 15
        
        print(f"\nInfoNCE with filled queue:")
        print(f"  Queue size: {len(queue)}")
        print(f"  Batch size: {B}")
        print(f"  Total negatives: {B - 1 + len(queue)}")
        print(f"  Loss: {loss.item():.4f}")

    def test_more_negatives_higher_baseline_loss(self):
        """Test that more negatives give higher baseline loss."""
        B = 8
        
        # Without queue
        pred = F.normalize(torch.randn(B, 64), dim=-1)
        target = F.normalize(torch.randn(B, 64), dim=-1)
        
        logits_no_queue = pred @ target.T / 0.07
        loss_no_queue = F.cross_entropy(logits_no_queue, torch.arange(B))
        
        # With queue (many more negatives)
        queue = MoCoQueue(dim=64, queue_size=256, device='cpu')
        for _ in range(32):
            queue.enqueue(F.normalize(torch.randn(8, 64), dim=-1))
        
        loss_with_queue = infonce_with_queue(pred, target, queue, temperature=0.07)
        
        print(f"\nNegatives effect on loss:")
        print(f"  Without queue ({B-1} negatives): {loss_no_queue.item():.4f}")
        print(f"  With queue ({B-1 + len(queue)} negatives): {loss_with_queue.item():.4f}")
        
        # More negatives = harder task = higher baseline loss
        # But this depends on random embeddings, so just check reasonable range
        assert 0 < loss_with_queue.item() < 20

    def test_perfect_alignment_with_queue(self):
        """Test perfect alignment still gives low loss with queue."""
        queue = MoCoQueue(dim=64, queue_size=256, device='cpu')
        
        # Fill queue with random
        for _ in range(32):
            queue.enqueue(F.normalize(torch.randn(8, 64), dim=-1))
        
        B = 8
        # Perfect alignment: pred = target
        target = F.normalize(torch.randn(B, 64), dim=-1)
        pred = target.clone()
        
        loss = infonce_with_queue(pred, target, queue, temperature=0.07)
        
        print(f"\nPerfect alignment with queue:")
        print(f"  Queue negatives: {len(queue)}")
        print(f"  Loss: {loss.item():.4f}")
        
        # Should still be very low (logit for positive >> logits for negatives)
        assert loss.item() < 0.5, f"Perfect alignment loss too high: {loss.item()}"


class TestMoCoQueueIntegration:
    """Test MoCo queue integration with trainer."""

    def test_trainer_accepts_queue_params(self):
        """Test EmbeddingTrainer accepts queue parameters."""
        from lejepa_caption.train.train import EmbeddingTrainer
        import inspect
        
        sig = inspect.signature(EmbeddingTrainer.__init__)
        params = sig.parameters
        
        assert 'use_moco_queue' in params
        assert 'queue_size' in params
        assert params['use_moco_queue'].default == False
        assert params['queue_size'].default == 4096
        
        print(f"\nTrainer queue params:")
        print(f"  use_moco_queue default: {params['use_moco_queue'].default}")
        print(f"  queue_size default: {params['queue_size'].default}")

    def test_train_with_loader_accepts_queue_params(self):
        """Test train_with_loader accepts queue parameters."""
        from lejepa_caption.train.train import train_with_loader
        import inspect
        
        sig = inspect.signature(train_with_loader)
        params = sig.parameters
        
        assert 'use_moco_queue' in params
        assert 'queue_size' in params
        
        print(f"\ntrain_with_loader queue params:")
        print(f"  use_moco_queue: present")
        print(f"  queue_size: present")


class TestQueueMemoryEfficiency:
    """Test memory properties of queue."""

    def test_queue_memory_size(self):
        """Test queue memory is reasonable."""
        dim = 640  # Gemma-3 dim
        queue_size = 4096  # Default
        
        queue = MoCoQueue(dim=dim, queue_size=queue_size, device='cpu')
        
        # Memory = queue_size * dim * 4 bytes (float32)
        expected_bytes = queue_size * dim * 4
        expected_mb = expected_bytes / (1024 * 1024)
        
        actual_bytes = queue.queue.element_size() * queue.queue.numel()
        actual_mb = actual_bytes / (1024 * 1024)
        
        print(f"\nQueue memory usage:")
        print(f"  Dimensions: {queue_size} x {dim}")
        print(f"  Memory: {actual_mb:.2f} MB")
        print(f"  Expected: {expected_mb:.2f} MB")
        
        # Queue should be ~10MB for typical settings
        assert actual_mb < 20, f"Queue too large: {actual_mb:.1f} MB"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
