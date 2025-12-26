"""
GradCache integration for large batch contrastive learning.

GradCache enables training with effective batch sizes of 512-1024+
without increasing GPU memory usage. It works by:
1. Forward passing sub-batches and caching representations
2. Computing loss with ALL representations (all negatives)
3. Backward passing in chunks (memory efficient)

Reference: https://github.com/luyug/GradCache
Paper: "Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup"

IMPORTANT: This module integrates with moco_queue.py for additional negatives.
The loss function passed to FunctionalGradCache should use infonce_with_queue()
from moco_queue.py to get consistent training/validation behavior.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Callable, Any, Dict, TYPE_CHECKING
from functools import partial

# Import MoCo queue for type hints
if TYPE_CHECKING:
    from .moco_queue import MoCoQueue

# Try to import GradCache - gracefully handle if not installed
try:
    from grad_cache import GradCache
    from grad_cache.functional import cached, cat_input_tensor
    GRADCACHE_AVAILABLE = True
except ImportError:
    GRADCACHE_AVAILABLE = False
    GradCache = None


def check_gradcache_available():
    """Check if GradCache is available."""
    if not GRADCACHE_AVAILABLE:
        raise ImportError(
            "GradCache not installed. Install with:\n"
            "  uv pip install git+https://github.com/luyug/GradCache.git"
        )
    return True


class CaptionerGradCache:
    """
    GradCache wrapper for LeJEPACaptioner.
    
    This enables training with large effective batch sizes by:
    1. Caching embeddings from sub-batches
    2. Computing contrastive loss with ALL negatives
    3. Memory-efficient chunked backward pass
    
    Parameters
    ----------
    model : LeJEPACaptioner
        The captioner model (encoder + connector + predictor).
    chunk_size : int
        Sub-batch size for forward/backward. Smaller = less memory.
        Recommended: 32-64 for 16GB GPU.
    loss_fn : callable
        Loss function that takes (pred_embeds, target_embeds) and returns loss.
    fp16 : bool
        Whether to use mixed precision.
    scaler : GradScaler, optional
        Gradient scaler for mixed precision training.
        
    Example
    -------
    >>> gc = CaptionerGradCache(model, chunk_size=32, loss_fn=infonce_loss)
    >>> # Accumulate 4 sub-batches for effective batch 128
    >>> loss = gc.cache_step(images_batch, target_embeds_batch)
    >>> optimizer.step()
    """
    
    def __init__(
        self,
        model,
        chunk_size: int = 32,
        loss_fn: Optional[Callable] = None,
        fp16: bool = False,
        scaler=None,
    ):
        check_gradcache_available()
        
        self.model = model
        self.chunk_size = chunk_size
        self.fp16 = fp16
        self.scaler = scaler
        
        # Default loss function: InfoNCE
        if loss_fn is None:
            loss_fn = self._default_infonce_loss
        
        # Create GradCache instance
        # We use a single "model" that takes images and returns embeddings
        self.gc = GradCache(
            models=[model],
            chunk_sizes=chunk_size,
            loss_fn=self._wrapped_loss_fn,
            get_rep_fn=self._get_rep_fn,
            fp16=fp16,
            scaler=scaler,
        )
        
        self._loss_fn = loss_fn
        self._target_embeds = None  # Will be set before cache_step
    
    def _get_rep_fn(self, model_output):
        """Extract representation from model output."""
        # LeJEPACaptioner returns (B, L, D) embeddings directly
        return model_output
    
    def _wrapped_loss_fn(self, pred_embeds):
        """
        Wrapped loss function for GradCache.
        
        GradCache calls this with cached representations.
        We use the stored target_embeds for the loss computation.
        """
        if self._target_embeds is None:
            raise RuntimeError("target_embeds not set. Call cache_step() properly.")
        
        return self._loss_fn(pred_embeds, self._target_embeds)
    
    def _default_infonce_loss(self, pred_embeds, target_embeds, temperature=0.07):
        """Default InfoNCE loss for contrastive learning."""
        B = pred_embeds.shape[0]
        device = pred_embeds.device
        
        # Pool sequence dimension: (B, L, D) -> (B, D)
        if pred_embeds.dim() == 3:
            pred_pooled = pred_embeds.mean(dim=1)
            target_pooled = target_embeds.mean(dim=1)
        else:
            pred_pooled = pred_embeds
            target_pooled = target_embeds
        
        # L2 normalize
        pred_norm = F.normalize(pred_pooled, dim=-1, p=2)
        target_norm = F.normalize(target_pooled, dim=-1, p=2)
        
        # Cosine similarity matrix
        logits = (pred_norm @ target_norm.T) / temperature
        
        # Labels: diagonal
        labels = torch.arange(B, device=device)
        
        # Bi-directional loss
        loss_pt = F.cross_entropy(logits, labels)
        loss_tp = F.cross_entropy(logits.T, labels)
        
        return (loss_pt + loss_tp) / 2
    
    def cache_step(
        self,
        images: torch.Tensor,
        target_embeds: torch.Tensor,
        no_sync_except_last: bool = False,
    ) -> torch.Tensor:
        """
        Run a gradient-cached training step.
        
        Parameters
        ----------
        images : torch.Tensor
            Batch of images (B, 3, 224, 224).
        target_embeds : torch.Tensor
            Target embeddings from LLM (B, L, D).
        no_sync_except_last : bool
            For DDP: only sync gradients on last chunk.
            
        Returns
        -------
        torch.Tensor
            Loss value (detached from graph).
        """
        # Store target embeds for loss computation
        self._target_embeds = target_embeds
        
        # GradCache handles chunking, forward, loss, and backward
        loss = self.gc(
            images,
            no_sync_except_last=no_sync_except_last,
        )
        
        # Clean up
        self._target_embeds = None
        
        return loss


class FunctionalGradCache:
    """
    Functional approach to GradCache using decorators.
    
    This is more flexible and allows custom model call patterns.
    Use this if CaptionerGradCache doesn't fit your use case.
    
    IMPORTANT: This class accepts a loss_fn that should use the MoCo queue
    from moco_queue.py for consistent loss computation with the standard
    training path. The loss_fn is provided by EmbeddingTrainer.
    
    Example
    -------
    >>> # Create with a loss function that uses MoCo queue
    >>> def loss_fn(pred, target):
    ...     return infonce_with_queue(pred, target, queue, temp)
    >>> 
    >>> fgc = FunctionalGradCache(model, loss_fn=loss_fn)
    >>> 
    >>> # Accumulate sub-batches
    >>> for images, captions in sub_batches:
    ...     fgc.accumulate(images, target_embeds)
    >>> 
    >>> # Compute loss with all accumulated embeddings
    >>> loss, metrics = fgc.compute_loss()
    >>> loss.backward()
    >>> 
    >>> # Propagate gradients through cached representations
    >>> fgc.backward_cached()
    >>> 
    >>> optimizer.step()
    >>> fgc.reset()
    
    Parameters
    ----------
    model : nn.Module
        The captioner model (encoder + connector + predictor).
    loss_fn : callable
        Loss function that takes (pred_embeds, target_embeds) and returns
        a dict with at least 'loss' (tensor). The dict should also contain
        metrics like 'infonce', 'mse', 'alignment', 'uniformity', 'num_negatives'.
        
        This function should use infonce_with_queue() from moco_queue.py
        to ensure consistent loss computation with the standard training path.
    """
    
    def __init__(
        self,
        model,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], Dict[str, Any]],
    ):
        check_gradcache_available()
        
        self.model = model
        self._loss_fn = loss_fn
        
        # Caches for accumulation
        self._pred_cache = []
        self._target_cache = []
        self._closures = []
        
        # Store last computed metrics for retrieval
        self._last_metrics: Optional[Dict[str, Any]] = None
    
    @property
    def num_accumulated(self) -> int:
        """Number of sub-batches accumulated."""
        return len(self._pred_cache)
    
    @property
    def last_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Get metrics from the last compute_loss() call.
        
        Returns dict with 'infonce', 'mse', 'alignment', 'uniformity', 'num_negatives'.
        Returns None if compute_loss() hasn't been called yet.
        """
        return self._last_metrics
    
    def reset(self):
        """Clear accumulated caches and metrics."""
        self._pred_cache = []
        self._target_cache = []
        self._closures = []
        self._last_metrics = None
    
    def accumulate(
        self,
        images: torch.Tensor,
        target_embeds: torch.Tensor,
    ):
        """
        Forward pass and cache representations.
        
        Parameters
        ----------
        images : torch.Tensor
            Sub-batch of images (B, 3, 224, 224).
        target_embeds : torch.Tensor
            Target embeddings for this sub-batch (B, L, D).
        """
        # Use the @cached decorator pattern
        # This returns (representation, closure) tuple
        pred_embeds, closure = self._cached_forward(images)
        
        self._pred_cache.append(pred_embeds)
        self._target_cache.append(target_embeds.detach())
        self._closures.append(closure)
    
    def _cached_forward(self, images):
        """
        Cached forward pass using grad_cache.functional.cached decorator.
        
        Returns representation tensor and a closure for backward.
        """
        # Manual implementation of what @cached does
        # Forward pass with gradient tracking disabled for non-leaf tensors
        with torch.enable_grad():
            pred_embeds = self.model(images)
        
        # Create leaf tensor for gradient accumulation
        pred_leaf = pred_embeds.detach().requires_grad_(True)
        
        # Closure to propagate gradients back
        def closure(cached_grad):
            if pred_embeds.grad_fn is not None:
                pred_embeds.backward(cached_grad)
        
        return pred_leaf, closure
    
    def compute_loss(self) -> torch.Tensor:
        """
        Compute loss with ALL accumulated embeddings using the provided loss_fn.
        
        The loss_fn is called with concatenated predictions and targets.
        It should return a dict with 'loss' tensor and metrics like
        'infonce', 'mse', 'alignment', 'uniformity', 'num_negatives'.
        
        Returns
        -------
        torch.Tensor
            Loss tensor (still attached to graph via leaf tensors).
            
        Note
        ----
        Call self.last_metrics after this to get the computed metrics.
        """
        if len(self._pred_cache) == 0:
            raise RuntimeError("No sub-batches accumulated. Call accumulate() first.")
        
        # Concatenate all cached representations
        all_pred = torch.cat(self._pred_cache, dim=0)
        all_target = torch.cat(self._target_cache, dim=0)
        
        # Compute loss using the provided loss function (which uses MoCo queue)
        loss_dict = self._loss_fn(all_pred, all_target)
        
        # Store metrics for retrieval
        self._last_metrics = {
            k: v for k, v in loss_dict.items() 
            if k != 'loss'  # Don't store the tensor
        }
        
        return loss_dict['loss']
    
    def backward_cached(self):
        """
        Propagate gradients through cached representations.
        
        Call this AFTER loss.backward() to complete the gradient computation.
        """
        for pred_leaf, closure in zip(self._pred_cache, self._closures):
            if pred_leaf.grad is not None:
                closure(pred_leaf.grad)

