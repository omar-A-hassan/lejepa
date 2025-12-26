"""
MoCo-style memory queue for contrastive learning.

Stores past target embeddings to increase the number of negatives
without increasing batch size or memory consumption.

Key insight: Since targets are from a frozen LLM, we don't need
a momentum encoder. The queue is just a buffer of past embeddings.
"""

import torch


class MoCoQueue:
    """
    Memory queue for contrastive learning.
    
    Stores past target embeddings to increase negative samples.
    Simplified version (no momentum encoder) since targets are frozen.
    
    Parameters
    ----------
    dim : int
        Embedding dimension (640 for Gemma-3).
    queue_size : int
        Number of embeddings to store (default 4096).
    device : str
        Device for queue storage.
    
    Attributes
    ----------
    queue : torch.Tensor
        Circular buffer of embeddings (queue_size, dim).
    ptr : int
        Current write pointer (circular).
    is_full : bool
        Whether queue has been filled at least once.
    
    Example
    -------
    >>> queue = MoCoQueue(dim=640, queue_size=4096)
    >>> # During training:
    >>> queue.enqueue(target_embeddings)  # Add new embeddings
    >>> negatives = queue.get_negatives()  # Get all stored embeddings
    """

    def __init__(self, dim: int = 640, queue_size: int = 4096, device: str = 'cuda'):
        self.dim = dim
        self.queue_size = queue_size
        self.ptr = 0
        self.device = device

        # Initialize queue with zeros
        # Will be filled gradually during first iterations
        self.queue = torch.zeros(queue_size, dim, device=device)
        self.is_full = False  # Track if queue has been filled once

    def reset(self):
        """Reset queue to empty state."""
        self.queue.zero_()
        self.ptr = 0
        self.is_full = False

    @torch.no_grad()
    def enqueue(self, embeddings: torch.Tensor):
        """
        Add new embeddings to queue (FIFO).
        
        Parameters
        ----------
        embeddings : torch.Tensor
            New embeddings to add (B, D) - already pooled and normalized.
            
        Note
        ----
        Embeddings should be detached from computation graph to avoid
        storing gradients in the queue.
        """
        if embeddings.dim() != 2:
            raise ValueError(f"Expected 2D tensor (B, D), got {embeddings.dim()}D")
        
        B, D = embeddings.shape
        if D != self.dim:
            raise ValueError(f"Embedding dim {D} doesn't match queue dim {self.dim}")
        
        # Ensure embeddings are on correct device and detached
        embeddings = embeddings.detach().to(self.device)
        
        ptr = self.ptr

        # Handle wraparound
        if ptr + B > self.queue_size:
            # Split into two parts
            remaining = self.queue_size - ptr
            self.queue[ptr:] = embeddings[:remaining]
            self.queue[:B - remaining] = embeddings[remaining:]
            self.is_full = True  # Marked as full once we wrap
        else:
            self.queue[ptr:ptr + B] = embeddings
            if ptr + B == self.queue_size:
                self.is_full = True

        # Update pointer (circular)
        self.ptr = (ptr + B) % self.queue_size

    def get_negatives(self) -> torch.Tensor:
        """
        Get all stored embeddings for contrastive loss.
        
        Returns
        -------
        torch.Tensor or None
            Queue embeddings (queue_size, D) if full,
            or (ptr, D) if still filling.
            Returns None if queue is empty.
        """
        if self.ptr == 0 and not self.is_full:
            return None
        
        if self.is_full:
            return self.queue.clone()
        else:
            # Queue not full yet, return only filled portion
            return self.queue[:self.ptr].clone()

    def __len__(self):
        """Return number of embeddings currently in queue."""
        if self.is_full:
            return self.queue_size
        return self.ptr

    def __repr__(self):
        return (
            f"MoCoQueue(dim={self.dim}, queue_size={self.queue_size}, "
            f"current_size={len(self)}, is_full={self.is_full})"
        )


def infonce_with_queue(
    pred_norm: torch.Tensor,
    target_norm: torch.Tensor,
    queue: MoCoQueue,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    Compute InfoNCE loss with memory queue negatives.
    
    Parameters
    ----------
    pred_norm : torch.Tensor
        Normalized predicted embeddings (B, D).
    target_norm : torch.Tensor  
        Normalized target embeddings (B, D).
    queue : MoCoQueue
        Memory queue with past target embeddings.
    temperature : float
        InfoNCE temperature parameter.
        
    Returns
    -------
    torch.Tensor
        InfoNCE loss scalar.
        
    Notes
    -----
    Unlike standard InfoNCE, this computes:
    - Positive: similarity between pred[i] and target[i]
    - In-batch negatives: similarity between pred[i] and target[j!=i]
    - Queue negatives: similarity between pred[i] and queue[k]
    
    The logits are structured as:
    [pos, batch_neg_0, batch_neg_1, ..., queue_neg_0, queue_neg_1, ...]
    
    Where pos is at index 0, so labels = 0 for all samples.
    """
    B = pred_norm.shape[0]
    device = pred_norm.device
    
    # Get queue negatives
    queue_embeds = queue.get_negatives()
    
    if queue_embeds is None or len(queue_embeds) == 0:
        # Queue empty, fall back to standard InfoNCE
        logits = (pred_norm @ target_norm.T) / temperature
        labels = torch.arange(B, device=device)
        loss_pt = torch.nn.functional.cross_entropy(logits, labels)
        loss_tp = torch.nn.functional.cross_entropy(logits.T, labels)
        return (loss_pt + loss_tp) / 2
    
    # Positive similarity (diagonal of batch similarity)
    pos_sim = (pred_norm * target_norm).sum(dim=-1, keepdim=True)  # (B, 1)
    
    # Batch negatives: mask out positive (diagonal)
    batch_sim = pred_norm @ target_norm.T  # (B, B)
    # We'll include all batch similarities (including positive) and use proper label
    
    # Queue negatives
    queue_norm = torch.nn.functional.normalize(queue_embeds, dim=-1, p=2)
    queue_sim = pred_norm @ queue_norm.T  # (B, queue_size)
    
    # Combine: [batch similarities, queue negatives]
    # The positive is at the diagonal of batch_sim
    logits = torch.cat([batch_sim, queue_sim], dim=1) / temperature  # (B, B + queue_size)
    
    # Labels: index of positive in logits (diagonal = index i for sample i)
    labels = torch.arange(B, device=device)
    
    # Cross-entropy loss
    loss = torch.nn.functional.cross_entropy(logits, labels)
    
    return loss
