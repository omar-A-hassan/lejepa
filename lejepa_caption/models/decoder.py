"""
Decoding for VL-JEPA predicted embeddings.

This module provides decoders that convert predicted embeddings (in hidden_states space)
back to text tokens. The recommended decoder is LMHeadDecoder which uses the LLM's
language modeling head for mathematically correct projection.

CRITICAL: Training targets are from `llm(...).hidden_states[-1]` (contextual embeddings),
NOT from `llm.get_input_embeddings()` (static embedding table). The decoder must project
from hidden_states space to vocabulary logits using the LLM's lm_head.

Decoder Hierarchy:
    - LMHeadDecoder: RECOMMENDED - uses lm_head for correct projection
    - HybridLMHeadDecoder: LMHeadDecoder + temperature/top-k options
    - FastNNDecoder: DEPRECATED - uses wrong embedding space
    - HybridDecoder: DEPRECATED - uses wrong embedding space
"""

import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Optional

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False


class LMHeadDecoder:
    """
    Decodes predicted embeddings using the LLM's language modeling head.
    
    This is the CORRECT decoder for VL-JEPA style training where:
    - Training targets are `llm(...).hidden_states[-1]` (contextual embeddings)
    - Predictor learns to output vectors in hidden_states space
    - lm_head projects hidden_states → vocabulary logits
    
    Mathematical Justification
    --------------------------
    The LLM's lm_head is trained to satisfy:
    
        P(token_i | context) = softmax(lm_head(hidden_states[-1][i]))
    
    Since our predictor outputs vectors in hidden_states space (via InfoNCE against
    hidden_states targets), applying lm_head gives valid vocabulary logits.
    
    Computational Cost
    ------------------
    - Parameters: 0 trainable (lm_head is frozen copy)
    - Memory: O(hidden_dim * vocab_size) for lm_head weights (~160MB for Gemma-3)
    - Inference: O(B * L * hidden_dim * vocab_size) - single matmul
    - Latency: ~5-20ms per batch on CPU (faster than FAISS search)
    
    Edge AI Deployment
    ------------------
    - lm_head can be quantized to int8 (~80MB)
    - No FAISS dependency required
    - Deterministic outputs (argmax decoding)
    
    Parameters
    ----------
    llm : torch.nn.Module
        Language model with lm_head attribute.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for decoding token IDs to text.
    device : str, optional
        Device for lm_head. Defaults to 'cpu'.
        
    Attributes
    ----------
    lm_head : torch.nn.Linear
        Frozen copy of LLM's language modeling head.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer instance.
        
    Examples
    --------
    >>> decoder = LMHeadDecoder(llm, tokenizer)
    >>> pred_embeds = model(images)  # (8, 50, 896) in hidden_states space
    >>> captions = decoder.decode(pred_embeds)  # List[str], length 8
    """
    
    def __init__(self, llm, tokenizer, device: str = 'cpu'):
        self.tokenizer = tokenizer
        self.device = device
        
        # Copy lm_head weights (don't hold reference to full LLM)
        # lm_head: Linear(hidden_dim, vocab_size)
        self.lm_head = nn.Linear(
            llm.lm_head.in_features,
            llm.lm_head.out_features,
            bias=llm.lm_head.bias is not None
        )
        self.lm_head.load_state_dict(llm.lm_head.state_dict())
        self.lm_head.to(device)
        self.lm_head.eval()
        
        # Freeze
        for param in self.lm_head.parameters():
            param.requires_grad = False
            
        print(f"[LMHead Decoder] Initialized: {llm.lm_head.in_features} → {llm.lm_head.out_features}")
    
    def decode(self, pred_embeds: torch.Tensor) -> List[str]:
        """
        Decode predicted embeddings to text via lm_head projection.
        
        Parameters
        ----------
        pred_embeds : torch.Tensor
            Predicted embeddings with shape (B, L, D) where:
            - B: batch size
            - L: sequence length  
            - D: hidden dimension (must match lm_head.in_features)
            
        Returns
        -------
        List[str]
            Decoded captions, length B.
            
        Notes
        -----
        Uses argmax decoding (greedy). For sampling-based decoding,
        use HybridLMHeadDecoder with temperature > 0.
        """
        pred_embeds = pred_embeds.to(self.device)
        
        with torch.no_grad():
            # Project to vocabulary: (B, L, D) → (B, L, vocab_size)
            logits = self.lm_head(pred_embeds)
            
            # Greedy decoding
            token_ids = logits.argmax(dim=-1)  # (B, L)
        
        # Decode to text
        captions = []
        for i in range(token_ids.shape[0]):
            tokens = token_ids[i].tolist()
            caption = self.tokenizer.decode(tokens, skip_special_tokens=True)
            captions.append(caption)
            
        return captions


class HybridLMHeadDecoder:
    """
    LMHead decoder with temperature scaling and top-k filtering.
    
    Extends LMHeadDecoder with controllable decoding strategies:
    - Temperature scaling: Higher temp = more diverse, lower = more deterministic
    - Top-k filtering: Only consider top k tokens at each position
    
    Parameters
    ----------
    llm : torch.nn.Module
        Language model with lm_head attribute.
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for decoding.
    device : str, optional
        Device for lm_head.
    temperature : float, optional
        Softmax temperature. Default 1.0 (no scaling).
    top_k : int, optional
        If set, only consider top-k tokens. Default None (all tokens).
        
    Examples
    --------
    >>> decoder = HybridLMHeadDecoder(llm, tokenizer, temperature=0.7, top_k=50)
    >>> captions = decoder.decode(pred_embeds)
    """
    
    def __init__(
        self, 
        llm, 
        tokenizer, 
        device: str = 'cpu',
        temperature: float = 1.0,
        top_k: Optional[int] = None
    ):
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.top_k = top_k
        
        # Copy lm_head
        self.lm_head = nn.Linear(
            llm.lm_head.in_features,
            llm.lm_head.out_features,
            bias=llm.lm_head.bias is not None
        )
        self.lm_head.load_state_dict(llm.lm_head.state_dict())
        self.lm_head.to(device)
        self.lm_head.eval()
        
        for param in self.lm_head.parameters():
            param.requires_grad = False
            
        print(f"[Hybrid LMHead Decoder] Initialized (temp={temperature}, top_k={top_k})")
    
    def decode(
        self, 
        pred_embeds: torch.Tensor,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None
    ) -> List[str]:
        """
        Decode with optional temperature and top-k override.
        
        Parameters
        ----------
        pred_embeds : torch.Tensor
            Predicted embeddings (B, L, D).
        temperature : float, optional
            Override instance temperature.
        top_k : int, optional
            Override instance top_k.
            
        Returns
        -------
        List[str]
            Decoded captions.
        """
        temp = temperature if temperature is not None else self.temperature
        k = top_k if top_k is not None else self.top_k
        
        pred_embeds = pred_embeds.to(self.device)
        
        with torch.no_grad():
            logits = self.lm_head(pred_embeds)
            
            # Temperature scaling
            if temp != 1.0:
                logits = logits / temp
            
            # Top-k filtering
            if k is not None and k > 0:
                # Keep only top-k logits, set rest to -inf
                top_values, _ = logits.topk(k, dim=-1)
                threshold = top_values[..., -1:]  # (B, L, 1)
                logits = torch.where(
                    logits >= threshold,
                    logits,
                    torch.full_like(logits, float('-inf'))
                )
            
            token_ids = logits.argmax(dim=-1)
        
        captions = []
        for i in range(token_ids.shape[0]):
            tokens = token_ids[i].tolist()
            caption = self.tokenizer.decode(tokens, skip_special_tokens=True)
            captions.append(caption)
            
        return captions


def get_lmhead_decoder(
    llm, 
    tokenizer, 
    device: str = 'cpu',
    temperature: float = 1.0,
    top_k: Optional[int] = None
):
    """
    Factory function for creating LMHead-based decoder.
    
    This is the RECOMMENDED way to create a decoder for VL-JEPA models.
    
    Parameters
    ----------
    llm : torch.nn.Module
        Language model (only lm_head will be extracted).
    tokenizer : transformers.PreTrainedTokenizer
        Tokenizer for text decoding.
    device : str, optional
        Device for inference.
    temperature : float, optional
        Decoding temperature (1.0 = greedy).
    top_k : int, optional
        Top-k filtering (None = disabled).
        
    Returns
    -------
    LMHeadDecoder or HybridLMHeadDecoder
        Simple decoder if temperature=1.0 and top_k=None,
        otherwise hybrid decoder with options.
    """
    if temperature == 1.0 and top_k is None:
        return LMHeadDecoder(llm, tokenizer, device=device)
    else:
        return HybridLMHeadDecoder(
            llm, tokenizer, 
            device=device, 
            temperature=temperature, 
            top_k=top_k
        )


# =============================================================================
# DEPRECATED CLASSES - Kept for backward compatibility
# These use input_embeddings space which doesn't match hidden_states training
# =============================================================================


class StatisticsNormalizer:
    """
    DEPRECATED: Uses input_embeddings statistics, but training uses hidden_states.
    
    This class normalizes to the WRONG target distribution. Training targets are
    from `llm(...).hidden_states[-1]`, not `llm.get_input_embeddings()`.
    
    Use LMHeadDecoder instead, which correctly projects hidden_states → vocabulary.
    
    .. deprecated::
        Will be removed in future version. Use :class:`LMHeadDecoder` instead.
    """

    def __init__(self, llm):
        warnings.warn(
            "StatisticsNormalizer uses input_embeddings statistics, but training "
            "targets are from hidden_states[-1]. This normalization is incorrect. "
            "Use LMHeadDecoder instead.",
            DeprecationWarning,
            stacklevel=2
        )
        vocab_embeds = llm.get_input_embeddings().weight.data
        self.target_mean = vocab_embeds.mean().item()
        self.target_std = vocab_embeds.std().item()

        print(f"[Stats Normalizer] Target distribution: μ={self.target_mean:.4f}, σ={self.target_std:.4f}")

    def normalize(self, pred_embeds: torch.Tensor) -> torch.Tensor:
        """
        Apply affine transformation to match target distribution.

        Parameters
        ----------
        pred_embeds : torch.Tensor
            Predicted embeddings with shape (B, L, D) where:
            - B: batch size
            - L: sequence length
            - D: embedding dimension

        Returns
        -------
        torch.Tensor
            Normalized embeddings with same shape (B, L, D) but statistics
            matching target distribution (μ_t, σ_t).

        Notes
        -----
        Normalization is computed globally across all dimensions for stability.
        Per-sample normalization would introduce variance across batch.
        """
        pred_mean = pred_embeds.mean()
        pred_std = pred_embeds.std()

        # Standardize then rescale
        normalized = (pred_embeds - pred_mean) / (pred_std + 1e-8)
        normalized = normalized * self.target_std + self.target_mean

        return normalized


class FastNNDecoder:
    """
    DEPRECATED: Searches in input_embeddings space, but training uses hidden_states.
    
    This decoder builds a FAISS index from `llm.get_input_embeddings()`, but training
    targets are from `llm(...).hidden_states[-1]`. These are different vector spaces,
    so nearest neighbor search will return incorrect tokens.
    
    Use LMHeadDecoder instead, which correctly projects hidden_states → vocabulary
    using the LLM's language modeling head.
    
    .. deprecated::
        Will be removed in future version. Use :class:`LMHeadDecoder` instead.
    """

    def __init__(self, llm, tokenizer, use_gpu: bool = False):
        warnings.warn(
            "FastNNDecoder searches in input_embeddings space, but training targets "
            "are from hidden_states[-1]. This is the wrong vector space. "
            "Use LMHeadDecoder instead for correct decoding.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not FAISS_AVAILABLE:
            raise ImportError("FAISS required. Install: pip install faiss-cpu")

        self.tokenizer = tokenizer

        # Extract and normalize vocabulary embeddings
        vocab_embeds = llm.get_input_embeddings().weight.data
        vocab_size, dim = vocab_embeds.shape

        print(f"[NN Decoder] Building FAISS index: {vocab_size:,} tokens, dim={dim}")

        vocab_norm = F.normalize(vocab_embeds, dim=-1, p=2)
        vocab_np = vocab_norm.cpu().float().numpy()

        # Build FAISS index
        if use_gpu and torch.cuda.is_available():
            res = faiss.StandardGpuResources()
            index_flat = faiss.IndexFlatIP(dim)
            self.index = faiss.index_cpu_to_gpu(res, 0, index_flat)
        else:
            self.index = faiss.IndexFlatIP(dim)

        self.index.add(vocab_np)
        print(f"[NN Decoder] Index built: {self.index.ntotal:,} vectors")

    def decode(self, pred_embeds: torch.Tensor) -> List[str]:
        """
        Decode embeddings to text via nearest neighbor search.

        For each position in the sequence, finds the vocabulary token with
        highest cosine similarity to the predicted embedding.

        Parameters
        ----------
        pred_embeds : torch.Tensor
            Predicted embeddings with shape (B, L, D) where:
            - B: batch size
            - L: sequence length
            - D: embedding dimension

        Returns
        -------
        List[str]
            Decoded captions, length B. Each caption is the concatenation of
            nearest neighbor tokens for all L positions.

        Notes
        -----
        Decoding is position-independent: each position's token is determined
        solely by its embedding, without considering neighboring positions.
        This may produce non-fluent text but ensures computational efficiency.

        Special tokens (BOS, EOS, PAD) are stripped via skip_special_tokens=True.
        """
        B, L, D = pred_embeds.shape

        # Normalize and flatten for batch search
        pred_norm = F.normalize(pred_embeds, dim=-1, p=2)
        pred_flat = pred_norm.reshape(-1, D).cpu().float().numpy()

        # Nearest neighbor search
        _, token_ids = self.index.search(pred_flat, k=1)
        token_ids = token_ids.squeeze(-1).reshape(B, L)  # Squeeze k=1 dimension first

        # Decode token sequences
        captions = []
        for i in range(B):
            tokens = token_ids[i].tolist()
            caption = self.tokenizer.decode(tokens, skip_special_tokens=True)
            captions.append(caption)

        return captions


class HybridDecoder:
    """
    DEPRECATED: Uses input_embeddings space, but training uses hidden_states.
    
    This decoder combines StatisticsNormalizer (wrong target stats) and FastNNDecoder
    (wrong vector space). Both components are fundamentally mismatched with training.
    
    Use LMHeadDecoder or get_lmhead_decoder() instead.
    
    .. deprecated::
        Will be removed in future version. Use :class:`LMHeadDecoder` instead.
    """

    def __init__(self, llm, tokenizer, use_gpu: bool = False):
        warnings.warn(
            "HybridDecoder uses input_embeddings space, but training targets are "
            "from hidden_states[-1]. Use LMHeadDecoder or get_lmhead_decoder() instead.",
            DeprecationWarning,
            stacklevel=2
        )
        print("\n[Hybrid Decoder] Initializing...")

        self.normalizer = StatisticsNormalizer(llm)
        self.nn_decoder = FastNNDecoder(llm, tokenizer, use_gpu=use_gpu)

        print("[Hybrid Decoder] Ready!\n")

    def decode(self, pred_embeds: torch.Tensor) -> List[str]:
        """
        Decode predicted embeddings to text.

        Applies two-stage pipeline:
        1. Statistics normalization (match LLM distribution)
        2. Nearest neighbor decoding (retrieve tokens)

        Parameters
        ----------
        pred_embeds : torch.Tensor
            Predicted embeddings with shape (B, L, D).

        Returns
        -------
        List[str]
            Decoded captions, length B.

        Notes
        -----
        This method is the recommended interface for inference. Both stages
        are parameter-free and deterministic.
        """
        normalized = self.normalizer.normalize(pred_embeds)
        captions = self.nn_decoder.decode(normalized)
        return captions


def get_hybrid_decoder(llm, tokenizer, use_gpu: bool = False):
    """
    DEPRECATED: Use get_lmhead_decoder() instead.
    
    This factory creates HybridDecoder which uses the wrong embedding space.
    
    .. deprecated::
        Will be removed in future version. Use :func:`get_lmhead_decoder` instead.
    """
    warnings.warn(
        "get_hybrid_decoder is deprecated. Use get_lmhead_decoder() instead.",
        DeprecationWarning,
        stacklevel=2
    )
    return HybridDecoder(llm, tokenizer, use_gpu=use_gpu)
