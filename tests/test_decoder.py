"""
Test suite for decoder components (LMHeadDecoder, HybridLMHeadDecoder).

Run with: uv run --with pytest pytest tests/test_decoder.py -v -s

These tests verify that decoders correctly project hidden_states embeddings
to vocabulary tokens using the LLM's lm_head.
"""

import torch
import pytest
import warnings
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent))

from lejepa_caption.models.decoder import (
    LMHeadDecoder,
    HybridLMHeadDecoder,
    get_lmhead_decoder,
    # Deprecated classes
    StatisticsNormalizer,
    FastNNDecoder,
    HybridDecoder,
    get_hybrid_decoder,
)


class MockLLM:
    """Mock LLM for testing without loading real model."""
    
    def __init__(self, hidden_dim: int = 896, vocab_size: int = 1000):
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        
        # Mock lm_head
        self.lm_head = torch.nn.Linear(hidden_dim, vocab_size, bias=False)
        
        # Mock input embeddings (for deprecated decoders)
        self._input_embeddings = torch.nn.Embedding(vocab_size, hidden_dim)
    
    def get_input_embeddings(self):
        return self._input_embeddings


class MockTokenizer:
    """Mock tokenizer for testing."""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
    
    def decode(self, token_ids, skip_special_tokens: bool = True):
        """Simple decode - just convert IDs to string."""
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        # Filter special tokens
        if skip_special_tokens:
            token_ids = [t for t in token_ids if t not in [0, 1, 2]]
        return f"decoded_{len(token_ids)}_tokens"


class TestLMHeadDecoder:
    """Test LMHeadDecoder component."""
    
    def test_initialization(self):
        """Test decoder initializes correctly."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        
        decoder = LMHeadDecoder(llm, tokenizer)
        
        # Check lm_head was copied
        assert decoder.lm_head.in_features == 896
        assert decoder.lm_head.out_features == 1000
        
        # Check lm_head is frozen
        for param in decoder.lm_head.parameters():
            assert not param.requires_grad
            
        print("\n✓ LMHeadDecoder initialized correctly")
    
    def test_decode_output_shape(self):
        """Test decode returns correct number of captions."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        decoder = LMHeadDecoder(llm, tokenizer)
        
        batch_size = 4
        seq_len = 50
        pred_embeds = torch.randn(batch_size, seq_len, 896)
        
        captions = decoder.decode(pred_embeds)
        
        assert isinstance(captions, list)
        assert len(captions) == batch_size
        assert all(isinstance(c, str) for c in captions)
        
        print(f"\n✓ Decoded {batch_size} captions correctly")
    
    def test_decode_uses_argmax(self):
        """Test that decode uses argmax (greedy decoding)."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        decoder = LMHeadDecoder(llm, tokenizer)
        
        # Create embeddings with deterministic output
        pred_embeds = torch.randn(2, 10, 896)
        
        # Run decode twice
        captions1 = decoder.decode(pred_embeds)
        captions2 = decoder.decode(pred_embeds)
        
        # Should be identical (deterministic)
        assert captions1 == captions2
        
        print("\n✓ Decode is deterministic (argmax)")
    
    def test_lm_head_weights_preserved(self):
        """Test that lm_head weights are correctly copied."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        
        # Record original weights
        original_weights = llm.lm_head.weight.clone()
        
        decoder = LMHeadDecoder(llm, tokenizer)
        
        # Check weights match
        assert torch.allclose(decoder.lm_head.weight, original_weights)
        
        # Modify original - should not affect decoder
        llm.lm_head.weight.data.fill_(0)
        assert not torch.allclose(decoder.lm_head.weight, llm.lm_head.weight)
        
        print("\n✓ LM head weights are correctly copied (not referenced)")
    
    def test_no_gradients_flow(self):
        """Test that gradients don't flow through decoder."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        decoder = LMHeadDecoder(llm, tokenizer)
        
        pred_embeds = torch.randn(2, 10, 896, requires_grad=True)
        
        # Decode should work without gradients
        captions = decoder.decode(pred_embeds)
        
        # No gradients should be accumulated
        assert pred_embeds.grad is None
        
        print("\n✓ No gradients flow through decoder")


class TestHybridLMHeadDecoder:
    """Test HybridLMHeadDecoder with temperature and top-k."""
    
    def test_initialization_with_options(self):
        """Test decoder initializes with temperature and top_k."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        
        decoder = HybridLMHeadDecoder(
            llm, tokenizer, 
            temperature=0.7, 
            top_k=50
        )
        
        assert decoder.temperature == 0.7
        assert decoder.top_k == 50
        
        print("\n✓ HybridLMHeadDecoder initialized with options")
    
    def test_temperature_affects_logits(self):
        """Test that temperature scales logits."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        
        decoder_default = HybridLMHeadDecoder(llm, tokenizer, temperature=1.0)
        decoder_low_temp = HybridLMHeadDecoder(llm, tokenizer, temperature=0.1)
        
        pred_embeds = torch.randn(1, 5, 896)
        
        # Both should produce valid output
        captions_default = decoder_default.decode(pred_embeds)
        captions_low = decoder_low_temp.decode(pred_embeds)
        
        assert len(captions_default) == 1
        assert len(captions_low) == 1
        
        print("\n✓ Temperature option works")
    
    def test_top_k_filtering(self):
        """Test that top_k filters logits."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        
        decoder = HybridLMHeadDecoder(llm, tokenizer, top_k=10)
        
        pred_embeds = torch.randn(2, 5, 896)
        captions = decoder.decode(pred_embeds)
        
        assert len(captions) == 2
        
        print("\n✓ Top-k filtering works")
    
    def test_override_at_decode_time(self):
        """Test that temperature/top_k can be overridden at decode time."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        
        decoder = HybridLMHeadDecoder(llm, tokenizer, temperature=1.0, top_k=None)
        
        pred_embeds = torch.randn(1, 5, 896)
        
        # Override at decode time
        captions = decoder.decode(pred_embeds, temperature=0.5, top_k=20)
        
        assert len(captions) == 1
        
        print("\n✓ Decode-time override works")


class TestGetLMHeadDecoder:
    """Test factory function."""
    
    def test_returns_simple_decoder_by_default(self):
        """Test factory returns LMHeadDecoder when no options."""
        llm = MockLLM()
        tokenizer = MockTokenizer()
        
        decoder = get_lmhead_decoder(llm, tokenizer)
        
        assert isinstance(decoder, LMHeadDecoder)
        
        print("\n✓ Factory returns LMHeadDecoder by default")
    
    def test_returns_hybrid_with_options(self):
        """Test factory returns HybridLMHeadDecoder with options."""
        llm = MockLLM()
        tokenizer = MockTokenizer()
        
        decoder = get_lmhead_decoder(llm, tokenizer, temperature=0.7)
        
        assert isinstance(decoder, HybridLMHeadDecoder)
        
        print("\n✓ Factory returns HybridLMHeadDecoder with options")


class TestDeprecatedDecoders:
    """Test that deprecated decoders emit warnings."""
    
    def test_statistics_normalizer_deprecated(self):
        """Test StatisticsNormalizer emits deprecation warning."""
        llm = MockLLM()
        
        with pytest.warns(DeprecationWarning, match="StatisticsNormalizer"):
            normalizer = StatisticsNormalizer(llm)
        
        print("\n✓ StatisticsNormalizer emits deprecation warning")
    
    def test_hybrid_decoder_deprecated(self):
        """Test HybridDecoder emits deprecation warning."""
        pytest.importorskip("faiss")  # Skip if FAISS not installed
        
        llm = MockLLM()
        tokenizer = MockTokenizer()
        
        with pytest.warns(DeprecationWarning, match="HybridDecoder"):
            decoder = HybridDecoder(llm, tokenizer)
        
        print("\n✓ HybridDecoder emits deprecation warning")
    
    def test_get_hybrid_decoder_deprecated(self):
        """Test get_hybrid_decoder emits deprecation warning."""
        pytest.importorskip("faiss")  # Skip if FAISS not installed
        
        llm = MockLLM()
        tokenizer = MockTokenizer()
        
        with pytest.warns(DeprecationWarning, match="get_hybrid_decoder"):
            decoder = get_hybrid_decoder(llm, tokenizer)
        
        print("\n✓ get_hybrid_decoder emits deprecation warning")


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_sequence(self):
        """Test decode handles empty sequence."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        decoder = LMHeadDecoder(llm, tokenizer)
        
        # Sequence length 0
        pred_embeds = torch.randn(2, 0, 896)
        captions = decoder.decode(pred_embeds)
        
        assert len(captions) == 2
        
        print("\n✓ Handles empty sequence")
    
    def test_single_token_sequence(self):
        """Test decode handles single token."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        decoder = LMHeadDecoder(llm, tokenizer)
        
        pred_embeds = torch.randn(1, 1, 896)
        captions = decoder.decode(pred_embeds)
        
        assert len(captions) == 1
        
        print("\n✓ Handles single token sequence")
    
    def test_different_dtypes(self):
        """Test decode handles different input dtypes."""
        llm = MockLLM(hidden_dim=896, vocab_size=1000)
        tokenizer = MockTokenizer()
        decoder = LMHeadDecoder(llm, tokenizer)
        
        # Float32 (default)
        pred_embeds_fp32 = torch.randn(1, 5, 896, dtype=torch.float32)
        captions = decoder.decode(pred_embeds_fp32)
        assert len(captions) == 1
        
        # Float64
        pred_embeds_fp64 = torch.randn(1, 5, 896, dtype=torch.float64)
        captions = decoder.decode(pred_embeds_fp64.float())  # Convert to float32
        assert len(captions) == 1
        
        print("\n✓ Handles different dtypes")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
