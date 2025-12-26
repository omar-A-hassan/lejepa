"""
Shared fixtures for tests.

This provides common fixtures like Gemma-3 tokenizer/model
that can be shared across test sessions.

Run with: uv run --with pytest pytest tests/ -v
"""

import pytest
import torch


@pytest.fixture(scope="session")
def gemma3_tokenizer():
    """Load Gemma-3-270m tokenizer once per test session."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer
    except Exception as e:
        pytest.skip(f"Could not load Gemma-3 tokenizer: {e}")


@pytest.fixture(scope="session")
def gemma3_model():
    """Load Gemma-3-270m model once per test session."""
    try:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(
            "google/gemma-3-270m-it",
            torch_dtype=torch.bfloat16
        )
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        return model
    except Exception as e:
        pytest.skip(f"Could not load Gemma-3 model: {e}")


@pytest.fixture(scope="session")
def gemma3_embed_layer(gemma3_model):
    """Get the embedding layer from Gemma-3 model."""
    return gemma3_model.get_input_embeddings()


@pytest.fixture
def small_captioner():
    """Create a small captioner for testing."""
    from lejepa_caption.models import get_captioner
    return get_captioner(config="small", encoder_pretrained=False)


@pytest.fixture
def sample_images():
    """Generate sample images for testing."""
    return torch.randn(2, 3, 224, 224)


@pytest.fixture
def sample_captions():
    """Sample captions for testing."""
    return [
        "A cat sitting on a mat",
        "A dog running in the park"
    ]
