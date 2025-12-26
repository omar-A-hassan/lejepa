"""
Test suite for model components (encoder, connector, predictor, captioner).

Run with: uv run --with pytest pytest tests/test_model_components.py -v -s
"""

import torch
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lejepa_caption.models import (
    VisionEncoder,
    CAbstractor,
    EmbeddingPredictor,
    LeJEPACaptioner,
    get_captioner,
)


class TestVisionEncoder:
    """Test Vision encoder component."""

    def test_encoder_output_shape(self):
        """Test encoder produces correct output shape."""
        encoder = VisionEncoder(pretrained=False)
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        output = encoder(images)

        # ViT-Tiny: 196 patches (14x14), 192 dim
        expected_shape = (batch_size, 196, 192)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"
        print(f"\nEncoder output shape: {output.shape} (correct)")

    def test_encoder_no_nans(self):
        """Test encoder doesn't produce NaNs."""
        encoder = VisionEncoder(pretrained=False)
        images = torch.randn(2, 3, 224, 224)

        output = encoder(images)

        assert not torch.isnan(output).any(), "Encoder produced NaN values"
        assert not torch.isinf(output).any(), "Encoder produced Inf values"
        print("\nEncoder produces valid values (no NaN/Inf)")

    def test_encoder_parameters_trainable(self):
        """Test encoder parameters are trainable."""
        encoder = VisionEncoder(pretrained=False)

        trainable_params = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in encoder.parameters())

        assert trainable_params > 0, "No trainable parameters in encoder"
        assert trainable_params == total_params, "Some parameters are frozen"

        print(f"\nEncoder trainable params: {trainable_params:,}")


class TestCAbstractor:
    """Test C-Abstractor connector component."""

    def test_connector_output_shape(self):
        """Test connector produces correct output shape."""
        connector = CAbstractor(enc_dim=192, llm_dim=640)
        batch_size = 2

        # Input: 196 patches from ViT
        enc_output = torch.randn(batch_size, 196, 192)

        output = connector(enc_output)

        # Expected: 49 tokens (14x14 -> 7x7), 640 dim
        expected_shape = (batch_size, 49, 640)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"
        print(f"\nConnector output shape: {output.shape} (correct)")

    def test_connector_downsampling(self):
        """Test connector downsamples correctly (196 -> 49)."""
        connector = CAbstractor(enc_dim=192, llm_dim=640)
        enc_output = torch.randn(2, 196, 192)

        output = connector(enc_output)

        input_tokens = 196
        output_tokens = output.shape[1]

        assert output_tokens == 49, f"Expected 49 tokens, got {output_tokens}"
        assert output_tokens < input_tokens, "Connector should downsample"

        print(f"\nConnector downsampling: {input_tokens} -> {output_tokens} tokens")

    def test_connector_no_nans(self):
        """Test connector doesn't produce NaNs."""
        connector = CAbstractor(enc_dim=192, llm_dim=640)
        enc_output = torch.randn(2, 196, 192)

        output = connector(enc_output)

        assert not torch.isnan(output).any(), "Connector produced NaN values"
        assert not torch.isinf(output).any(), "Connector produced Inf values"
        print("\nConnector produces valid values (no NaN/Inf)")


class TestEmbeddingPredictor:
    """Test Embedding predictor component."""

    def test_predictor_output_shape(self):
        """Test predictor produces correct output shape."""
        predictor = EmbeddingPredictor(
            dim=640,
            num_layers=4,
            num_heads=8,
            max_len=50
        )
        batch_size = 2

        # Input: 49 tokens from connector
        connector_output = torch.randn(batch_size, 49, 640)

        output = predictor(connector_output)

        # Expected: 50 tokens (max_caption_len), 640 dim
        expected_shape = (batch_size, 50, 640)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"
        print(f"\nPredictor output shape: {output.shape} (correct)")

    def test_predictor_sequence_expansion(self):
        """Test predictor expands sequence (49 -> 50)."""
        predictor = EmbeddingPredictor(dim=640, max_len=50)
        connector_output = torch.randn(2, 49, 640)

        output = predictor(connector_output)

        input_tokens = 49
        output_tokens = output.shape[1]

        assert output_tokens == 50, f"Expected 50 tokens, got {output_tokens}"

        print(f"\nPredictor sequence: {input_tokens} -> {output_tokens} tokens")

    def test_predictor_no_nans(self):
        """Test predictor doesn't produce NaNs."""
        predictor = EmbeddingPredictor(dim=640)
        connector_output = torch.randn(2, 49, 640)

        output = predictor(connector_output)

        assert not torch.isnan(output).any(), "Predictor produced NaN values"
        assert not torch.isinf(output).any(), "Predictor produced Inf values"
        print("\nPredictor produces valid values (no NaN/Inf)")


class TestLeJEPACaptioner:
    """Test full LeJEPA captioner model."""

    def test_full_model_forward_pass(self):
        """Test full model end-to-end forward pass."""
        model = LeJEPACaptioner(
            enc_dim=192,
            llm_dim=640,
            encoder_pretrained=False
        )
        batch_size = 2
        images = torch.randn(batch_size, 3, 224, 224)

        output = model(images)

        # Expected: (batch, 50, 640)
        expected_shape = (batch_size, 50, 640)
        assert output.shape == expected_shape, \
            f"Expected {expected_shape}, got {output.shape}"
        print(f"\nFull model output shape: {output.shape} (correct)")

    def test_full_model_no_nans(self):
        """Test full model doesn't produce NaNs."""
        model = LeJEPACaptioner(enc_dim=192, llm_dim=640, encoder_pretrained=False)
        images = torch.randn(2, 3, 224, 224)

        output = model(images)

        assert not torch.isnan(output).any(), "Model produced NaN values"
        assert not torch.isinf(output).any(), "Model produced Inf values"
        print("\nFull model produces valid values (no NaN/Inf)")

    def test_model_parameter_count(self):
        """Test model has expected number of parameters."""
        model = LeJEPACaptioner(enc_dim=192, llm_dim=640, encoder_pretrained=False)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # Expected: ~33.4M params
        # Encoder: 5.5M, Connector: 1.1M, Predictor: 26.7M
        expected_min = 30_000_000
        expected_max = 40_000_000

        assert expected_min < total_params < expected_max, \
            f"Expected ~33M params, got {total_params:,}"

        assert trainable_params == total_params, "Some parameters are frozen unexpectedly"

        print(f"\nModel parameter count:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Expected range: 30M-40M (within range)")

    def test_model_components_connected(self):
        """Test that all model components are properly connected."""
        model = LeJEPACaptioner(enc_dim=192, llm_dim=640, encoder_pretrained=False)

        # Check components exist
        assert hasattr(model, 'encoder'), "Model missing encoder"
        assert hasattr(model, 'connector'), "Model missing connector"
        assert hasattr(model, 'predictor'), "Model missing predictor"

        # Check components are modules
        assert isinstance(model.encoder, torch.nn.Module), "Encoder not a Module"
        assert isinstance(model.connector, torch.nn.Module), "Connector not a Module"
        assert isinstance(model.predictor, torch.nn.Module), "Predictor not a Module"

        print("\nModel components properly connected:")
        print(f"  Encoder: {type(model.encoder).__name__}")
        print(f"  Connector: {type(model.connector).__name__}")
        print(f"  Predictor: {type(model.predictor).__name__}")

    def test_get_captioner_helper(self):
        """Test get_captioner helper function."""
        model = get_captioner(config="small", encoder_pretrained=False)

        assert isinstance(model, LeJEPACaptioner), \
            "get_captioner should return LeJEPACaptioner instance"

        output = model(torch.randn(1, 3, 224, 224))
        assert output.shape == (1, 50, 640), \
            f"get_captioner model has wrong output shape: {output.shape}"

        print("\nget_captioner() works correctly")


class TestModelDimensions:
    """Test dimensional consistency across model components."""

    def test_dimension_flow(self):
        """Test that dimensions flow correctly through the model."""
        batch_size = 2

        # Create components
        encoder = VisionEncoder(pretrained=False)
        connector = CAbstractor(enc_dim=192, llm_dim=640)
        predictor = EmbeddingPredictor(dim=640)

        # Forward pass through each component
        images = torch.randn(batch_size, 3, 224, 224)

        enc_out = encoder(images)
        print(f"\nDimension flow:")
        print(f"  Input: {images.shape}")
        print(f"  After encoder: {enc_out.shape}")

        conn_out = connector(enc_out)
        print(f"  After connector: {conn_out.shape}")

        pred_out = predictor(conn_out)
        print(f"  After predictor: {pred_out.shape}")

        # Verify each step
        assert enc_out.shape == (batch_size, 196, 192), "Encoder output shape wrong"
        assert conn_out.shape == (batch_size, 49, 640), "Connector output shape wrong"
        assert pred_out.shape == (batch_size, 50, 640), "Predictor output shape wrong"

        print("\nAll dimensions correct!")

    def test_batch_size_independence(self):
        """Test model works with different batch sizes."""
        model = LeJEPACaptioner(enc_dim=192, llm_dim=640, encoder_pretrained=False)

        batch_sizes = [1, 2, 4, 8]

        print("\nTesting different batch sizes:")
        for bs in batch_sizes:
            images = torch.randn(bs, 3, 224, 224)
            output = model(images)

            expected_shape = (bs, 50, 640)
            assert output.shape == expected_shape, \
                f"Batch size {bs}: expected {expected_shape}, got {output.shape}"

            print(f"  Batch size {bs}: {output.shape} (correct)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
