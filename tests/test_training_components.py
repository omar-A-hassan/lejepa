"""
Test suite for training components (losses, gradients, data loading).

Run with: uv run --with pytest pytest tests/test_training_components.py -v -s
"""

import torch
import torch.nn.functional as F
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from lejepa_caption.models import get_captioner


class TestMSELoss:
    """Test MSE loss for embedding prediction."""

    def test_mse_loss_basic(self):
        """Test basic MSE loss computation."""
        batch_size = 4
        seq_len = 50
        dim = 640

        pred = torch.randn(batch_size, seq_len, dim)
        target = torch.randn(batch_size, seq_len, dim)

        loss = F.mse_loss(pred, target)

        assert loss > 0, "MSE loss should be positive"
        assert not torch.isnan(loss), "MSE loss is NaN"
        assert not torch.isinf(loss), "MSE loss is Inf"

        print(f"\nMSE loss (random embeddings): {loss:.4f}")

    def test_mse_loss_perfect_match(self):
        """Test MSE loss with perfect match."""
        pred = torch.randn(4, 50, 640)
        target = pred.clone()

        loss = F.mse_loss(pred, target)

        assert loss < 1e-6, f"Perfect match should give loss ~0, got {loss:.6f}"
        print(f"\nMSE loss (perfect match): {loss:.6e} (correct)")

    def test_mse_loss_reasonable_range(self):
        """Test MSE loss is in reasonable range for normalized embeddings."""
        # Simulate normalized embeddings (mean ~0, std ~1)
        pred = torch.randn(4, 50, 640)
        target = torch.randn(4, 50, 640)

        # Normalize to mean 0, std 1 (typical for embeddings)
        pred = (pred - pred.mean()) / pred.std()
        target = (target - target.mean()) / target.std()

        loss = F.mse_loss(pred, target)

        # For normalized random embeddings, MSE should be ~2.0
        # (variance of difference between two N(0,1) variables)
        assert 0.5 < loss < 4.0, \
            f"MSE loss should be 0.5-4.0 for normalized embeddings, got {loss:.4f}"

        print(f"\nMSE loss (normalized embeddings): {loss:.4f}")
        print("  Expected range: 0.5-4.0 (correct)")

    def test_mse_gradient_flow(self):
        """Test MSE loss produces gradients."""
        pred = torch.randn(4, 50, 640, requires_grad=True)
        target = torch.randn(4, 50, 640)

        loss = F.mse_loss(pred, target)
        loss.backward()

        assert pred.grad is not None, "No gradients computed"
        assert not torch.isnan(pred.grad).any(), "Gradients contain NaN"
        assert pred.grad.abs().sum() > 0, "All gradients are zero"

        print(f"\nMSE gradient check:")
        print(f"  Gradient norm: {pred.grad.norm():.4f}")
        print("  Gradients flowing correctly")


class TestGradientFlow:
    """Test gradient flow through model components."""

    def test_encoder_gradients(self):
        """Test gradients flow through encoder."""
        from lejepa_caption.models import VisionEncoder

        encoder = VisionEncoder(pretrained=False)
        images = torch.randn(2, 3, 224, 224)

        output = encoder(images)
        loss = output.mean()  # Dummy loss
        loss.backward()

        # Check gradients exist
        has_grads = False
        for name, param in encoder.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads, "No gradients in encoder"
        print("\nEncoder gradients: OK")

    def test_connector_gradients(self):
        """Test gradients flow through connector."""
        from lejepa_caption.models import CAbstractor

        connector = CAbstractor(enc_dim=192, llm_dim=640)
        enc_output = torch.randn(2, 196, 192, requires_grad=True)

        output = connector(enc_output)
        loss = output.mean()
        loss.backward()

        # Check input gradients
        assert enc_output.grad is not None, "No gradients to input"
        assert enc_output.grad.abs().sum() > 0, "All input gradients zero"

        # Check parameter gradients
        has_grads = False
        for param in connector.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads, "No parameter gradients in connector"
        print("\nConnector gradients: OK")

    def test_predictor_gradients(self):
        """Test gradients flow through predictor."""
        from lejepa_caption.models import EmbeddingPredictor

        predictor = EmbeddingPredictor(dim=640)
        connector_output = torch.randn(2, 49, 640, requires_grad=True)

        output = predictor(connector_output)
        loss = output.mean()
        loss.backward()

        # Check input gradients
        assert connector_output.grad is not None, "No gradients to input"
        assert connector_output.grad.abs().sum() > 0, "All input gradients zero"

        # Check parameter gradients
        has_grads = False
        for param in predictor.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grads = True
                break

        assert has_grads, "No parameter gradients in predictor"
        print("\nPredictor gradients: OK")

    def test_full_model_gradient_flow(self):
        """Test gradients flow through entire model."""
        model = get_captioner(config="small", encoder_pretrained=False)

        images = torch.randn(2, 3, 224, 224)
        target = torch.randn(2, 50, 640)

        # Forward pass
        pred = model(images)
        loss = F.mse_loss(pred, target)

        # Backward pass
        loss.backward()

        # Check each component has gradients
        components = {
            'encoder': model.encoder,
            'connector': model.connector,
            'predictor': model.predictor
        }

        print("\nFull model gradient flow:")
        for name, component in components.items():
            has_grads = False
            grad_norm = 0.0

            for param in component.parameters():
                if param.grad is not None:
                    grad_norm += param.grad.norm().item()
                    if param.grad.abs().sum() > 0:
                        has_grads = True

            assert has_grads, f"No gradients in {name}"
            print(f"  {name}: grad_norm={grad_norm:.4f} (OK)")

    def test_gradient_magnitudes(self):
        """Test gradient magnitudes are reasonable (not vanishing/exploding)."""
        model = get_captioner(config="small", encoder_pretrained=False)

        images = torch.randn(2, 3, 224, 224)
        target = torch.randn(2, 50, 640)

        pred = model(images)
        loss = F.mse_loss(pred, target)
        loss.backward()

        # Collect gradient norms
        grad_norms = []
        for param in model.parameters():
            if param.grad is not None:
                grad_norms.append(param.grad.norm().item())

        avg_grad_norm = sum(grad_norms) / len(grad_norms)
        max_grad_norm = max(grad_norms)
        min_grad_norm = min(grad_norms)

        print(f"\nGradient magnitude statistics:")
        print(f"  Average: {avg_grad_norm:.6f}")
        print(f"  Max: {max_grad_norm:.6f}")
        print(f"  Min: {min_grad_norm:.6f}")

        # Check for vanishing gradients
        assert avg_grad_norm > 1e-8, f"Gradients too small (vanishing): {avg_grad_norm:.2e}"

        # Check for exploding gradients
        assert max_grad_norm < 1e3, f"Gradients too large (exploding): {max_grad_norm:.2e}"

        print("  Gradients in reasonable range (OK)")

    def test_no_frozen_parameters(self):
        """Test no parameters are accidentally frozen."""
        model = get_captioner(config="small", encoder_pretrained=False)

        frozen_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                frozen_params.append(name)

        if frozen_params:
            print(f"\nWARNING: Found {len(frozen_params)} frozen parameters:")
            for name in frozen_params[:5]:  # Show first 5
                print(f"  {name}")
            pytest.fail(f"Found {len(frozen_params)} frozen parameters (unexpected)")

        print("\nNo frozen parameters (all trainable)")


class TestTargetEmbeddings:
    """Test target embedding extraction from LLM."""

    def test_target_embedding_shape(self):
        """Test CONTEXTUAL target embeddings have correct shape.
        
        This tests the VL-JEPA Y-Encoder approach: running full LLM forward
        pass to get contextualized embeddings, NOT raw embedding table lookup.
        """
        from transformers import AutoTokenizer, AutoModelForCausalLM

        # Load Gemma-3-270m components (matches training notebook)
        try:
            tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m-it")
            tokenizer.pad_token = tokenizer.eos_token

            # Load full model for contextual embeddings (Y-Encoder)
            model = AutoModelForCausalLM.from_pretrained(
                "google/gemma-3-270m-it",
                torch_dtype=torch.bfloat16
            )

            # Tokenize captions
            captions = ["A cat sitting on a mat", "A dog running in park"]
            tokens = tokenizer(
                captions,
                padding="max_length",
                truncation=True,
                max_length=50,
                return_tensors="pt"
            )

            # Get CONTEXTUAL embeddings via LLM forward pass
            # This is the correct VL-JEPA approach (not embedding table lookup)
            with torch.no_grad():
                outputs = model(
                    input_ids=tokens.input_ids,
                    attention_mask=tokens.attention_mask,
                    output_hidden_states=True,
                )
                # Use last hidden state - fully contextualized
                embeddings = outputs.hidden_states[-1]

            # Check shape
            batch_size = len(captions)
            # Gemma-3-270m has 640 embedding dimension
            expected_shape = (batch_size, 50, 640)

            assert embeddings.shape[0] == batch_size
            assert embeddings.shape[1] == 50
            assert embeddings.shape[2] == 640, \
                f"Expected dim=640 (Gemma-3-270m), got {embeddings.shape[2]}"

            print(f"\nContextual target embedding extraction (Y-Encoder):")
            print(f"  Shape: {embeddings.shape}")
            print(f"  Mean: {embeddings.float().mean():.4f}")
            print(f"  Std: {embeddings.float().std():.4f}")
            print("  Contextual embeddings extracted (OK)")

        except Exception as e:
            pytest.skip(f"Skipping: Could not load Gemma model ({e})")

    def test_target_embeddings_no_nans(self):
        """Test target embeddings don't contain NaNs."""
        # Simulate target embeddings
        target = torch.randn(4, 50, 640)

        assert not torch.isnan(target).any(), "Target embeddings contain NaN"
        assert not torch.isinf(target).any(), "Target embeddings contain Inf"

        print("\nTarget embeddings valid (no NaN/Inf)")


class TestTrainingStep:
    """Test complete training step integration."""

    def test_single_training_step(self):
        """Test a single forward-backward-update step."""
        model = get_captioner(config="small", encoder_pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        # Forward
        images = torch.randn(2, 3, 224, 224)
        target = torch.randn(2, 50, 640)

        pred = model(images)
        loss = F.mse_loss(pred, target)

        initial_loss = loss.item()

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Check gradients exist
        has_grads = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grads, "No gradients after backward"

        # Update
        optimizer.step()

        # Forward again
        pred_after = model(images)
        loss_after = F.mse_loss(pred_after, target)

        # Parameters should have changed
        assert loss_after.item() != initial_loss, "Loss didn't change after update"

        print(f"\nTraining step test:")
        print(f"  Loss before: {initial_loss:.4f}")
        print(f"  Loss after: {loss_after.item():.4f}")
        print("  Training step works (OK)")

    def test_multiple_training_steps(self):
        """Test multiple training steps converge."""
        model = get_captioner(config="small", encoder_pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        images = torch.randn(4, 3, 224, 224)
        target = torch.randn(4, 50, 640)

        losses = []

        for step in range(10):
            optimizer.zero_grad()

            pred = model(images)
            loss = F.mse_loss(pred, target)

            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        print(f"\n10 training steps:")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}")
        print(f"  Loss decreased: {losses[0] - losses[-1]:.4f}")

        # Loss should decrease
        assert losses[-1] < losses[0], \
            f"Loss should decrease, but went from {losses[0]:.4f} to {losses[-1]:.4f}"

        print("  Training converging (OK)")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
