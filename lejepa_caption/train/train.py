"""
Training utilities for LeJEPA Edge Captioner.

Notebook-friendly: assumes the notebook already created the model, tokenizer,
and LLM; only handles the training loop with pure MSE loss.

Following LeJEPA philosophy: Simple, principled, no heuristics.
"""

import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lejepa_caption.models import LeJEPACaptioner, get_captioner


def get_best_device() -> torch.device:
    """Select best available device: CUDA -> MPS -> CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EmbeddingTrainer:
    """
    Trainer for VL-JEPA style embedding prediction.

    Key concept: We train the model to predict continuous text embeddings
    that match what Gemma3 would produce for the caption.

    Uses pure MSE loss - supervised regression to target embeddings.
    No heuristics (no SIGReg, no gradient clipping) following LeJEPA philosophy.
    """

    def __init__(
        self,
        model: LeJEPACaptioner,
        tokenizer,
        llm,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        use_amp: bool = False,
    ):
        if tokenizer is None or llm is None:
            raise ValueError("Tokenizer and LLM must be provided (preloaded in notebook).")

        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp and torch.cuda.is_available())

        # Preloaded tokenizer/LLM
        self.tokenizer = tokenizer
        self.llm = llm

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    
    def get_target_embeddings(
        self, 
        captions: list, 
        max_len: int = 50
    ) -> torch.Tensor:
        """
        Get target text embeddings from Gemma3.
        
        Args:
            captions: List of caption strings
            max_len: Max sequence length
            
        Returns:
            Target embeddings (B, max_len, llm_dim)
        """
        # Tokenize
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)
        
        # Get embeddings from LLM embedding layer
        with torch.no_grad():
            embeddings = self.llm.get_input_embeddings()(tokens.input_ids)
        
        return embeddings.float()
    
    def train_step(
        self,
        images: torch.Tensor,
        captions: list
    ) -> dict:
        """
        Single training step.

        Args:
            images: Batch of images (B, 3, 224, 224)
            captions: List of caption strings

        Returns:
            Dict with loss value
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        images = images.to(self.device)

        # Forward pass - predict embeddings
        with autocast(enabled=self.use_amp):
            pred_embeds = self.model(images)  # (B, 50, 640)

            # Get target embeddings from frozen Gemma-3
            target_embeds = self.get_target_embeddings(
                captions,
                max_len=pred_embeds.size(1)
            )
            target_embeds = target_embeds.to(pred_embeds.dtype)

            # Pure MSE loss - supervised regression
            loss = F.mse_loss(pred_embeds, target_embeds)

        # Backward (no gradient clipping - MSE is stable)
        if self.scaler is not None and self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        return {"loss": loss.item()}
    
    def validate(self, dataloader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        with torch.inference_mode():
            for images, captions in tqdm(dataloader, desc="Validating"):
                images = images.to(self.device)

                pred_embeds = self.model(images)
                target_embeds = self.get_target_embeddings(
                    captions,
                    max_len=pred_embeds.size(1)
                )

                target_embeds = target_embeds.to(pred_embeds.dtype)
                loss = F.mse_loss(pred_embeds, target_embeds)

                total_loss += loss.item()
                num_batches += 1

        denom = max(1, num_batches)
        return {"val_loss": total_loss / denom}
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"  Saved checkpoint: {path}")
def train_with_loader(
    model: LeJEPACaptioner,
    train_loader,
    tokenizer,
    llm,
    device: str = "cuda",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    use_amp: bool = False,
    epochs: int = 3,
):
    """
    Notebook-friendly training helper.

    Uses pure MSE loss for supervised embedding prediction.
    No heuristics (no SIGReg, no gradient clipping).

    Args:
        model: LeJEPACaptioner instance
        train_loader: DataLoader with (images, captions)
        tokenizer: Pretrained tokenizer (from notebook)
        llm: Pretrained LLM (from notebook, frozen)
        device: 'cuda' or 'cpu'
        lr: Learning rate
        weight_decay: AdamW weight decay
        use_amp: Use automatic mixed precision
        epochs: Number of training epochs

    Returns:
        EmbeddingTrainer instance (with trained model)
    """
    device = torch.device(device)
    trainer = EmbeddingTrainer(
        model,
        tokenizer=tokenizer,
        llm=llm,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        use_amp=use_amp,
    )

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        for images, captions in pbar:
            stats = trainer.train_step(images, captions)
            pbar.set_postfix({"loss": f"{stats['loss']:.4f}"})

    return trainer
