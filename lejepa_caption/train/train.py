"""
Training utilities for LeJEPA Edge Captioner.

Notebook-friendly: assumes the notebook already created the model, tokenizer,
and LLM; only handles the training loop (cosine + SIGReg).
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

from lejepa_caption.models import LeJEPACaptioner, SIGRegLoss, get_captioner


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
    """
    
    def __init__(
        self,
        model: LeJEPACaptioner,
        tokenizer,
        llm,
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lambda_sigreg: float = 0.05,
        align_weight: float = 1.0,
        sigreg_n_points: int = 17,
        sigreg_num_slices: int = 128,
        use_amp: bool = False,
    ):
        if tokenizer is None or llm is None:
            raise ValueError("Tokenizer and LLM must be provided (preloaded in notebook).")

        self.model = model.to(device)
        self.device = device
        self.lambda_sigreg = lambda_sigreg
        self.align_weight = align_weight
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

        # Loss functions
        self.sigreg = SIGRegLoss(
            n_points=sigreg_n_points,
            num_slices=sigreg_num_slices,
            reduction="mean",
        ).to(device)
    
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
            Dict with loss values
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        
        images = images.to(self.device)
        
        # Forward pass - predict embeddings
        with autocast(enabled=self.use_amp):
            pred_embeds = self.model(images)  # (B, max_len, 1024)

            # Get target embeddings from LLM
            target_embeds = self.get_target_embeddings(
                captions, 
                max_len=pred_embeds.size(1)
            )
            target_embeds = target_embeds.to(pred_embeds.dtype)

            # Cosine alignment + SIGReg regularization
            align_loss = 1.0 - F.cosine_similarity(pred_embeds, target_embeds, dim=-1, eps=1e-8).mean()
            sigreg_loss = self.sigreg(pred_embeds)
            loss = self.align_weight * align_loss + self.lambda_sigreg * sigreg_loss
        
        # Backward
        if self.scaler is not None and self.use_amp:
            self.scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        
        return {
            "loss": loss.item(),
            "align_loss": align_loss.item(),
            "sigreg_loss": sigreg_loss.item(),
        }
    
    def validate(self, dataloader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        total_align = 0
        total_sigreg = 0
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
                align_loss = 1.0 - F.cosine_similarity(pred_embeds, target_embeds, dim=-1, eps=1e-8).mean()
                sigreg_loss = self.sigreg(pred_embeds)
                loss = self.align_weight * align_loss + self.lambda_sigreg * sigreg_loss

                total_loss += loss.item()
                total_align += align_loss.item()
                total_sigreg += sigreg_loss.item()
                num_batches += 1

        denom = max(1, num_batches)
        return {
            "val_loss": total_loss / denom,
            "val_align": total_align / denom,
            "val_sigreg": total_sigreg / denom,
        }
    
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
    lambda_sigreg: float = 0.05,
    align_weight: float = 1.0,
    sigreg_n_points: int = 17,
    sigreg_num_slices: int = 128,
    use_amp: bool = False,
    epochs: int = 3,
):
    """
    Notebook-friendly training helper. Requires prebuilt model, tokenizer, llm, and dataloader.
    """
    device = torch.device(device)
    trainer = EmbeddingTrainer(
        model,
        tokenizer=tokenizer,
        llm=llm,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        lambda_sigreg=lambda_sigreg,
        align_weight=align_weight,
        sigreg_n_points=sigreg_n_points,
        sigreg_num_slices=sigreg_num_slices,
        use_amp=use_amp,
    )

    for epoch in range(1, epochs + 1):
        pbar = tqdm(train_loader, desc=f"Train {epoch}/{epochs}")
        for images, captions in pbar:
            stats = trainer.train_step(images, captions)
            pbar.set_postfix({
                "loss": f"{stats['loss']:.4f}",
                "align": f"{stats['align_loss']:.4f}",
                "sigreg": f"{stats['sigreg_loss']:.4f}",
            })

    return trainer
