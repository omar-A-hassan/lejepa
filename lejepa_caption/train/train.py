"""
Training Script for LeJEPA Edge Captioner.

Trains the model to predict text embeddings from images (VL-JEPA paradigm).
Uses MSE loss between predicted embeddings and Gemma3 target embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse
import os
from typing import Optional

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lejepa_caption.models import LeJEPACaptioner, get_captioner
from lejepa_caption.train.dataset import get_dataloader


class EmbeddingTrainer:
    """
    Trainer for VL-JEPA style embedding prediction.
    
    Key concept: We train the model to predict continuous text embeddings
    that match what Gemma3 would produce for the caption.
    """
    
    def __init__(
        self,
        model: LeJEPACaptioner,
        llm_model_name: str = "google/gemma-3-270m-it",
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
    ):
        self.model = model.to(device)
        self.device = device
        
        # Load Gemma3 for target embeddings (frozen)
        self._load_llm(llm_model_name)
        
        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
        
        # Loss function
        self.criterion = nn.MSELoss()
        
    def _load_llm(self, model_name: str):
        """Load frozen LLM for target embedding extraction."""
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            
            print(f"Loading {model_name} for target embeddings...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Only load embedding layer, not full model (saves memory)
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
            )
            self.llm.eval()
            
            # Freeze LLM
            for param in self.llm.parameters():
                param.requires_grad = False
                
            print(f"  LLM loaded and frozen")
            
        except Exception as e:
            print(f"Warning: Could not load LLM: {e}")
            print("  Using random target embeddings for testing")
            self.tokenizer = None
            self.llm = None
    
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
        if self.llm is None:
            # Fallback: random embeddings for testing
            return torch.randn(len(captions), max_len, 1024, device=self.device)
        
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
        self.optimizer.zero_grad()
        
        images = images.to(self.device)
        
        # Forward pass - predict embeddings
        pred_embeds = self.model(images)  # (B, max_len, 1024)
        
        # Get target embeddings from LLM
        target_embeds = self.get_target_embeddings(
            captions, 
            max_len=pred_embeds.size(1)
        )
        
        # MSE loss
        loss = self.criterion(pred_embeds, target_embeds)
        
        # Backward
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        
        self.optimizer.step()
        
        return {"loss": loss.item()}
    
    @torch.no_grad()
    def validate(self, dataloader) -> dict:
        """Run validation."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        for images, captions in tqdm(dataloader, desc="Validating"):
            images = images.to(self.device)
            
            pred_embeds = self.model(images)
            target_embeds = self.get_target_embeddings(
                captions,
                max_len=pred_embeds.size(1)
            )
            
            loss = self.criterion(pred_embeds, target_embeds)
            total_loss += loss.item()
            num_batches += 1
        
        return {"val_loss": total_loss / num_batches}
    
    def save_checkpoint(self, path: str, epoch: int):
        """Save model checkpoint."""
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)
        print(f"  Saved checkpoint: {path}")


def train(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    model_config: str = "small",
    max_samples: Optional[int] = None,
    save_dir: str = "checkpoints",
    device: str = "cuda",
):
    """
    Main training function.
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size
        lr: Learning rate
        model_config: 'tiny', 'small', or 'base'
        max_samples: Limit samples for debugging
        save_dir: Checkpoint save directory
        device: 'cuda' or 'cpu'
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Model
    print(f"Creating {model_config} model...")
    model = get_captioner(model_config)
    print(f"  Total params: {model.num_parameters['total'] / 1e6:.1f}M")
    
    # Data
    print("Loading datasets...")
    train_loader = get_dataloader(
        split="train",
        batch_size=batch_size,
        max_samples=max_samples,
    )
    val_loader = get_dataloader(
        split="validation",
        batch_size=batch_size,
        max_samples=max_samples // 10 if max_samples else None,
    )
    
    # Trainer
    trainer = EmbeddingTrainer(
        model=model,
        device=device,
        lr=lr,
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=epochs * len(train_loader),
        eta_min=lr / 100,
    )
    
    # Training loop
    print(f"\nTraining for {epochs} epochs...")
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        
        # Train
        epoch_loss = 0
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, captions) in enumerate(pbar):
            metrics = trainer.train_step(images, captions)
            epoch_loss += metrics["loss"]
            scheduler.step()
            
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"  Train Loss: {avg_loss:.4f}")
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        print(f"  Val Loss: {val_metrics['val_loss']:.4f}")
        
        # Save checkpoint
        trainer.save_checkpoint(
            os.path.join(save_dir, f"epoch_{epoch + 1}.pt"),
            epoch + 1,
        )
    
    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeJEPA Captioner")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_config=args.model,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        device=args.device,
    )
