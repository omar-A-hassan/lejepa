"""
Training Script for LeJEPA Edge Captioner.

Predicts text embeddings from images (VL-JEPA paradigm) with cosine alignment
plus SIGReg regularization for dispersion.
"""

import argparse
import logging
import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Add parent to path for imports
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lejepa_caption.models import LeJEPACaptioner, SIGRegLoss, get_captioner
from lejepa_caption.train.dataset import get_dataloader


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
        llm_model_name: str = "google/gemma-3-270m-it",
        device: str = "cuda",
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        lambda_sigreg: float = 0.05,
        align_weight: float = 1.0,
        sigreg_n_points: int = 17,
        sigreg_num_slices: int = 128,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.device = device
        self.lambda_sigreg = lambda_sigreg
        self.align_weight = align_weight
        self.use_amp = use_amp
        self.scaler = GradScaler(enabled=use_amp and torch.cuda.is_available())
        
        # Load Gemma3 for target embeddings (frozen)
        self._load_llm(llm_model_name)
        
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


def train(
    epochs: int = 10,
    batch_size: int = 32,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    model_config: str = "small",
    max_samples: Optional[int] = None,
    save_dir: str = "checkpoints",
    device: str = "cuda",
    lambda_sigreg: float = 0.05,
    align_weight: float = 1.0,
    sigreg_n_points: int = 17,
    sigreg_num_slices: int = 128,
    use_amp: bool = False,
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
        device: 'cuda', 'mps', 'cpu', or 'auto'
        lambda_sigreg: Weight for SIGReg regularizer
        align_weight: Weight for cosine alignment term
        sigreg_n_points: Integration points for Epps-Pulley
        sigreg_num_slices: Number of random slices for SIGReg
        use_amp: Enable autocast/GradScaler mixed precision
    """
    os.makedirs(save_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    logging.getLogger("datasets").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    # Model
    logger.info(f"Creating {model_config} model...")
    model = get_captioner(model_config)
    logger.info(f"Total params: {model.num_parameters['total'] / 1e6:.1f}M")
    
    # Data
    logger.info("Loading datasets...")
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

    # Device
    device_resolved = device if device != "auto" else str(get_best_device())
    logger.info(f"Device: {device_resolved}")
    
    # Trainer
    trainer = EmbeddingTrainer(
        model=model,
        device=device_resolved,
        lr=lr,
        weight_decay=weight_decay,
        lambda_sigreg=lambda_sigreg,
        align_weight=align_weight,
        sigreg_n_points=sigreg_n_points,
        sigreg_num_slices=sigreg_num_slices,
        use_amp=use_amp,
    )
    
    # Scheduler
    scheduler = CosineAnnealingLR(
        trainer.optimizer,
        T_max=epochs * len(train_loader),
        eta_min=lr / 100,
    )
    
    # Training loop
    best_val = float("inf")
    logger.info(f"Training for {epochs} epochs...")
    for epoch in range(epochs):
        logger.info(f"Epoch {epoch + 1}/{epochs}")
        
        # Train
        epoch_loss = 0.0
        epoch_align = 0.0
        epoch_sigreg = 0.0
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (images, captions) in enumerate(pbar):
            metrics = trainer.train_step(images, captions)
            epoch_loss += metrics["loss"]
            epoch_align += metrics["align_loss"]
            epoch_sigreg += metrics["sigreg_loss"]
            scheduler.step()
            
            pbar.set_postfix(loss=f"{metrics['loss']:.4f}")
        
        denom = max(1, len(train_loader))
        avg_loss = epoch_loss / denom
        avg_align = epoch_align / denom
        avg_sigreg = epoch_sigreg / denom
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        val_loss = val_metrics["val_loss"]
        
        logger.info(
            f"Epoch {epoch + 1:03d} | "
            f"train_loss {avg_loss:.4f} | align {avg_align:.4f} | "
            f"sigreg {avg_sigreg:.4f} | val_loss {val_loss:.4f}"
        )
        
        # Save checkpoint
        ckpt_path = os.path.join(save_dir, f"epoch_{epoch + 1}.pt")
        trainer.save_checkpoint(ckpt_path, epoch + 1)

        if val_loss < best_val:
            best_val = val_loss
            best_path = os.path.join(save_dir, "best_model.pt")
            trainer.save_checkpoint(best_path, epoch + 1)
            logger.info(f"  New best val_loss {best_val:.4f} -> saved {best_path}")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeJEPA Captioner")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--model", type=str, default="small", choices=["tiny", "small", "base"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--lambda_sigreg", type=float, default=0.05)
    parser.add_argument("--align_weight", type=float, default=1.0)
    parser.add_argument("--sigreg_n_points", type=int, default=17)
    parser.add_argument("--sigreg_num_slices", type=int, default=128)
    parser.add_argument("--use_amp", action="store_true")
    
    args = parser.parse_args()
    
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        model_config=args.model,
        max_samples=args.max_samples,
        save_dir=args.save_dir,
        device=args.device,
        weight_decay=args.weight_decay,
        lambda_sigreg=args.lambda_sigreg,
        align_weight=args.align_weight,
        sigreg_n_points=args.sigreg_n_points,
        sigreg_num_slices=args.sigreg_num_slices,
        use_amp=args.use_amp,
    )
