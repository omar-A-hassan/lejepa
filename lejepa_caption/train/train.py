"""
Training utilities for LeJEPA Edge Captioner.

Following VL-JEPA paper (Dec 2024):
- Combined InfoNCE + MSE loss (direction + magnitude)
- Optional MoCo-style memory queue for more negatives
- Y-Encoder with 0.05x LR multiplier (learns 20x slower)
- LR warmup + cosine decay
- Wandb logging
- Best model checkpoint saving

Key benefits:
1. InfoNCE cares about direction (more forgiving than pure MSE)
2. MSE keeps embeddings on-manifold (better decoding)
3. MoCo queue: 4096+ negatives without large batch sizes
4. VL-JEPA validated: This approach works in their 1.6B model
"""

import os
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from tqdm import tqdm

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from lejepa_caption.models import LeJEPACaptioner, get_captioner
from lejepa_caption.train.moco_queue import MoCoQueue, infonce_with_queue
from lejepa_caption.train.gradcache import (
    GRADCACHE_AVAILABLE,
    FunctionalGradCache,
)


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

    Following VL-JEPA paper:
    - InfoNCE loss (not MSE or SIGReg)
    - Y-Encoder with 0.05x LR multiplier
    - LR warmup + cosine decay
    - Optional MoCo queue for more negatives
    - Wandb logging
    """

    def __init__(
        self,
        model: LeJEPACaptioner,
        tokenizer,
        llm,
        device: str = "cuda",
        lr: float = 5e-4,  # VL-JEPA uses 5e-4
        y_encoder_lr_multiplier: float = 0.05,  # Y-Encoder learns 20x slower
        weight_decay: float = 0.01,
        temperature: float = 0.07,  # InfoNCE temperature
        mse_alpha: float = 0.0,  # MSE weight (0 = pure InfoNCE, 0.15 = recommended)
        use_moco_queue: bool = False,  # Enable memory queue for negatives
        queue_size: int = 4096,  # Number of negatives to store
        use_gradcache: bool = False,  # Enable GradCache for large effective batches
        gradcache_chunk_size: int = 32,  # Sub-batch size for GradCache
        use_amp: bool = False,
        use_lr_schedule: bool = True,
        total_steps: Optional[int] = None,  # Required if use_lr_schedule=True
        use_wandb: bool = False,
    ):
        if tokenizer is None or llm is None:
            raise ValueError("Tokenizer and LLM must be provided (preloaded in notebook).")

        self.model = model.to(device)
        self.device = device
        self.use_amp = use_amp
        self.temperature = temperature
        self.mse_alpha = mse_alpha  # Combined loss weight
        self.use_moco_queue = use_moco_queue
        self.use_gradcache = use_gradcache
        self.gradcache_chunk_size = gradcache_chunk_size
        self.use_wandb = use_wandb
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp and torch.cuda.is_available())

        # GradCache for large effective batch sizes
        if use_gradcache:
            if not GRADCACHE_AVAILABLE:
                raise ImportError(
                    "GradCache requested but not installed. Install with:\n"
                    "  pip install git+https://github.com/luyug/GradCache.git"
                )
            # Note: We initialize grad_cache later in _create_gradcache() 
            # after moco_queue is created so the loss_fn can use it
            self._pending_gradcache = True
        else:
            self._pending_gradcache = False
            self.grad_cache = None

        # MoCo-style memory queue for negatives
        if use_moco_queue:
            # Get LLM embedding dimension
            llm_dim = llm.get_input_embeddings().embedding_dim
            self.moco_queue = MoCoQueue(dim=llm_dim, queue_size=queue_size, device=device)
        else:
            self.moco_queue = None
        
        # Now create GradCache with loss function that uses moco_queue
        if self._pending_gradcache:
            self._create_gradcache()

        # Preloaded tokenizer/LLM
        self.tokenizer = tokenizer
        self.llm = llm

        # Optimizer with separate LR for Y-Encoder
        # Y-Encoder = LLM's input embedding layer (Gemma-3)
        y_encoder_params = list(llm.get_input_embeddings().parameters())
        y_encoder_ids = {id(p) for p in y_encoder_params}

        # Model params (encoder + connector + predictor)
        model_params = [p for p in model.parameters() if id(p) not in y_encoder_ids]

        # Separate parameter groups with different LRs
        self.optimizer = AdamW([
            {
                'params': model_params,
                'lr': lr,
                'weight_decay': weight_decay,
            },
            {
                'params': y_encoder_params,
                'lr': lr * y_encoder_lr_multiplier,  # 0.05x slower!
                'weight_decay': weight_decay,
            }
        ])

        # LR scheduler (warmup + cosine decay)
        # Note: Scheduler applies decay to BOTH parameter groups
        # So Y-Encoder stays 20x slower throughout training
        self.scheduler = None
        if use_lr_schedule and total_steps is not None:
            warmup_steps = max(1, total_steps // 10)  # 10% warmup
            s1 = LinearLR(
                self.optimizer,
                start_factor=0.01,
                total_iters=warmup_steps
            )
            s2 = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=lr / 1000  # Final LR = initial / 1000
            )
            self.scheduler = SequentialLR(
                self.optimizer,
                schedulers=[s1, s2],
                milestones=[warmup_steps]
            )

        # Track best validation loss for checkpointing
        self.best_val_loss = float('inf')

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
        tokens = self.tokenizer(
            captions,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            embeddings = self.llm.get_input_embeddings()(tokens.input_ids)

        return embeddings.float()
    
    def _create_gradcache(self):
        """
        Create FunctionalGradCache with a loss function that uses the MoCo queue.
        
        This ensures GradCache computes the same loss as the standard training path,
        including MoCo queue negatives for consistent train/val behavior.
        """
        # Create a closure that uses self.infonce_loss (which uses moco_queue)
        def gradcache_loss_fn(pred_embeds: torch.Tensor, target_embeds: torch.Tensor) -> dict:
            """
            Loss function for GradCache that uses MoCo queue.
            
            This wraps infonce_loss but handles the queue update carefully:
            - We compute loss using ALL accumulated embeddings
            - We only update the queue once after the loss is computed
            """
            return self.infonce_loss(pred_embeds, target_embeds)
        
        self.grad_cache = FunctionalGradCache(
            model=self.model,
            loss_fn=gradcache_loss_fn,
        )

    def infonce_loss(
        self,
        pred_embeds: torch.Tensor,
        target_embeds: torch.Tensor
    ) -> dict:
        """
        Compute combined InfoNCE + MSE loss.

        InfoNCE = Alignment (cosine similarity) + Uniformity (anti-collapse)
        MSE = Mean squared error on normalized embeddings (magnitude matching)

        Combined loss: L_total = L_InfoNCE + alpha * L_MSE

        If MoCo queue is enabled, uses queue negatives for better contrastive signal.

        Benefits:
        1. InfoNCE cares about direction (more forgiving)
        2. MSE keeps embeddings on-manifold (better decoding)
        3. MoCo queue: 4096+ negatives without large batch sizes
        4. Combined: Direction + magnitude = better alignment

        Args:
            pred_embeds: Predicted embeddings (B, L, D)
            target_embeds: Target embeddings (B, L, D)

        Returns:
            dict with 'loss', 'infonce', 'mse', 'alignment', 'uniformity', 'num_negatives'
        """
        B, L, D = pred_embeds.shape

        # Pool sequence dimension: (B, L, D) -> (B, D)
        # One embedding per caption, not per token position
        pred_pooled = pred_embeds.mean(dim=1)
        target_pooled = target_embeds.mean(dim=1)

        # L2 normalize for cosine similarity
        pred_norm = F.normalize(pred_pooled, dim=-1, p=2)
        target_norm = F.normalize(target_pooled, dim=-1, p=2)

        # ===== InfoNCE Component =====
        if self.moco_queue is not None and len(self.moco_queue) > 0:
            # Use queue for extra negatives
            infonce_loss = infonce_with_queue(
                pred_norm, target_norm, 
                self.moco_queue, 
                self.temperature
            )
            num_negatives = B - 1 + len(self.moco_queue)
            
            # Update queue with current batch targets (after computing loss)
            self.moco_queue.enqueue(target_norm.detach())
        else:
            # Standard bi-directional InfoNCE
            logits = (pred_norm @ target_norm.T) / self.temperature
            labels = torch.arange(B, device=self.device)
            loss_pred_to_target = F.cross_entropy(logits, labels)
            loss_target_to_pred = F.cross_entropy(logits.T, labels)
            infonce_loss = (loss_pred_to_target + loss_target_to_pred) / 2
            num_negatives = B - 1
            
            # If queue exists but is empty, start filling it
            if self.moco_queue is not None:
                self.moco_queue.enqueue(target_norm.detach())

        # ===== MSE Component (on normalized embeddings) =====
        # This keeps predictions on the LLM embedding manifold
        mse_loss = F.mse_loss(pred_norm, target_norm)

        # ===== Combined Loss =====
        # alpha=0 → pure InfoNCE, alpha=0.15 → recommended hybrid
        total_loss = infonce_loss + self.mse_alpha * mse_loss

        # Compute metrics for logging
        with torch.no_grad():
            # Alignment: average cosine similarity of matched pairs
            # Higher is better (1.0 = perfect alignment)
            alignment = (pred_norm * target_norm).sum(dim=-1).mean()

            # Uniformity: how spread out embeddings are
            # Lower is better (prevents collapse to single point)
            pred_sim = pred_norm @ pred_norm.T
            uniformity = pred_sim.triu(diagonal=1).mean()

        return {
            'loss': total_loss,
            'infonce': infonce_loss.item() if hasattr(infonce_loss, 'item') else infonce_loss,
            'mse': mse_loss.item(),
            'alignment': alignment.item(),
            'uniformity': uniformity.item(),
            'num_negatives': num_negatives,
        }

    def gradcache_accumulate(
        self,
        images: torch.Tensor,
        captions: list,
    ):
        """
        Accumulate a sub-batch for GradCache training.
        
        Call this multiple times to accumulate sub-batches, then call
        gradcache_step() to compute loss and update weights.
        
        Args:
            images: Sub-batch of images (B, 3, 224, 224)
            captions: List of caption strings
        """
        if self.grad_cache is None:
            raise RuntimeError("GradCache not enabled. Set use_gradcache=True in constructor.")
        
        self.model.train()
        images = images.to(self.device)
        
        # Get target embeddings
        with torch.no_grad():
            # Need to determine target seq len - use model's config
            max_len = self.model.config.get('max_caption_len', 50)
            target_embeds = self.get_target_embeddings(captions, max_len=max_len)
        
        # Accumulate in GradCache
        # NOTE: Do NOT update MoCo queue here! The queue is updated inside 
        # infonce_loss() which is called during compute_loss(). Updating here 
        # would pollute the queue before we compute loss with all negatives.
        self.grad_cache.accumulate(images, target_embeds)
    
    def gradcache_step(self, global_step: int = 0) -> dict:
        """
        Compute loss from accumulated sub-batches and update weights.
        
        Call this after accumulating sub-batches with gradcache_accumulate().
        
        Args:
            global_step: Current training step (for logging)
            
        Returns:
            Dict with loss and metrics (same format as train_step)
        """
        if self.grad_cache is None:
            raise RuntimeError("GradCache not enabled. Set use_gradcache=True in constructor.")
        
        if self.grad_cache.num_accumulated == 0:
            raise RuntimeError("No sub-batches accumulated. Call gradcache_accumulate() first.")
        
        self.optimizer.zero_grad(set_to_none=True)
        
        # Compute loss with ALL accumulated embeddings
        # This uses self.infonce_loss which uses the MoCo queue
        loss = self.grad_cache.compute_loss()
        
        # Get metrics from the loss computation (before reset!)
        metrics = self.grad_cache.last_metrics or {}
        effective_batch = sum(p.shape[0] for p in self.grad_cache._pred_cache)
        
        loss.backward()
        
        # Propagate gradients through cached representations
        self.grad_cache.backward_cached()
        
        # Optimizer step
        self.optimizer.step()
        
        # Update LR schedule
        if self.scheduler is not None:
            self.scheduler.step()
        
        # Reset cache for next accumulation cycle
        self.grad_cache.reset()
        
        # Get learning rates
        loss_value = loss.item()
        lr_predictor = self.optimizer.param_groups[0]['lr']
        lr_y_encoder = self.optimizer.param_groups[1]['lr']
        
        # Extract metrics (same as train_step for consistency)
        infonce = metrics.get('infonce', loss_value)
        mse = metrics.get('mse', 0.0)
        alignment = metrics.get('alignment', 0.0)
        uniformity = metrics.get('uniformity', 0.0)
        num_negatives = metrics.get('num_negatives', effective_batch - 1)
        
        # Wandb logging (same metrics as train_step)
        if self.use_wandb:
            import wandb
            wandb.log({
                "train/loss": loss_value,
                "train/infonce": infonce,
                "train/mse": mse,
                "train/alignment": alignment,
                "train/uniformity": uniformity,
                "train/num_negatives": num_negatives,
                "train/lr_predictor": lr_predictor,
                "train/lr_y_encoder": lr_y_encoder,
                "train/effective_batch_size": effective_batch,
                "train/step": global_step,
            })
        
        return {
            'loss': loss_value,
            'infonce': infonce,
            'mse': mse,
            'alignment': alignment,
            'uniformity': uniformity,
            'num_negatives': num_negatives,
            'lr_predictor': lr_predictor,
            'lr_y_encoder': lr_y_encoder,
            'effective_batch_size': effective_batch,
        }

    def train_step(
        self,
        images: torch.Tensor,
        captions: list,
        global_step: int = 0,
    ) -> dict:
        """
        Single training step with InfoNCE loss.

        Args:
            images: Batch of images (B, 3, 224, 224)
            captions: List of caption strings
            global_step: Current training step (for logging)

        Returns:
            Dict with loss and metrics
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        images = images.to(self.device)

        # Forward pass
        with autocast(enabled=self.use_amp):
            pred_embeds = self.model(images)  # (B, 50, 640)

            # Get target embeddings from frozen Gemma-3
            target_embeds = self.get_target_embeddings(
                captions,
                max_len=pred_embeds.size(1)
            )
            target_embeds = target_embeds.to(pred_embeds.dtype)

            # InfoNCE loss
            loss_dict = self.infonce_loss(pred_embeds, target_embeds)
            loss = loss_dict['loss']

        # Backward (no gradient clipping - InfoNCE is stable)
        if self.scaler is not None and self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()

        # Update LR schedule
        if self.scheduler is not None:
            self.scheduler.step()

        # Get current learning rates
        lr_predictor = self.optimizer.param_groups[0]['lr']
        lr_y_encoder = self.optimizer.param_groups[1]['lr']

        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log({
                "train/loss": loss.item(),
                "train/infonce": loss_dict['infonce'],
                "train/mse": loss_dict['mse'],
                "train/alignment": loss_dict['alignment'],
                "train/uniformity": loss_dict['uniformity'],
                "train/lr_predictor": lr_predictor,
                "train/lr_y_encoder": lr_y_encoder,
                "train/step": global_step,
            })

        return {
            'loss': loss.item(),
            'infonce': loss_dict['infonce'],
            'mse': loss_dict['mse'],
            'alignment': loss_dict['alignment'],
            'uniformity': loss_dict['uniformity'],
            'lr_predictor': lr_predictor,
            'lr_y_encoder': lr_y_encoder,
        }

    def validate(self, dataloader, epoch: int = 0) -> dict:
        """
        Run validation with InfoNCE loss.

        Args:
            dataloader: Validation dataloader
            epoch: Current epoch (for logging)

        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        total_loss = 0
        total_infonce = 0
        total_mse = 0
        total_alignment = 0
        total_uniformity = 0
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

                loss_dict = self.infonce_loss(pred_embeds, target_embeds)

                total_loss += loss_dict['loss'].item()
                total_infonce += loss_dict['infonce']
                total_mse += loss_dict['mse']
                total_alignment += loss_dict['alignment']
                total_uniformity += loss_dict['uniformity']
                num_batches += 1

        denom = max(1, num_batches)
        val_metrics = {
            "val_loss": total_loss / denom,
            "val_infonce": total_infonce / denom,
            "val_mse": total_mse / denom,
            "val_alignment": total_alignment / denom,
            "val_uniformity": total_uniformity / denom,
        }

        # Wandb logging
        if self.use_wandb:
            import wandb
            wandb.log({
                "val/loss": val_metrics["val_loss"],
                "val/infonce": val_metrics["val_infonce"],
                "val/mse": val_metrics["val_mse"],
                "val/alignment": val_metrics["val_alignment"],
                "val/uniformity": val_metrics["val_uniformity"],
                "val/epoch": epoch,
            })

        return val_metrics

    def save_checkpoint(self, path: str, epoch: int, is_best: bool = False):
        """
        Save model checkpoint.

        Args:
            path: Path to save checkpoint
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_loss": self.best_val_loss,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, path)

        if is_best:
            print(f"  ✓ Saved BEST model: {path} (val_loss: {self.best_val_loss:.4f})")
        else:
            print(f"  Saved checkpoint: {path}")


def train_with_loader(
    model: LeJEPACaptioner,
    train_loader,
    val_loader,
    tokenizer,
    llm,
    device: str = "cuda",
    lr: float = 5e-4,  # VL-JEPA default (was 1e-4 for MSE)
    y_encoder_lr_multiplier: float = 0.05,  # Y-Encoder learns 20x slower
    weight_decay: float = 0.01,
    temperature: float = 0.07,  # InfoNCE temperature
    mse_alpha: float = 0.0,  # MSE weight (0 = pure InfoNCE, 0.15 = recommended)
    use_moco_queue: bool = False,  # Enable memory queue for negatives
    queue_size: int = 4096,  # Number of negatives to store
    use_gradcache: bool = False,  # Enable GradCache for large effective batches
    gradcache_accum_steps: int = 4,  # Number of sub-batches to accumulate
    use_amp: bool = False,
    use_lr_schedule: bool = True,
    epochs: int = 3,
    use_wandb: bool = False,
    wandb_project: str = "lejepa-captioner",
    wandb_run_name: Optional[str] = None,
    checkpoint_dir: str = "checkpoints",
):
    """
    Train with VL-JEPA approach.

    Key features:
    - Combined InfoNCE + MSE loss (alpha controls MSE weight)
    - Optional MoCo queue for 4096+ negatives
    - Optional GradCache for large effective batch sizes
    - Y-Encoder with 0.05x LR multiplier
    - LR warmup + cosine decay
    - Validation every epoch
    - Save only the best model (based on val loss)
    - Wandb logging (optional)

    Args:
        model: LeJEPACaptioner instance
        train_loader: Training dataloader
        val_loader: Validation dataloader
        tokenizer: Pretrained tokenizer
        llm: Pretrained LLM (frozen)
        device: Device to train on
        lr: Base learning rate (predictor uses this, Y-encoder uses lr * 0.05)
        y_encoder_lr_multiplier: LR multiplier for Y-Encoder (default 0.05)
        weight_decay: AdamW weight decay
        temperature: InfoNCE temperature
        mse_alpha: MSE loss weight (0 = pure InfoNCE, 0.15 = recommended hybrid)
        use_moco_queue: Enable MoCo-style memory queue for negatives
        queue_size: Number of embeddings to store in queue (default 4096)
        use_gradcache: Enable GradCache for large effective batch sizes
        gradcache_accum_steps: Number of sub-batches to accumulate before update
        use_amp: Use automatic mixed precision
        use_lr_schedule: Use warmup + cosine decay
        epochs: Number of training epochs
        use_wandb: Enable wandb logging
        wandb_project: Wandb project name
        wandb_run_name: Wandb run name (optional)
        checkpoint_dir: Directory to save checkpoints

    Returns:
        EmbeddingTrainer instance (with trained model)
    """
    device = torch.device(device)

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config={
                "lr": lr,
                "y_encoder_lr_multiplier": y_encoder_lr_multiplier,
                "weight_decay": weight_decay,
                "temperature": temperature,
                "mse_alpha": mse_alpha,
                "use_moco_queue": use_moco_queue,
                "queue_size": queue_size if use_moco_queue else 0,
                "use_gradcache": use_gradcache,
                "gradcache_accum_steps": gradcache_accum_steps if use_gradcache else 0,
                "use_amp": use_amp,
                "use_lr_schedule": use_lr_schedule,
                "epochs": epochs,
                "batch_size": train_loader.batch_size,
                "effective_batch_size": train_loader.batch_size * gradcache_accum_steps if use_gradcache else train_loader.batch_size,
                "model_params": sum(p.numel() for p in model.parameters()) / 1e6,
            }
        )

    # Calculate total steps for LR schedule
    # With GradCache, we update less frequently (every accum_steps batches)
    if use_gradcache:
        steps_per_epoch = len(train_loader) // gradcache_accum_steps
    else:
        steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * epochs if use_lr_schedule else None

    trainer = EmbeddingTrainer(
        model,
        tokenizer=tokenizer,
        llm=llm,
        device=device,
        lr=lr,
        y_encoder_lr_multiplier=y_encoder_lr_multiplier,
        weight_decay=weight_decay,
        temperature=temperature,
        mse_alpha=mse_alpha,
        use_moco_queue=use_moco_queue,
        queue_size=queue_size,
        use_gradcache=use_gradcache,
        gradcache_chunk_size=train_loader.batch_size,  # Each batch is a chunk
        use_amp=use_amp,
        use_lr_schedule=use_lr_schedule,
        total_steps=total_steps,
        use_wandb=use_wandb,
    )

    effective_batch = train_loader.batch_size * gradcache_accum_steps if use_gradcache else train_loader.batch_size

    print(f"\n{'='*60}")
    print(f"Training with VL-JEPA approach:")
    print(f"  - Loss: InfoNCE + MSE (alpha={mse_alpha})")
    print(f"  - Predictor LR: {lr}")
    print(f"  - Y-Encoder LR: {lr * y_encoder_lr_multiplier} (0.05x slower)")
    print(f"  - MoCo Queue: {'Enabled (size=' + str(queue_size) + ')' if use_moco_queue else 'Disabled'}")
    print(f"  - GradCache: {'Enabled (accum=' + str(gradcache_accum_steps) + ', effective_batch=' + str(effective_batch) + ')' if use_gradcache else 'Disabled'}")
    print(f"  - LR Schedule: {'Warmup + Cosine Decay' if use_lr_schedule else 'Constant'}")
    print(f"  - Wandb: {'Enabled' if use_wandb else 'Disabled'}")
    print(f"  - Checkpoint Dir: {checkpoint_dir}")
    print(f"{'='*60}\n")

    global_step = 0

    for epoch in range(1, epochs + 1):
        # Training
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        if use_gradcache:
            # GradCache training loop: accumulate then update
            accum_count = 0
            for images, captions in pbar:
                # Accumulate sub-batch
                trainer.gradcache_accumulate(images, captions)
                accum_count += 1
                
                # Update after accumulating enough sub-batches
                if accum_count >= gradcache_accum_steps:
                    stats = trainer.gradcache_step(global_step)
                    pbar.set_postfix({
                        "loss": f"{stats['loss']:.4f}",
                        "infonce": f"{stats['infonce']:.4f}",
                        "mse": f"{stats['mse']:.4f}",
                        "align": f"{stats['alignment']:.3f}",
                        "eff_bs": stats['effective_batch_size'],
                        "lr": f"{stats['lr_predictor']:.2e}",
                    })
                    global_step += 1
                    accum_count = 0
            
            # Handle remaining accumulated batches at end of epoch
            if accum_count > 0:
                stats = trainer.gradcache_step(global_step)
                global_step += 1
        else:
            # Standard training loop
            for images, captions in pbar:
                stats = trainer.train_step(images, captions, global_step)
                pbar.set_postfix({
                    "loss": f"{stats['loss']:.4f}",
                    "infonce": f"{stats['infonce']:.4f}",
                    "mse": f"{stats['mse']:.4f}",
                    "align": f"{stats['alignment']:.3f}",
                    "lr": f"{stats['lr_predictor']:.2e}",
                })
                global_step += 1

        # Validation
        val_metrics = trainer.validate(val_loader, epoch)
        val_loss = val_metrics["val_loss"]

        print(f"\nEpoch {epoch}/{epochs} Summary:")
        print(f"  Val Loss:      {val_loss:.4f}")
        print(f"  Val Alignment: {val_metrics['val_alignment']:.4f}")
        print(f"  Val Uniformity: {val_metrics['val_uniformity']:.4f}")

        # Save checkpoint if best model
        is_best = val_loss < trainer.best_val_loss
        if is_best:
            trainer.best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, "best_model.pt")
            trainer.save_checkpoint(best_path, epoch, is_best=True)

        # Also save latest checkpoint
        latest_path = os.path.join(checkpoint_dir, "latest_checkpoint.pt")
        trainer.save_checkpoint(latest_path, epoch, is_best=False)

        print()  # Blank line between epochs

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best Val Loss: {trainer.best_val_loss:.4f}")
    print(f"  Best Model: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"{'='*60}\n")

    if use_wandb:
        import wandb
        wandb.finish()

    return trainer
