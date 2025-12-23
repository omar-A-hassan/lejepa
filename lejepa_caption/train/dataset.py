"""
COCO Captions Dataset for LeJEPA Training.

Loads images and captions for embedding prediction training.
Supports both HuggingFace datasets and local COCO format.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from typing import Optional, Callable, List, Tuple
import os


class COCOCaptionsDataset(Dataset):
    """
    COCO Captions dataset for VL-JEPA training.
    
    Loads images and returns (image, caption) pairs.
    Uses HuggingFace datasets for easy access.
    """
    
    def __init__(
        self,
        split: str = "train",
        transform: Optional[Callable] = None,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            split: 'train' or 'validation'
            transform: Image transforms (default: standard 224x224)
            max_samples: Limit dataset size for debugging
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")

        # Load COCO from a script-free source to avoid trust_remote_code bans (e.g., Kaggle)
        # Dataset fields: image (PIL), sentences_raw (list[str])
        print(f"Loading COCO Captions ({split})...")
        requested_split = "train" if split == "train" else "validation"
        try:
            self.dataset = load_dataset(
                "Multimodal-Fatima/COCO_captions_train",
                split=requested_split,
            )
        except Exception:
            # Fallback: if only train split exists, reuse it for validation
            self.dataset = load_dataset(
                "Multimodal-Fatima/COCO_captions_train",
                split="train",
            )
        
        if max_samples:
            self.dataset = self.dataset.select(range(min(max_samples, len(self.dataset))))
        
        # Default transforms for ViT
        self.transform = transform or transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
        
        print(f"  Loaded {len(self.dataset)} samples")
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        """
        Returns:
            image: Tensor (3, 224, 224)
            caption: String (first caption for the image)
        """
        item = self.dataset[idx]
        
        # Handle image
        image = item["image"]
        if not isinstance(image, Image.Image):
            image = Image.open(image).convert("RGB")
        else:
            image = image.convert("RGB")
        
        image = self.transform(image)
        
        # Handle caption (dataset stores a list of raw sentences)
        captions = item.get("sentences_raw", item.get("captions", []))
        if isinstance(captions, list) and len(captions) > 0:
            caption = captions[0]
        else:
            caption = str(captions)
        
        return image, caption


def get_train_augmentations() -> transforms.Compose:
    """Deterministic train transform matching the notebook path."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def get_dataloader(
    split: str = "train",
    batch_size: int = 32,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
) -> DataLoader:
    """
    Create a DataLoader for COCO Captions.
    
    Args:
        split: 'train' or 'validation'
        batch_size: Batch size
        num_workers: Data loading workers
        max_samples: Limit samples (for debugging)
    """
    transform = get_train_augmentations() if split == "train" else None
    
    dataset = COCOCaptionsDataset(
        split=split,
        transform=transform,
        max_samples=max_samples,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )


if __name__ == "__main__":
    # Quick test
    print("Testing dataset loader...")
    
    # Test with small subset
    dataset = COCOCaptionsDataset(split="validation", max_samples=10)
    
    img, cap = dataset[0]
    print(f"  Image shape: {img.shape}")
    print(f"  Caption: {cap[:50]}...")
    
    # Test dataloader
    loader = get_dataloader(split="validation", batch_size=4, max_samples=10)
    for imgs, caps in loader:
        print(f"  Batch: {imgs.shape}, {len(caps)} captions")
        break
    
    print("Dataset test passed!")
