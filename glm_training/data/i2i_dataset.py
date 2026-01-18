"""
Image-to-Image dataset for GLM-Image training.
"""
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class I2IDataset(Dataset):
    """Image-to-Image dataset."""
    
    def __init__(
        self,
        source_images_dir: str,
        prompts_file: str,
        target_images_dir: str,
        image_size: Tuple[int, int] = (1024, 1024),
        transform: Optional[callable] = None,
        augmentation: Optional[dict] = None,
    ):
        """
        Initialize I2I dataset.
        
        Args:
            source_images_dir: Directory containing source images
            prompts_file: Path to file containing edit prompts (one per line)
            target_images_dir: Directory containing target images
            image_size: Target image size (height, width)
            transform: Optional transform to apply to images
            augmentation: Augmentation configuration
        """
        self.source_images_dir = Path(source_images_dir)
        self.prompts_file = Path(prompts_file)
        self.target_images_dir = Path(target_images_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Load prompts
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f if line.strip()]
        
        # Find corresponding source and target images
        self.source_image_paths = []
        self.target_image_paths = []
        
        for idx in range(len(self.prompts)):
            # Find source image
            source_found = False
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                source_path = self.source_images_dir / f"{idx}{ext}"
                if source_path.exists():
                    self.source_image_paths.append(source_path)
                    source_found = True
                    break
            
            if not source_found:
                raise FileNotFoundError(
                    f"No source image found for prompt {idx} in {self.source_images_dir}"
                )
            
            # Find target image
            target_found = False
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                target_path = self.target_images_dir / f"{idx}{ext}"
                if target_path.exists():
                    self.target_image_paths.append(target_path)
                    target_found = True
                    break
            
            if not target_found:
                raise FileNotFoundError(
                    f"No target image found for prompt {idx} in {self.target_images_dir}"
                )
        
        # Setup augmentation
        self.augmentation = augmentation or {}
        self._setup_augmentation()
    
    def _setup_augmentation(self):
        """Setup data augmentation transforms."""
        aug_transforms = []
        
        if self.augmentation.get("random_flip", False):
            aug_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if self.augmentation.get("color_jitter", False):
            aug_transforms.append(transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ))
        
        if aug_transforms:
            self.aug_transform = transforms.Compose(aug_transforms)
        else:
            self.aug_transform = None
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing prompt, source image, and target image
        """
        prompt = self.prompts[idx]
        
        # Load source image
        source_image = Image.open(self.source_image_paths[idx]).convert("RGB")
        
        # Load target image
        target_image = Image.open(self.target_image_paths[idx]).convert("RGB")
        
        # Resize if needed
        if source_image.size != (self.image_size[1], self.image_size[0]):
            source_image = source_image.resize(
                (self.image_size[1], self.image_size[0]),
                Image.LANCZOS
            )
        
        if target_image.size != (self.image_size[1], self.image_size[0]):
            target_image = target_image.resize(
                (self.image_size[1], self.image_size[0]),
                Image.LANCZOS
            )
        
        # Apply augmentation (same for both source and target)
        if self.aug_transform is not None and self.augmentation.get("enabled", False):
            # Set same random seed for both images to apply same augmentation
            seed = torch.randint(0, 2**32, (1,)).item()
            
            torch.manual_seed(seed)
            source_image = self.aug_transform(source_image)
            
            torch.manual_seed(seed)
            target_image = self.aug_transform(target_image)
        
        # Apply transform if provided
        if self.transform is not None:
            source_image = self.transform(source_image)
            target_image = self.transform(target_image)
        else:
            # Convert to tensor
            source_image = torch.from_numpy(
                torch.tensor(source_image).numpy()
            ).float() / 255.0
            source_image = source_image.permute(2, 0, 1)  # HWC -> CHW
            
            target_image = torch.from_numpy(
                torch.tensor(target_image).numpy()
            ).float() / 255.0
            target_image = target_image.permute(2, 0, 1)  # HWC -> CHW
        
        return {
            "prompt": prompt,
            "source_image": source_image,
            "target_image": target_image,
            "idx": idx,
        }


def collate_i2i(batch: List[dict]) -> dict:
    """
    Collate function for I2I dataset.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched dictionary
    """
    prompts = [item["prompt"] for item in batch]
    source_images = torch.stack([item["source_image"] for item in batch])
    target_images = torch.stack([item["target_image"] for item in batch])
    indices = torch.tensor([item["idx"] for item in batch])
    
    return {
        "prompts": prompts,
        "source_images": source_images,
        "target_images": target_images,
        "indices": indices,
    }
