"""
Text-to-Image dataset for GLM-Image training.
"""
from pathlib import Path
from typing import Optional, List, Tuple

import torch
from torch.utils.data import Dataset
from PIL import Image


class T2IDataset(Dataset):
    """Text-to-Image dataset."""
    
    def __init__(
        self,
        prompts_file: str,
        target_images_dir: str,
        image_size: Tuple[int, int] = (1024, 1024),
        transform: Optional[callable] = None,
    ):
        """
        Initialize T2I dataset.
        
        Args:
            prompts_file: Path to file containing prompts (one per line)
            target_images_dir: Directory containing target images
            image_size: Target image size (height, width)
            transform: Optional transform to apply to images
        """
        self.prompts_file = Path(prompts_file)
        self.target_images_dir = Path(target_images_dir)
        self.image_size = image_size
        self.transform = transform
        
        # Load prompts
        with open(self.prompts_file, "r", encoding="utf-8") as f:
            self.prompts = [line.strip() for line in f if line.strip()]
        
        # Find corresponding target images
        self.image_paths = []
        for idx in range(len(self.prompts)):
            # Try common image extensions
            for ext in [".png", ".jpg", ".jpeg", ".webp"]:
                image_path = self.target_images_dir / f"{idx}{ext}"
                if image_path.exists():
                    self.image_paths.append(image_path)
                    break
            else:
                raise FileNotFoundError(
                    f"No target image found for prompt {idx} in {self.target_images_dir}"
                )
        
        assert len(self.prompts) == len(self.image_paths), \
            "Number of prompts must match number of target images"
    
    def __len__(self) -> int:
        return len(self.prompts)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing prompt and target image
        """
        prompt = self.prompts[idx]
        
        # Load target image
        target_image = Image.open(self.image_paths[idx]).convert("RGB")
        
        # Resize if needed
        if target_image.size != (self.image_size[1], self.image_size[0]):
            target_image = target_image.resize(
                (self.image_size[1], self.image_size[0]),
                Image.LANCZOS
            )
        
        # Apply transform if provided
        if self.transform is not None:
            target_image = self.transform(target_image)
        else:
            # Convert to tensor
            target_image = torch.from_numpy(
                torch.tensor(target_image).numpy()
            ).float() / 255.0
            target_image = target_image.permute(2, 0, 1)  # HWC -> CHW
        
        return {
            "prompt": prompt,
            "target_image": target_image,
            "idx": idx,
        }


def collate_t2i(batch: List[dict]) -> dict:
    """
    Collate function for T2I dataset.
    
    Args:
        batch: List of samples
        
    Returns:
        Batched dictionary
    """
    prompts = [item["prompt"] for item in batch]
    target_images = torch.stack([item["target_image"] for item in batch])
    indices = torch.tensor([item["idx"] for item in batch])
    
    return {
        "prompts": prompts,
        "target_images": target_images,
        "indices": indices,
    }
