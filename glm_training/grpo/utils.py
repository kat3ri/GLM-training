"""
Utility functions for GRPO training.

Day 1: Basic utilities for reward computation and tensor operations.
"""
import torch
import torch.nn.functional as F
from typing import List
from PIL import Image
import numpy as np


def simple_reward_function(
    images: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """
    Simple reward: negative MSE loss.
    
    Start with something simple that works.
    Later can add LPIPS, aesthetic scores, etc.
    
    Args:
        images: [N, C, H, W] - Generated images
        targets: [N, C, H, W] - Target images
        
    Returns:
        rewards: [N] - Reward per image (higher is better)
    """
    # MSE per image
    mse = F.mse_loss(images, targets, reduction='none')
    mse_per_image = mse.view(images.size(0), -1).mean(dim=1)
    
    # Convert to reward (negative loss)
    rewards = -mse_per_image
    
    return rewards


def pil_to_tensor(images: List[Image.Image]) -> torch.Tensor:
    """
    Convert list of PIL images to tensor.
    
    Args:
        images: List of PIL images
        
    Returns:
        tensor: [N, C, H, W] normalized to [0, 1]
    """
    tensors = []
    for img in images:
        arr = np.array(img).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        tensors.append(tensor)
    return torch.stack(tensors)


def tensor_to_pil(tensor: torch.Tensor) -> List[Image.Image]:
    """
    Convert tensor to list of PIL images.
    
    Args:
        tensor: [N, C, H, W] normalized to [0, 1]
        
    Returns:
        images: List of PIL images
    """
    images = []
    for t in tensor:
        arr = (t.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        img = Image.fromarray(arr)
        images.append(img)
    return images
