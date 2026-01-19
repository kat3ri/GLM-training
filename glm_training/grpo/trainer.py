"""
Minimal GRPO trainer - starting from scratch.

Based on nanoGRPO + TRL patterns, but simplified.
Day 3: Integrated with GLM-Image.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Callable, Union
from PIL import Image
import numpy as np

from .loss import compute_advantages, compute_grpo_loss


class MinimalGRPOTrainer:
    """
    Minimal GRPO trainer - just the essentials.
    
    Now integrated with GLMImageWrapper for proper generation and log prob tracking.
    
    Args:
        model: GLMImageWrapper instance
        reward_function: Function that takes (images, targets) -> rewards
        train_dataloader: Training data iterator
        group_size: Samples per prompt (default: 4)
        clip_range: PPO clipping epsilon (default: 0.2)
        learning_rate: Learning rate (default: 5e-6)
        height: Image height (default: 1024)
        width: Image width (default: 1024)
        num_inference_steps: DiT inference steps (default: 50)
        guidance_scale: Guidance scale (default: 1.5)
    """
    
    def __init__(
        self,
        model,
        reward_function: Callable,
        train_dataloader,
        group_size: int = 4,
        clip_range: float = 0.2,
        learning_rate: float = 5e-6,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
    ):
        """Initialize minimal GRPO trainer."""
        self.model = model
        self.train_dataloader = train_dataloader
        self.reward_function = reward_function
        self.group_size = group_size
        self.clip_range = clip_range
        
        # Generation settings
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        
        # Setup optimizer
        # Get trainable parameters from model
        trainable_params = model.get_trainable_parameters()
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        self.device = next(model.parameters()).device
        self.step_count = 0
        
    def generate_samples(
        self, 
        prompts: List[str],
        source_images: Optional[List[Image.Image]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple samples per prompt using GLMImageWrapper.
        
        Args:
            prompts: List of text prompts
            source_images: Optional source images for i2i mode
            
        Returns:
            images: [batch_size * group_size, C, H, W]
            old_logprobs: [batch_size * group_size]
            token_ids: [batch_size * group_size, seq_len]
        """
        # Repeat prompts for group sampling
        repeated_prompts = []
        repeated_source_images = []
        
        for i, prompt in enumerate(prompts):
            repeated_prompts.extend([prompt] * self.group_size)
            if source_images is not None:
                repeated_source_images.extend([source_images[i]] * self.group_size)
        
        # Use source images if provided
        images_to_use = repeated_source_images if source_images is not None else None
        
        # Generate with tracking (no gradients)
        with torch.no_grad():
            results = self.model.generate_with_tracking(
                prompts=repeated_prompts,
                images=images_to_use,
                height=self.height,
                width=self.width,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
            )
        
        # Extract results
        image_tensors = results["image_tensors"].to(self.device)
        old_logprobs = results["log_probs"].to(self.device)
        token_ids = results["token_ids"].to(self.device)
        
        return image_tensors, old_logprobs, token_ids
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single GRPO training step.
        
        This is the core training loop following GRPO algorithm:
        1. Generate multiple samples per prompt (group)
        2. Compute rewards for all samples
        3. Compute advantages (z-score within groups)
        4. Recompute log probs with gradients
        5. Compute GRPO loss
        6. Backward and optimize
        
        Args:
            batch: Training batch with 'prompts' and 'target_images'
            
        Returns:
            metrics: Dictionary of training metrics
        """
        prompts = batch["prompts"]
        target_images = batch["target_images"].to(self.device)
        
        # Get source images for i2i mode if present
        source_images = None
        if "source_images" in batch:
            # Convert tensor to PIL images
            source_images_tensor = batch["source_images"].to(self.device)
            source_images = []
            for img_tensor in source_images_tensor:
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                source_images.append(Image.fromarray(img_np))
        
        # Step 1: Generate samples (no gradients)
        images, old_logprobs, token_ids = self.generate_samples(prompts, source_images)
        
        # Step 2: Compute rewards
        # Repeat targets for each sample in the group
        repeated_targets = target_images.repeat_interleave(
            self.group_size, dim=0
        )
        rewards = self.reward_function(images, repeated_targets)
        
        # Step 3: Compute advantages (z-score per group)
        advantages = compute_advantages(rewards, self.group_size)
        
        # Step 4: Recompute log probs (with gradients)
        # This allows gradients to flow through the policy
        new_logprobs = self.model.compute_sequence_logprobs(token_ids)
        
        # Step 5: Compute GRPO loss
        loss, metrics = compute_grpo_loss(
            new_logprobs,
            old_logprobs,
            advantages,
            self.clip_range,
        )
        
        # Step 6: Backward and optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.get_trainable_parameters(), 1.0)
        self.optimizer.step()
        
        # Add reward metrics
        metrics["reward_mean"] = rewards.mean().item()
        metrics["reward_std"] = rewards.std().item()
        
        self.step_count += 1
        
        return metrics
    
    def train(self, num_steps: int = 1000):
        """
        Main training loop.
        
        Keep it simple for now - just iterate and train.
        
        Args:
            num_steps: Number of training steps
        """
        self.model.train()
        
        for step, batch in enumerate(self.train_dataloader):
            if step >= num_steps:
                break
            
            metrics = self.train_step(batch)
            
            # Log every 10 steps
            if step % 10 == 0:
                print(f"Step {step}: "
                      f"loss={metrics['loss']:.4f}, "
                      f"reward={metrics['reward_mean']:.4f}, "
                      f"ratio={metrics['ratio_mean']:.4f}")
    
    def save_checkpoint(self, path: str):
        """
        Save training checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "step_count": self.step_count,
            "group_size": self.group_size,
            "clip_range": self.clip_range,
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str):
        """
        Load training checkpoint.
        
        Args:
            path: Path to checkpoint file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step_count = checkpoint["step_count"]
        print(f"Checkpoint loaded from {path} (step {self.step_count})")
