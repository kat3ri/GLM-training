"""
Minimal GRPO trainer - starting from scratch.

Based on nanoGRPO + TRL patterns, but simplified.
Day 2: Trainer skeleton with placeholders.
"""
import torch
import torch.nn.functional as F
from typing import Dict, Any, List, Optional, Callable
from PIL import Image
import numpy as np

from .loss import compute_advantages, compute_grpo_loss


class MinimalGRPOTrainer:
    """
    Minimal GRPO trainer - just the essentials.
    
    Start simple, add features later.
    
    Args:
        model: GLM-Image model (or wrapper)
        tokenizer: Tokenizer for text processing
        train_dataloader: Training data iterator
        reward_function: Function that takes (images, targets) -> rewards
        group_size: Samples per prompt (default: 4)
        clip_range: PPO clipping epsilon (default: 0.2)
        learning_rate: Learning rate (default: 5e-6)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        train_dataloader,
        reward_function: Callable,
        group_size: int = 4,
        clip_range: float = 0.2,
        learning_rate: float = 5e-6,
    ):
        """Initialize minimal GRPO trainer."""
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.reward_function = reward_function
        self.group_size = group_size
        self.clip_range = clip_range
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        self.device = next(model.parameters()).device
        self.step_count = 0
        
    def generate_samples(
        self, 
        prompts: List[str]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate multiple samples per prompt.
        
        Args:
            prompts: List of text prompts
            
        Returns:
            images: [batch_size * group_size, C, H, W]
            old_logprobs: [batch_size * group_size]
            token_ids: [batch_size * group_size, seq_len] - stored for recomputation
        """
        # Repeat prompts for group sampling
        repeated_prompts = []
        for prompt in prompts:
            repeated_prompts.extend([prompt] * self.group_size)
        
        # Tokenize
        inputs = self.tokenizer(
            repeated_prompts,
            return_tensors="pt",
            padding=True,
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate (no gradients)
        with torch.no_grad():
            # Generate token IDs
            # TODO: Replace with actual GLM-Image generation
            generated_ids = self.model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
            )
            
            # Compute old log probs (before training)
            old_logprobs = self._compute_logprobs(generated_ids)
            
            # Convert to images
            # TODO: Replace with actual GLM-Image decoding
            images = self._ids_to_images(generated_ids)
        
        return images, old_logprobs, generated_ids
    
    def _compute_logprobs(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute sequence log probabilities.
        
        This is simplified - real version needs proper masking.
        
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            log_probs: Sequence log probabilities [batch_size]
        """
        outputs = self.model(input_ids)
        logits = outputs.logits
        
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]
        
        # Compute log probs
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, -1, shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mean over sequence
        return token_log_probs.mean(dim=-1)
    
    def _ids_to_images(self, generated_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert generated token IDs to images.
        
        This is GLM-Image specific - placeholder for now.
        TODO: Implement actual GLM-Image decoding pipeline
        
        Args:
            generated_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            images: Image tensors [batch_size, C, H, W]
        """
        # Placeholder: return dummy images
        batch_size = generated_ids.size(0)
        return torch.randn(batch_size, 3, 1024, 1024, device=self.device)
    
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
        
        # Step 1: Generate samples (no gradients)
        images, old_logprobs, token_ids = self.generate_samples(prompts)
        
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
        new_logprobs = self._compute_logprobs(token_ids)
        
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
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
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
