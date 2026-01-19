"""
Reward-based trainer using GRPO (Group Relative Policy Optimization).
"""
from typing import Dict, Any, List, Optional
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from .base_trainer import BaseTrainer
from ..models import GLMImageWrapper
from ..rewards import RewardCalculator
from ..utils import wrap_model_ddp


class RewardTrainer(BaseTrainer):
    """Trainer with reward-based optimization using GRPO."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize reward trainer.
        
        Args:
            config: Training configuration
        """
        super().__init__(config)
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(
            metrics=config["reward"]["metrics"],
            device=self.device,
        )
        
        # GRPO settings
        self.num_samples = config["reward"]["grpo"]["num_samples"]
        self.kl_coef = config["reward"]["grpo"]["kl_coef"]
        self.clip_range = config["reward"]["grpo"]["clip_range"]
        
        # Component-specific reward weights
        self.ar_reward_weight = config["reward"]["ar_reward_weight"]
        self.dit_reward_weight = config["reward"]["dit_reward_weight"]
        
        # Training mode
        self.mode = config["training"]["mode"]
    
    def _build_model(self) -> GLMImageWrapper:
        """Build GLM-Image model."""
        model_config = self.config["model"]
        
        # Convert dtype string to torch dtype
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        torch_dtype = dtype_map.get(
            model_config["torch_dtype"],
            torch.bfloat16
        )
        
        model = GLMImageWrapper(
            model_name=model_config["name"],
            component=model_config["component"],
            torch_dtype=torch_dtype,
            device_map="cpu" if self.world_size > 1 else model_config["device_map"],
        )
        
        model = model.to(self.device)
        
        # Enable gradient checkpointing if specified
        if self.config["training"]["gradient_checkpointing"]:
            if hasattr(model.ar_model, "gradient_checkpointing_enable"):
                model.ar_model.gradient_checkpointing_enable()
            if hasattr(model.dit_model, "enable_gradient_checkpointing"):
                model.dit_model.enable_gradient_checkpointing()
        
        # Wrap with DDP if distributed
        if self.world_size > 1:
            model = wrap_model_ddp(
                model,
                device_ids=[self.local_rank],
                find_unused_parameters=self.config["distributed"]["find_unused_parameters"],
            )
        
        return model
    
    def _generate_samples(
        self,
        prompts: List[str],
        source_images: Optional[List[Image.Image]] = None,
    ) -> List[torch.Tensor]:
        """
        Generate multiple samples for GRPO.
        
        Args:
            prompts: Text prompts
            source_images: Source images for i2i
            
        Returns:
            List of generated image tensors
        """
        samples = []
        
        image_config = self.config["data"]["image_size"]
        height = image_config["height"]
        width = image_config["width"]
        
        dit_config = self.config["model"]["dit"]
        
        for i in range(self.num_samples):
            # Generate with different random seeds
            generator = torch.Generator(device=self.device).manual_seed(
                self.global_step * self.num_samples + i
            )
            
            # Generate images
            generated_images = self.model.generate(
                prompts=prompts,
                images=source_images,
                height=height,
                width=width,
                num_inference_steps=dit_config["num_inference_steps"],
                guidance_scale=dit_config["guidance_scale"],
                generator=generator,
            )
            
            # Convert PIL images to tensors
            image_tensors = []
            for img in generated_images:
                img_array = np.array(img).astype(np.float32) / 255.0
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
                image_tensors.append(img_tensor)
            
            samples.append(torch.stack(image_tensors).to(self.device))
        
        return samples
    
    def _compute_policy_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute GRPO policy loss with PPO-style clipping.
        
        Args:
            rewards: Relative rewards (num_samples, batch_size)
            log_probs: Current policy log probabilities
            old_log_probs: Old policy log probabilities
            
        Returns:
            Policy loss
        """
        # Compute probability ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Clipped surrogate objective
        clipped_ratio = torch.clamp(
            ratio,
            1.0 - self.clip_range,
            1.0 + self.clip_range,
        )
        
        # Policy loss
        policy_loss = -torch.min(
            ratio * rewards,
            clipped_ratio * rewards,
        ).mean()
        
        return policy_loss
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step with reward-based optimization.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of metrics
        """
        prompts = batch["prompts"]
        target_images = batch["target_images"].to(self.device)
        
        # Get source images for i2i mode
        source_images = None
        if self.mode == "i2i" and "source_images" in batch:
            source_images_tensor = batch["source_images"].to(self.device)
            # Convert to PIL images for generation
            source_images = []
            for img_tensor in source_images_tensor:
                img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                source_images.append(Image.fromarray(img_np))
        
        self.optimizer.zero_grad()
        
        # Generate multiple samples for GRPO
        with torch.no_grad():
            samples = self._generate_samples(prompts, source_images)
        
        # Compute rewards for all samples
        rewards = self.reward_calculator.compute_grpo_rewards(
            samples=samples,
            target_images=target_images,
            prompts=prompts,
            source_images=source_images_tensor if source_images else None,
        )
        
        # For now, we'll use a simplified training approach
        # In a full implementation, you'd need to:
        # 1. Store old policy log probabilities
        # 2. Compute new policy log probabilities
        # 3. Use GRPO/PPO update rule
        
        # Simplified loss: use best sample and compute reconstruction loss
        best_sample_idx = rewards.mean(dim=1).argmax()
        best_sample = samples[best_sample_idx]
        
        # Reconstruction loss between best sample and target
        recon_loss = F.mse_loss(best_sample, target_images)
        
        # Add reward as negative loss (we want to maximize reward)
        reward_loss = -rewards[best_sample_idx].mean()
        
        # Combined loss
        loss = recon_loss + reward_loss
        
        # Backward pass
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                self.config["training"]["max_grad_norm"]
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.get_trainable_parameters(),
                self.config["training"]["max_grad_norm"]
            )
            self.optimizer.step()
        
        # Metrics
        metrics = {
            "loss": loss.item(),
            "recon_loss": recon_loss.item(),
            "reward": -reward_loss.item(),
            "avg_reward": rewards.mean().item(),
        }
        
        return metrics
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate the model.
        
        Args:
            eval_loader: Evaluation data loader
            
        Returns:
            Evaluation metrics
        """
        self.model.eval()
        
        total_reward = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in eval_loader:
                prompts = batch["prompts"]
                target_images = batch["target_images"].to(self.device)
                
                source_images = None
                if self.mode == "i2i" and "source_images" in batch:
                    source_images_tensor = batch["source_images"].to(self.device)
                    source_images = []
                    for img_tensor in source_images_tensor:
                        img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        source_images.append(Image.fromarray(img_np))
                
                # Generate samples
                samples = self._generate_samples(prompts, source_images)
                
                # Compute rewards
                rewards = self.reward_calculator.compute_grpo_rewards(
                    samples=samples,
                    target_images=target_images,
                    prompts=prompts,
                    source_images=source_images_tensor if source_images else None,
                )
                
                total_reward += rewards.mean().item()
                num_samples += 1
        
        self.model.train()
        
        return {
            "avg_reward": total_reward / num_samples if num_samples > 0 else 0.0,
        }
