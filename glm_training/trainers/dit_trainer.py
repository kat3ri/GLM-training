"""
DiT (Diffusion Decoder) trainer.
"""
from typing import Dict, Any
import torch

from .reward_trainer import RewardTrainer
from ..models import GLMImageWrapper


class DiTTrainer(RewardTrainer):
    """Trainer specifically for the DiT (Diffusion Decoder) component."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize DiT trainer.
        
        Args:
            config: Training configuration
        """
        # Ensure we're training only DiT component
        config["model"]["component"] = "dit"
        super().__init__(config)
        
        # Override learning rate with DiT-specific LR if provided
        if "dit_learning_rate" in config["training"]:
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = config["training"]["dit_learning_rate"]
    
    def _build_optimizer(self):
        """Build optimizer for DiT model."""
        config = self.config["training"]
        
        # Use DiT-specific learning rate if provided
        lr = config.get("dit_learning_rate", config["learning_rate"])
        
        # Get DiT parameters
        params = self.model.get_dit_parameters()
        
        if config["optimizer"] == "adamw":
            optimizer = torch.optim.AdamW(
                params,
                lr=lr,
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_epsilon"],
                weight_decay=config["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
        return optimizer
