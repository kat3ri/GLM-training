"""
Base trainer for GLM-Image.
"""
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from torch.cuda.amp import GradScaler, autocast
import yaml

from ..models import GLMImageWrapper
from ..utils import (
    init_distributed,
    cleanup_distributed,
    is_main_process,
    get_rank,
    wrap_model_ddp,
    barrier,
    Logger,
    format_time,
)


class BaseTrainer:
    """Base trainer class for GLM-Image."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize base trainer.
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Setup distributed training
        if config["distributed"]["enabled"]:
            self.rank, self.world_size, self.local_rank = init_distributed(
                backend=config["distributed"]["backend"]
            )
            self.device = torch.device(f"cuda:{self.local_rank}")
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0
        
        # Setup random seed
        self._set_seed(config.get("seed", 42))
        
        # Initialize logger
        self.logger = None
        if is_main_process():
            self.logger = Logger(
                log_dir=config["logging"]["log_dir"],
                use_tensorboard=config["logging"]["use_tensorboard"],
                use_wandb=config["logging"]["use_wandb"],
                wandb_project=config["logging"].get("wandb_project"),
                wandb_entity=config["logging"].get("wandb_entity"),
                config=config,
            )
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float("inf")
        
        # Setup model
        self.model = self._build_model()
        
        # Setup optimizer and scheduler
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Setup mixed precision training
        self.use_amp = config["training"]["mixed_precision"] != "no"
        self.scaler = GradScaler() if self.use_amp else None
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config["checkpoint"]["save_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Resume from checkpoint if specified
        if config["checkpoint"].get("resume_from"):
            self._load_checkpoint(config["checkpoint"]["resume_from"])
    
    def _set_seed(self, seed: int):
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    def _build_model(self) -> nn.Module:
        """Build the model. To be implemented by subclasses."""
        raise NotImplementedError
    
    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer."""
        config = self.config["training"]
        
        # Get trainable parameters
        params = self.model.get_trainable_parameters()
        
        if config["optimizer"] == "adamw":
            optimizer = optim.AdamW(
                params,
                lr=config["learning_rate"],
                betas=(config["adam_beta1"], config["adam_beta2"]),
                eps=config["adam_epsilon"],
                weight_decay=config["weight_decay"],
            )
        else:
            raise ValueError(f"Unknown optimizer: {config['optimizer']}")
        
        return optimizer
    
    def _build_scheduler(self):
        """Build learning rate scheduler."""
        config = self.config["training"]
        
        if config["lr_scheduler"] == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR
            
            num_epochs = config["num_epochs"]
            scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=num_epochs,
            )
        elif config["lr_scheduler"] == "linear":
            from torch.optim.lr_scheduler import LinearLR
            
            scheduler = LinearLR(self.optimizer)
        else:
            scheduler = None
        
        return scheduler
    
    def _save_checkpoint(self, filename: str = "checkpoint.pt"):
        """Save checkpoint."""
        if not is_main_process():
            return
        
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_metric": self.best_metric,
            "config": self.config,
        }
        
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
        
        if self.scaler is not None:
            checkpoint["scaler_state_dict"] = self.scaler.state_dict()
        
        save_path = self.checkpoint_dir / filename
        torch.save(checkpoint, save_path)
        
        if self.logger:
            self.logger.info(f"Checkpoint saved to {save_path}")
        
        # Manage checkpoint limit
        self._cleanup_checkpoints()
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond save_total_limit."""
        save_limit = self.config["checkpoint"].get("save_total_limit", 3)
        if save_limit is None:
            return
        
        checkpoints = sorted(
            self.checkpoint_dir.glob("checkpoint_step_*.pt"),
            key=lambda x: int(x.stem.split("_")[-1])
        )
        
        while len(checkpoints) > save_limit:
            oldest = checkpoints.pop(0)
            oldest.unlink()
            if self.logger:
                self.logger.info(f"Removed old checkpoint: {oldest}")
    
    def _load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_metric = checkpoint["best_metric"]
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        
        if "scaler_state_dict" in checkpoint and self.scaler is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])
        
        if self.logger:
            self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
    
    def train_step(self, batch: Dict[str, Any]) -> Dict[str, float]:
        """
        Single training step. To be implemented by subclasses.
        
        Args:
            batch: Training batch
            
        Returns:
            Dictionary of metrics
        """
        raise NotImplementedError
    
    def train(self, train_loader: DataLoader, eval_loader: Optional[DataLoader] = None):
        """
        Main training loop.
        
        Args:
            train_loader: Training data loader
            eval_loader: Evaluation data loader (optional)
        """
        num_epochs = self.config["training"]["num_epochs"]
        log_interval = self.config["logging"]["log_interval"]
        save_interval = self.config["logging"]["save_interval"]
        eval_interval = self.config["logging"].get("eval_interval", 1000)
        
        if self.logger:
            self.logger.info("Starting training...")
            self.logger.info(f"  Num epochs: {num_epochs}")
            self.logger.info(f"  World size: {self.world_size}")
            self.logger.info(f"  Batch size per device: {self.config['training']['batch_size']}")
            self.logger.info(f"  Total batch size: {self.config['training']['batch_size'] * self.world_size}")
        
        start_time = time.time()
        
        for epoch in range(self.epoch, num_epochs):
            self.epoch = epoch
            
            # Set epoch for distributed sampler
            if isinstance(train_loader.sampler, DistributedSampler):
                train_loader.sampler.set_epoch(epoch)
            
            # Training epoch
            self.model.train()
            epoch_metrics = defaultdict(float)
            
            for batch_idx, batch in enumerate(train_loader):
                # Train step
                metrics = self.train_step(batch)
                
                # Accumulate metrics
                for key, value in metrics.items():
                    epoch_metrics[key] += value
                
                self.global_step += 1
                
                # Logging
                if self.global_step % log_interval == 0 and self.logger:
                    elapsed = time.time() - start_time
                    self.logger.info(
                        f"Epoch [{epoch}/{num_epochs}] "
                        f"Step [{self.global_step}] "
                        f"Loss: {metrics.get('loss', 0):.4f} "
                        f"Time: {format_time(elapsed)}"
                    )
                    
                    # Log to tensorboard/wandb
                    self.logger.log_scalars(metrics, self.global_step)
                
                # Save checkpoint
                if self.global_step % save_interval == 0:
                    self._save_checkpoint(f"checkpoint_step_{self.global_step}.pt")
                
                # Evaluation
                if eval_loader is not None and self.global_step % eval_interval == 0:
                    eval_metrics = self.evaluate(eval_loader)
                    if self.logger:
                        self.logger.info(f"Eval metrics: {eval_metrics}")
                        self.logger.log_scalars(
                            {f"eval/{k}": v for k, v in eval_metrics.items()},
                            self.global_step
                        )
            
            # Epoch end
            if self.scheduler is not None:
                self.scheduler.step()
            
            # Log epoch metrics
            if self.logger:
                avg_metrics = {
                    k: v / len(train_loader)
                    for k, v in epoch_metrics.items()
                }
                self.logger.info(f"Epoch {epoch} avg metrics: {avg_metrics}")
        
        # Final save
        self._save_checkpoint("checkpoint_final.pt")
        
        if self.logger:
            self.logger.info("Training completed!")
            self.logger.close()
        
        # Cleanup
        cleanup_distributed()
    
    def evaluate(self, eval_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluation loop. To be implemented by subclasses.
        
        Args:
            eval_loader: Evaluation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        raise NotImplementedError
