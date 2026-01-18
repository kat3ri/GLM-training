"""
Logging utilities for training.
"""
import os
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any

import torch
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """Unified logger for TensorBoard and Weights & Biases."""
    
    def __init__(
        self,
        log_dir: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
        wandb_entity: Optional[str] = None,
        wandb_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize logger.
        
        Args:
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_entity: W&B entity name
            wandb_name: W&B run name
            config: Configuration dictionary to log
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # TensorBoard
        self.use_tensorboard = use_tensorboard
        self.tb_writer = None
        if use_tensorboard:
            self.tb_writer = SummaryWriter(log_dir=str(self.log_dir))
            
        # Weights & Biases
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        if self.use_wandb:
            if wandb_name is None:
                wandb_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=wandb_name,
                config=config,
                dir=str(self.log_dir)
            )
    
    def _setup_logging(self):
        """Setup Python logging."""
        log_file = self.log_dir / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def log_scalar(self, tag: str, value: float, step: int):
        """
        Log a scalar value.
        
        Args:
            tag: Name of the scalar
            value: Value to log
            step: Current step
        """
        if self.use_tensorboard:
            self.tb_writer.add_scalar(tag, value, step)
        if self.use_wandb:
            wandb.log({tag: value}, step=step)
    
    def log_scalars(self, scalars: Dict[str, float], step: int):
        """
        Log multiple scalar values.
        
        Args:
            scalars: Dictionary of scalar values
            step: Current step
        """
        for tag, value in scalars.items():
            self.log_scalar(tag, value, step)
    
    def log_image(self, tag: str, image: torch.Tensor, step: int):
        """
        Log an image.
        
        Args:
            tag: Name of the image
            image: Image tensor (C, H, W)
            step: Current step
        """
        if self.use_tensorboard:
            self.tb_writer.add_image(tag, image, step)
        if self.use_wandb:
            # Convert tensor to numpy for wandb
            import numpy as np
            if image.ndim == 3:
                image_np = image.cpu().numpy()
                if image_np.shape[0] == 3:  # CHW format
                    image_np = np.transpose(image_np, (1, 2, 0))
                wandb.log({tag: wandb.Image(image_np)}, step=step)
    
    def log_images(self, tag: str, images: torch.Tensor, step: int):
        """
        Log multiple images.
        
        Args:
            tag: Name prefix for images
            images: Image tensor (B, C, H, W)
            step: Current step
        """
        for i, image in enumerate(images):
            self.log_image(f"{tag}/{i}", image, step)
    
    def log_text(self, tag: str, text: str, step: int):
        """
        Log text.
        
        Args:
            tag: Name of the text
            text: Text to log
            step: Current step
        """
        if self.use_tensorboard:
            self.tb_writer.add_text(tag, text, step)
        if self.use_wandb:
            wandb.log({tag: text}, step=step)
    
    def info(self, message: str):
        """Log info message."""
        self.logger.info(message)
    
    def warning(self, message: str):
        """Log warning message."""
        self.logger.warning(message)
    
    def error(self, message: str):
        """Log error message."""
        self.logger.error(message)
    
    def close(self):
        """Close logger."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()
        if self.use_wandb:
            wandb.finish()


def format_time(seconds: float) -> str:
    """
    Format time in seconds to human-readable format.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_dict(d: Dict[str, Any], indent: int = 0) -> str:
    """
    Format dictionary for pretty printing.
    
    Args:
        d: Dictionary to format
        indent: Indentation level
        
    Returns:
        Formatted string
    """
    lines = []
    for key, value in d.items():
        if isinstance(value, dict):
            lines.append(f"{'  ' * indent}{key}:")
            lines.append(format_dict(value, indent + 1))
        else:
            lines.append(f"{'  ' * indent}{key}: {value}")
    return "\n".join(lines)
