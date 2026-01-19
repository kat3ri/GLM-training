#!/usr/bin/env python3
"""
GRPO Training Script for GLM-Image.

Minimal training script following the clean implementation plan.
Supports text-to-image (t2i) and image-to-image (i2i) modes.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from glm_training.models import GLMImageWrapper
from glm_training.grpo import MinimalGRPOTrainer, simple_reward_function
from glm_training.data import T2IDataset, I2IDataset, collate_t2i, collate_i2i
from glm_training.rewards import RewardCalculator


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GLM-Image with GRPO"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["t2i", "i2i"],
        default=None,
        help="Training mode (overrides config)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/grpo",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=None,
        help="Number of training steps (overrides config)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log metrics every N steps"
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=500,
        help="Save checkpoint every N steps"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataset(config: Dict[str, Any], mode: str):
    """
    Create dataset based on mode.
    
    Args:
        config: Configuration dictionary
        mode: Training mode ("t2i" or "i2i")
        
    Returns:
        Dataset and collate function
    """
    data_config = config["data"]
    
    if mode == "t2i":
        dataset = T2IDataset(
            prompts_file=data_config["prompts_file"],
            target_images_dir=data_config["target_images_dir"],
            image_size=(
                data_config["image_size"]["height"],
                data_config["image_size"]["width"]
            ),
        )
        collate_fn = collate_t2i
    elif mode == "i2i":
        dataset = I2IDataset(
            prompts_file=data_config["prompts_file"],
            source_images_dir=data_config["source_images_dir"],
            target_images_dir=data_config["target_images_dir"],
            image_size=(
                data_config["image_size"]["height"],
                data_config["image_size"]["width"]
            ),
        )
        collate_fn = collate_i2i
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return dataset, collate_fn


def create_reward_function(config: Dict[str, Any]):
    """
    Create reward function based on config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Reward function
    """
    reward_config = config.get("reward", {})
    
    if reward_config.get("use_advanced", False):
        # Use advanced reward calculator with LPIPS, aesthetic, etc.
        reward_calculator = RewardCalculator(
            lpips_weight=reward_config["metrics"].get("lpips", 0.4),
            aesthetic_weight=reward_config["metrics"].get("aesthetic", 0.3),
            text_accuracy_weight=reward_config["metrics"].get("text_accuracy", 0.3),
        )
        return reward_calculator.compute_rewards
    else:
        # Use simple MSE-based reward
        return simple_reward_function


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Override config with CLI args
    mode = args.mode or config["training"]["mode"]
    num_steps = args.num_steps or config["training"].get("max_steps", 1000)
    
    # Set seed
    set_seed(args.seed)
    logger.info(f"Set random seed to {args.seed}")
    
    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    log_dir = output_dir / "logs"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup TensorBoard
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard logs will be saved to {log_dir}")
    
    # Load model
    logger.info("Loading GLM-Image model...")
    model_config = config["model"]
    model = GLMImageWrapper(
        model_name=model_config["name"],
        component=model_config["component"],
        torch_dtype=model_config.get("torch_dtype", "bfloat16"),
        device_map=model_config.get("device_map", "auto"),
    )
    logger.info(f"Model loaded: {model_config['name']} (component={model_config['component']})")
    
    # Create dataset and dataloader
    logger.info(f"Creating {mode} dataset...")
    dataset, collate_fn = create_dataset(config, mode)
    dataloader = DataLoader(
        dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=config["data"].get("shuffle", True),
        num_workers=config["data"].get("num_workers", 4),
        pin_memory=config["data"].get("pin_memory", True),
        collate_fn=collate_fn,
    )
    logger.info(f"Dataset created with {len(dataset)} samples")
    
    # Create reward function
    logger.info("Creating reward function...")
    reward_function = create_reward_function(config)
    
    # Create GRPO trainer
    logger.info("Creating GRPO trainer...")
    grpo_config = config["reward"]["grpo"]
    trainer = MinimalGRPOTrainer(
        model=model,
        train_dataloader=dataloader,
        reward_function=reward_function,
        group_size=grpo_config.get("num_samples", 4),
        clip_range=grpo_config.get("clip_range", 0.2),
        learning_rate=config["training"]["learning_rate"],
        max_grad_norm=config["training"].get("max_grad_norm", 1.0),
        height=config["data"]["image_size"]["height"],
        width=config["data"]["image_size"]["width"],
        num_inference_steps=model_config["dit"].get("num_inference_steps", 50),
        guidance_scale=model_config["dit"].get("guidance_scale", 1.5),
    )
    logger.info(f"Trainer created (group_size={grpo_config['num_samples']})")
    
    # Resume from checkpoint if specified
    start_step = 0
    if args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location="cpu")
        trainer.load_checkpoint(args.resume)
        start_step = checkpoint.get("step", 0)
        logger.info(f"Resumed from step {start_step}")
    
    # Training loop
    logger.info("=" * 80)
    logger.info(f"Starting GRPO training ({mode} mode)")
    logger.info(f"Total steps: {num_steps}")
    logger.info(f"Group size: {grpo_config['num_samples']}")
    logger.info(f"Learning rate: {config['training']['learning_rate']}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 80)
    
    try:
        step = start_step
        while step < num_steps:
            for batch in dataloader:
                if step >= num_steps:
                    break
                
                # Extract data based on mode
                if mode == "t2i":
                    prompts = batch["prompts"]
                    targets = batch["target_images"]
                    source_images = None
                elif mode == "i2i":
                    prompts = batch["prompts"]
                    targets = batch["target_images"]
                    source_images = batch["source_images"]
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                
                # Training step
                metrics = trainer.train_step(batch)
                
                # Log metrics
                if step % args.log_interval == 0:
                    logger.info(
                        f"Step {step}/{num_steps} | "
                        f"Loss: {metrics['loss']:.4f} | "
                        f"Reward: {metrics['reward_mean']:.4f} | "
                        f"Ratio Mean: {metrics['ratio_mean']:.4f} | "
                        f"Clipped: {metrics['clip_fraction']:.2%}"
                    )
                    
                    # Write to TensorBoard
                    for key, value in metrics.items():
                        writer.add_scalar(f"train/{key}", value, step)
                
                # Save checkpoint
                if step % args.save_interval == 0 and step > 0:
                    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pt"
                    trainer.save_checkpoint(str(checkpoint_path))
                    logger.info(f"Checkpoint saved: {checkpoint_path}")
                
                step += 1
        
        # Save final checkpoint
        final_checkpoint_path = checkpoint_dir / "checkpoint_final.pt"
        trainer.save_checkpoint(str(final_checkpoint_path))
        logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
        
        logger.info("=" * 80)
        logger.info("Training completed successfully!")
        logger.info(f"Total steps: {step}")
        logger.info(f"Checkpoints saved to: {checkpoint_dir}")
        logger.info(f"Logs saved to: {log_dir}")
        logger.info("=" * 80)
        
    except KeyboardInterrupt:
        logger.info("\nTraining interrupted by user")
        
        # Save checkpoint on interrupt
        interrupt_checkpoint_path = checkpoint_dir / "checkpoint_interrupt.pt"
        trainer.save_checkpoint(str(interrupt_checkpoint_path))
        logger.info(f"Checkpoint saved on interrupt: {interrupt_checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise
    
    finally:
        # Close TensorBoard writer
        writer.close()


if __name__ == "__main__":
    main()
