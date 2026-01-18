#!/usr/bin/env python3
"""
Training script for Autoregressive model only.
"""
import argparse
import yaml

from torch.utils.data import DataLoader, DistributedSampler

from glm_training.trainers import ARTrainer
from glm_training.data import T2IDataset, I2IDataset, collate_t2i, collate_i2i
from glm_training.utils import is_main_process


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train GLM-Image Autoregressive model"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file"
    )
    
    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def create_dataset(config: dict, mode: str, is_eval: bool = False):
    """Create dataset based on mode."""
    data_config = config["data"]
    
    if mode == "t2i":
        if is_eval and config["evaluation"]["enabled"]:
            dataset = T2IDataset(
                prompts_file=config["evaluation"]["eval_prompts_file"],
                target_images_dir=config["evaluation"]["eval_target_images_dir"],
                image_size=(
                    data_config["image_size"]["height"],
                    data_config["image_size"]["width"]
                ),
            )
        else:
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
        if is_eval and config["evaluation"]["enabled"]:
            dataset = I2IDataset(
                source_images_dir=config["evaluation"]["eval_source_images_dir"],
                prompts_file=config["evaluation"]["eval_prompts_file"],
                target_images_dir=config["evaluation"]["eval_target_images_dir"],
                image_size=(
                    data_config["image_size"]["height"],
                    data_config["image_size"]["width"]
                ),
                augmentation=data_config.get("augmentation"),
            )
        else:
            dataset = I2IDataset(
                source_images_dir=data_config["source_images_dir"],
                prompts_file=data_config["prompts_file"],
                target_images_dir=data_config["target_images_dir"],
                image_size=(
                    data_config["image_size"]["height"],
                    data_config["image_size"]["width"]
                ),
                augmentation=data_config.get("augmentation"),
            )
        collate_fn = collate_i2i
    
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return dataset, collate_fn


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Force AR component
    config["model"]["component"] = "ar"
    
    mode = config["training"]["mode"]
    
    if is_main_process():
        print("=" * 80)
        print("GLM-Image Autoregressive Model Training")
        print("=" * 80)
        print(f"Mode: {mode}")
        print(f"Reward-based training: {config['reward']['enabled']}")
        print(f"Multi-GPU: {config['distributed']['enabled']}")
        print("=" * 80)
    
    # Create datasets
    train_dataset, collate_fn = create_dataset(config, mode, is_eval=False)
    
    # Create data loaders
    if config["distributed"]["enabled"]:
        train_sampler = DistributedSampler(
            train_dataset,
            shuffle=config["data"]["shuffle"]
        )
    else:
        train_sampler = None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=train_sampler,
        shuffle=config["data"]["shuffle"] if train_sampler is None else False,
        num_workers=config["data"]["num_workers"],
        collate_fn=collate_fn,
        pin_memory=config["data"]["pin_memory"],
    )
    
    # Create evaluation loader if enabled
    eval_loader = None
    if config["evaluation"]["enabled"]:
        eval_dataset, eval_collate_fn = create_dataset(config, mode, is_eval=True)
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"]["num_workers"],
            collate_fn=eval_collate_fn,
            pin_memory=config["data"]["pin_memory"],
        )
    
    # Create trainer
    trainer = ARTrainer(config)
    
    # Start training
    trainer.train(train_loader, eval_loader)


if __name__ == "__main__":
    main()
