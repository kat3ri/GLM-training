#!/usr/bin/env python3
"""
Quick start example for GRPO training.

This is a minimal example to get started with GRPO training quickly.
"""
import torch
from torch.utils.data import DataLoader
from PIL import Image

from glm_training.models import GLMImageWrapper
from glm_training.grpo import MinimalGRPOTrainer, simple_reward_function
from glm_training.data import T2IDataset, collate_t2i


def main():
    """Quick start example."""
    print("=" * 80)
    print("GRPO Training Quick Start Example")
    print("=" * 80)
    
    # 1. Load GLM-Image model
    print("\n[1/5] Loading GLM-Image model...")
    model = GLMImageWrapper(
        model_name="zai-org/GLM-Image",
        component="both",  # Train both AR and DiT
        torch_dtype="bfloat16",
        device_map="auto",
    )
    print("✓ Model loaded")
    
    # 2. Create dataset
    print("\n[2/5] Creating dataset...")
    dataset = T2IDataset(
        prompts_file="data/t2i/prompts.txt",
        target_images_dir="data/t2i/target_images",
        image_size=(1024, 1024),
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Must be 1 for AR model
        shuffle=True,
        num_workers=4,
        collate_fn=collate_t2i,
    )
    print(f"✓ Dataset created ({len(dataset)} samples)")
    
    # 3. Create GRPO trainer
    print("\n[3/5] Creating GRPO trainer...")
    trainer = MinimalGRPOTrainer(
        model=model,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,  # Generate 4 samples per prompt
        clip_range=0.2,  # PPO clipping
        learning_rate=5e-6,
        max_grad_norm=1.0,
        height=1024,
        width=1024,
        num_inference_steps=50,
        guidance_scale=1.5,
    )
    print("✓ Trainer created")
    
    # 4. Train for a few steps
    print("\n[4/5] Training...")
    print("-" * 80)
    
    num_steps = 10  # Quick test
    
    for step, batch in enumerate(dataloader):
        if step >= num_steps:
            break
        
        # Training step
        metrics = trainer.train_step(batch)
        
        # Log
        print(
            f"Step {step + 1}/{num_steps} | "
            f"Loss: {metrics['loss']:.4f} | "
            f"Reward: {metrics['reward_mean']:.4f} | "
            f"Ratio: {metrics['ratio_mean']:.4f} | "
            f"Clipped: {metrics['clip_fraction']:.2%}"
        )
    
    print("-" * 80)
    print("✓ Training complete")
    
    # 5. Save checkpoint
    print("\n[5/5] Saving checkpoint...")
    trainer.save_checkpoint("outputs/grpo_quickstart.pt")
    print("✓ Checkpoint saved to outputs/grpo_quickstart.pt")
    
    print("\n" + "=" * 80)
    print("Quick start example completed successfully!")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Check outputs/grpo_quickstart.pt for the saved checkpoint")
    print("2. Use train_grpo.py for full training with more options")
    print("3. Customize reward function for your use case")
    print("=" * 80)


if __name__ == "__main__":
    main()
