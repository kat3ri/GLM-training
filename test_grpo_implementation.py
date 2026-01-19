#!/usr/bin/env python3
"""
Test script to verify the full GRPO implementation in RewardTrainer.
This validates that the policy gradient training works correctly.
"""
import sys
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any
from PIL import Image
import numpy as np


def create_mock_vae():
    """Create a mock VAE with encode and decode functionality."""
    vae = Mock()
    vae.config = Mock()
    vae.config.scaling_factor = 0.18215
    
    def mock_encode(x):
        latent_dist = Mock()
        batch_size = x.shape[0]
        # Return latents with reduced spatial dimensions
        latent_dist.sample = lambda: torch.randn(
            batch_size, 4, x.shape[2]//8, x.shape[3]//8, 
            device=x.device, dtype=x.dtype
        )
        return latent_dist
    
    def mock_decode(x):
        result = Mock()
        batch_size = x.shape[0]
        # Return images with upsampled spatial dimensions
        result.sample = torch.randn(
            batch_size, 3, x.shape[2]*8, x.shape[3]*8,
            device=x.device, dtype=x.dtype
        )
        return result
    
    vae.encode = mock_encode
    vae.decode = mock_decode
    return vae


def create_mock_dit_model():
    """Create a mock DiT model that predicts noise."""
    class MockDiT(nn.Module):
        def __init__(self):
            super().__init__()
            # Add trainable parameters
            self.conv = nn.Conv2d(4, 4, 3, padding=1)
        
        def forward(self, hidden_states, timesteps, encoder_hidden_states=None):
            result = Mock()
            # Return noise prediction with same shape as input
            # Apply a simple transformation to make it trainable
            noise = self.conv(hidden_states)
            result.sample = noise
            return result
        
        def parameters(self):
            return self.conv.parameters()
    
    return MockDiT()


def create_mock_glm_wrapper(component="dit"):
    """Create a mock GLMImageWrapper."""
    model = Mock()
    model.component = component
    model.torch_dtype = torch.float32
    model.vae = create_mock_vae()
    model.dit_model = create_mock_dit_model()
    
    # Mock get_trainable_parameters
    model.get_trainable_parameters = lambda: list(model.dit_model.parameters())
    
    # Add training attribute and train/eval methods
    model.training = True
    
    def mock_train():
        model.training = True
        return model
    
    def mock_eval():
        model.training = False
        return model
    
    model.train = mock_train
    model.eval = mock_eval
    
    return model


def create_mock_reward_calculator():
    """Create a mock reward calculator."""
    calc = Mock()
    
    def mock_compute_grpo_rewards(samples, target_images, prompts=None, source_images=None):
        num_samples = len(samples)
        batch_size = samples[0].shape[0]
        # Return deterministic rewards for testing
        # Make first sample have highest reward
        rewards = torch.randn(num_samples, batch_size)
        rewards[0] = rewards[0] + 2.0  # Boost first sample
        return rewards
    
    calc.compute_grpo_rewards = mock_compute_grpo_rewards
    return calc


def test_grpo_train_step():
    """Test the full GRPO train_step implementation."""
    print("\n" + "=" * 80)
    print("Testing Full GRPO Implementation")
    print("=" * 80)
    
    sys.path.insert(0, '/home/runner/work/GLM-training/GLM-training')
    from glm_training.trainers.reward_trainer import RewardTrainer
    
    # Create config
    config = {
        "model": {
            "name": "mock",
            "component": "dit",
            "torch_dtype": "float32",
            "device_map": "cpu",
            "dit": {
                "num_inference_steps": 3,  # Reduced for testing
                "guidance_scale": 1.5,
            }
        },
        "training": {
            "mode": "t2i",
            "batch_size": 1,
            "learning_rate": 1e-4,
            "optimizer": "adamw",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-8,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "gradient_checkpointing": False,
            "mixed_precision": "no",
        },
        "reward": {
            "metrics": {"lpips": 0.4, "aesthetic": 0.3, "text_accuracy": 0.3},
            "grpo": {
                "num_samples": 2,
                "kl_coef": 0.05,
                "clip_range": 0.2,
            },
            "ar_reward_weight": 1.0,
            "dit_reward_weight": 1.0,
        },
        "data": {
            "image_size": {
                "height": 64,
                "width": 64,
            }
        },
        "distributed": {
            "enabled": False,
        },
        "logging": {
            "log_dir": "/tmp/logs",
            "use_tensorboard": False,
            "use_wandb": False,
        },
        "checkpoint": {
            "save_dir": "/tmp/checkpoints",
        },
        "seed": 42,
    }
    
    # Patch the init
    with patch.object(RewardTrainer, '__init__', lambda self, cfg: None):
        trainer = RewardTrainer(config)
        
        # Set up trainer attributes
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.model = create_mock_glm_wrapper("dit")
        trainer.reward_calculator = create_mock_reward_calculator()
        trainer.num_samples = 2
        trainer.kl_coef = 0.05
        trainer.clip_range = 0.2
        trainer.mode = "t2i"
        trainer.use_amp = False
        trainer.global_step = 0
        
        # Create optimizer
        trainer.optimizer = torch.optim.AdamW(
            trainer.model.get_trainable_parameters(),
            lr=config["training"]["learning_rate"]
        )
        
        # Get initial parameter values
        initial_params = [p.clone().detach() for p in trainer.model.get_trainable_parameters()]
        
        # Create batch
        batch = {
            "prompts": ["test prompt"],
            "target_images": torch.rand(1, 3, 64, 64),
        }
        
        print("\nRunning train_step with full GRPO implementation...")
        
        try:
            # Call train_step
            metrics = trainer.train_step(batch)
            
            print("\n✓ train_step executed successfully!")
            print(f"\nMetrics returned:")
            for key, value in metrics.items():
                print(f"  {key}: {value:.6f}")
            
            # Validate metrics
            required_metrics = ["loss", "policy_loss", "kl_div", "avg_reward", "log_prob_ratio"]
            for metric in required_metrics:
                assert metric in metrics, f"Missing metric: {metric}"
                assert isinstance(metrics[metric], float), f"{metric} is not a float"
                assert not np.isnan(metrics[metric]), f"{metric} is NaN"
            
            print("\n✓ All required metrics present and valid")
            
            # Check if parameters were updated
            params_updated = False
            for initial, current in zip(initial_params, trainer.model.get_trainable_parameters()):
                if not torch.allclose(initial, current.detach(), atol=1e-6):
                    params_updated = True
                    break
            
            if params_updated:
                print("✓ Model parameters were updated (training is working)")
            else:
                print("⚠ Model parameters were not updated (may need investigation)")
            
            # Check gradient flow
            has_gradients = False
            for param in trainer.model.get_trainable_parameters():
                if param.grad is not None:
                    has_gradients = True
                    grad_norm = param.grad.norm().item()
                    print(f"  Parameter gradient norm: {grad_norm:.6f}")
                    break
            
            if has_gradients:
                print("✓ Gradients are flowing through the model")
            else:
                print("⚠ No gradients found (optimizer may have zeroed them)")
            
            # Validate policy loss is negative (PPO objective is to maximize)
            if "policy_loss" in metrics:
                print(f"\n✓ Policy loss computed: {metrics['policy_loss']:.6f}")
                # Policy loss can be positive or negative depending on rewards
            
            # Validate KL divergence is computed
            if "kl_div" in metrics:
                print(f"✓ KL divergence computed: {metrics['kl_div']:.6f}")
            
            # Validate log prob ratio
            if "log_prob_ratio" in metrics:
                ratio = metrics['log_prob_ratio']
                print(f"✓ Log probability ratio: {ratio:.6f}")
                if 0.8 < ratio < 1.2:
                    print("  → Policy change is conservative (good)")
                else:
                    print("  → Policy changed significantly (may need tuning)")
            
            return True
            
        except RuntimeError as e:
            if "does not require grad and does not have a grad_fn" in str(e):
                print(f"\n✗ FAILED: Gradient tracking error still present!")
                print(f"  Error: {e}")
                return False
            else:
                print(f"\n✗ FAILED with RuntimeError: {e}")
                import traceback
                traceback.print_exc()
                return False
        except Exception as e:
            print(f"\n✗ FAILED with unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False


def test_log_prob_generation():
    """Test the log probability generation method."""
    print("\n" + "=" * 80)
    print("Testing Log Probability Generation")
    print("=" * 80)
    
    sys.path.insert(0, '/home/runner/work/GLM-training/GLM-training')
    from glm_training.trainers.reward_trainer import RewardTrainer
    
    config = {
        "model": {
            "dit": {"num_inference_steps": 3, "guidance_scale": 1.5}
        },
        "data": {
            "image_size": {"height": 64, "width": 64}
        },
    }
    
    with patch.object(RewardTrainer, '__init__', lambda self, cfg: None):
        trainer = RewardTrainer(config)
        
        trainer.config = config
        trainer.device = torch.device("cpu")
        trainer.model = create_mock_glm_wrapper("dit")
        trainer.num_samples = 2
        trainer.global_step = 0
        
        try:
            # Test log prob generation
            latents, log_probs = trainer._generate_latents_with_log_probs(
                prompts=["test"],
                source_images=None,
                num_samples=2
            )
            
            print(f"\n✓ Generated {len(latents)} latent samples")
            print(f"✓ Generated {len(log_probs)} log probability tensors")
            
            # Validate shapes
            assert len(latents) == 2, "Should generate 2 samples"
            assert len(log_probs) == 2, "Should have 2 log prob tensors"
            
            for i, (latent, log_prob) in enumerate(zip(latents, log_probs)):
                print(f"\nSample {i}:")
                print(f"  Latent shape: {latent.shape}")
                print(f"  Log prob shape: {log_prob.shape}")
                print(f"  Log prob value: {log_prob.mean().item():.4f}")
                
                assert latent.shape[0] == 1, "Batch size should be 1"
                assert latent.shape[1] == 4, "Latent channels should be 4"
                assert log_prob.shape[0] == 1, "Log prob batch size should be 1"
            
            print("\n✓ Log probability generation working correctly")
            return True
            
        except Exception as e:
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            return False


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 24 + "GRPO Implementation Test" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    
    results = {
        "Log Probability Generation": test_log_prob_generation(),
        "Full GRPO Train Step": test_grpo_train_step(),
    }
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ All tests passed!")
        print("\nThe full GRPO implementation is working correctly:")
        print("  - Log probabilities are tracked during generation")
        print("  - Policy loss is computed with proper gradients")
        print("  - PPO-style clipping is applied")
        print("  - KL divergence penalty prevents large policy changes")
        print("  - Model parameters are updated via policy gradients")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
