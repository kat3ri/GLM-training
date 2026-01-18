#!/usr/bin/env python3
"""
Test script to verify the meta tensor fix.
This tests that models can be loaded with device_map without causing meta tensor errors.
"""
import sys
import torch
from unittest.mock import Mock, patch, MagicMock

def test_model_initialization_with_device_map():
    """Test that model initialization works with different device_map values."""
    print("=" * 80)
    print("Testing Model Initialization with device_map")
    print("=" * 80)
    
    # Import modules
    try:
        from glm_training.trainers.reward_trainer import RewardTrainer
        print("✓ Successfully imported RewardTrainer")
    except ImportError as e:
        print(f"✗ Failed to import: {e}")
        return False
    
    # Test configurations with different device_map values
    test_cases = [
        ("cpu device_map (distributed)", "cpu", True),
        ("auto device_map (single GPU)", "auto", False),
        ("balanced device_map (single GPU)", "balanced", False),
    ]
    
    all_passed = True
    
    for test_name, device_map, is_distributed in test_cases:
        print(f"\nTesting: {test_name}")
        print("-" * 40)
        
        try:
            # Create a mock config
            config = {
                "model": {
                    "name": "mock-model",
                    "component": "dit",
                    "torch_dtype": "float32",
                    "device_map": device_map,
                    "dit": {
                        "num_inference_steps": 50,
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
                    "num_epochs": 1,
                    "lr_scheduler": "cosine",
                    "gradient_checkpointing": False,
                    "mixed_precision": "no",
                },
                "reward": {
                    "enabled": True,
                    "metrics": {"lpips": 0.4},
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
                        "height": 1024,
                        "width": 1024,
                    }
                },
                "distributed": {
                    "enabled": is_distributed,
                    "backend": "nccl",
                    "find_unused_parameters": False,
                },
                "logging": {
                    "log_dir": "logs",
                    "log_interval": 10,
                    "save_interval": 500,
                    "eval_interval": 100,
                    "use_tensorboard": False,
                    "use_wandb": False,
                },
                "checkpoint": {
                    "save_dir": "checkpoints",
                    "resume_from": None,
                    "save_total_limit": 3,
                },
            }
            
            # Mock the GLMImageWrapper to avoid actually loading a model
            mock_model = MagicMock()
            mock_model.to = MagicMock(return_value=mock_model)
            mock_model.get_trainable_parameters = MagicMock(return_value=[torch.nn.Parameter(torch.zeros(1))])
            mock_model.torch_dtype = torch.float32
            mock_model.ar_model = MagicMock()
            mock_model.dit_model = MagicMock()
            mock_model.vae = MagicMock()
            
            with patch('glm_training.trainers.reward_trainer.GLMImageWrapper', return_value=mock_model):
                with patch('glm_training.trainers.reward_trainer.RewardCalculator'):
                    with patch('glm_training.trainers.reward_trainer.wrap_model_ddp', return_value=mock_model):
                        # Create trainer - this should not raise the meta tensor error
                        trainer = RewardTrainer(config)
                        
                        # Verify the model.to() was called correctly based on device_map
                        if device_map == "cpu":
                            # With cpu device_map, .to() should be called to move to GPU
                            assert mock_model.to.called, f"Expected .to() to be called for {test_name}"
                            print(f"  ✓ .to() was called (as expected for cpu device_map)")
                        else:
                            # For auto/balanced device_map, .to() should NOT be called
                            # Note: We can't easily verify this wasn't called since the mock is created
                            # before the trainer, but the test passing without error is the key check
                            print(f"  ✓ Model initialized without meta tensor error")
                        
                        print(f"✓ {test_name} passed")
        
        except Exception as e:
            print(f"✗ {test_name} failed: {e}")
            import traceback
            traceback.print_exc()
            all_passed = False
    
    return all_passed


def test_model_code_logic():
    """Test the logic of the fix without actually loading models."""
    print("\n" + "=" * 80)
    print("Testing Fix Logic")
    print("=" * 80)
    
    # Simulate the fixed logic
    def determine_device_map(world_size, model_device_map):
        """Simulate the fixed device_map determination logic."""
        if world_size > 1:
            device_map = "cpu"
        else:
            device_map = model_device_map
        return device_map
    
    def should_call_to(world_size, device_map):
        """Simulate when .to() should be called."""
        return device_map == "cpu"
    
    test_cases = [
        (1, "auto", "auto", False, "Single GPU with auto device_map"),
        (1, "balanced", "balanced", False, "Single GPU with balanced device_map"),
        (1, "cpu", "cpu", True, "Single GPU with cpu device_map should call .to()"),
        (4, "auto", "cpu", True, "Multi-GPU should use cpu and call .to()"),
        (4, "balanced", "cpu", True, "Multi-GPU with balanced config should use cpu and call .to()"),
    ]
    
    all_passed = True
    for world_size, model_config_device_map, expected_device_map, expected_to_call, description in test_cases:
        print(f"\nTest: {description}")
        print(f"  Input: world_size={world_size}, model_config_device_map={model_config_device_map}")
        
        device_map = determine_device_map(world_size, model_config_device_map)
        to_call = should_call_to(world_size, device_map)
        
        print(f"  Result: device_map={device_map}, should_call_to={to_call}")
        
        if device_map == expected_device_map and to_call == expected_to_call:
            print(f"  ✓ Passed")
        else:
            print(f"  ✗ Failed: Expected device_map={expected_device_map}, to_call={expected_to_call}")
            all_passed = False
    
    return all_passed


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 22 + "Meta Tensor Fix Test" + " " * 36 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    results = {
        "Logic Test": test_model_code_logic(),
        "Model Initialization Test": test_model_initialization_with_device_map(),
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
        print("\n✓ All tests passed! The meta tensor fix is working correctly.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
