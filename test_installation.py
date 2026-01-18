#!/usr/bin/env python3
"""
Test script to verify GLM-training installation and configuration.
Run this script after installation to check if everything is set up correctly.
"""
import sys
import os
from pathlib import Path

def check_imports():
    """Check if all required packages can be imported."""
    print("=" * 80)
    print("Checking Package Imports...")
    print("=" * 80)
    
    required_packages = [
        ("torch", "PyTorch"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("numpy", "NumPy"),
        ("transformers", "Transformers"),
        ("diffusers", "Diffusers"),
    ]
    
    optional_packages = [
        ("lpips", "LPIPS (for perceptual loss)"),
        ("pytesseract", "Pytesseract (for text accuracy)"),
        ("wandb", "Weights & Biases (for logging)"),
    ]
    
    all_good = True
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING (required)")
            all_good = False
    
    print("\nOptional packages:")
    for package, name in optional_packages:
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"○ {name} - Not installed (optional)")
    
    return all_good


def check_glm_training():
    """Check if glm_training package is installed correctly."""
    print("\n" + "=" * 80)
    print("Checking GLM-Training Package...")
    print("=" * 80)
    
    try:
        from glm_training.trainers import RewardTrainer, ARTrainer, DiTTrainer
        print("✓ Trainers module")
    except ImportError as e:
        print(f"✗ Trainers module - {e}")
        return False
    
    try:
        from glm_training.data import T2IDataset, I2IDataset
        print("✓ Data module")
    except ImportError as e:
        print(f"✗ Data module - {e}")
        return False
    
    try:
        from glm_training.rewards import RewardCalculator
        print("✓ Rewards module")
    except ImportError as e:
        print(f"✗ Rewards module - {e}")
        return False
    
    try:
        from glm_training.models import GLMImageWrapper
        print("✓ Models module")
    except ImportError as e:
        print(f"✗ Models module - {e}")
        return False
    
    try:
        from glm_training.utils import init_distributed, Logger
        print("✓ Utils module")
    except ImportError as e:
        print(f"✗ Utils module - {e}")
        return False
    
    return True


def check_configs():
    """Check if configuration files exist and are valid."""
    print("\n" + "=" * 80)
    print("Checking Configuration Files...")
    print("=" * 80)
    
    import yaml
    
    config_files = [
        "configs/t2i_training.yaml",
        "configs/i2i_training.yaml",
    ]
    
    all_good = True
    for config_file in config_files:
        if not Path(config_file).exists():
            print(f"✗ {config_file} - NOT FOUND")
            all_good = False
            continue
        
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
            print(f"✓ {config_file}")
        except Exception as e:
            print(f"✗ {config_file} - INVALID: {e}")
            all_good = False
    
    return all_good


def check_data_structure():
    """Check if data directories exist."""
    print("\n" + "=" * 80)
    print("Checking Data Structure...")
    print("=" * 80)
    
    required_dirs = [
        "data/t2i",
        "data/i2i",
    ]
    
    recommended_files = [
        "data/t2i/prompts.txt",
        "data/t2i/target_images",
        "data/i2i/prompts.txt",
        "data/i2i/source_images",
        "data/i2i/target_images",
    ]
    
    print("Required directories:")
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"✓ {dir_path}/")
        else:
            print(f"✗ {dir_path}/ - NOT FOUND")
    
    print("\nRecommended files/directories (for training):")
    for file_path in recommended_files:
        path = Path(file_path)
        if path.exists():
            if path.is_dir():
                num_files = len(list(path.glob("*")))
                print(f"✓ {file_path}/ ({num_files} files)")
            else:
                with open(file_path) as f:
                    num_lines = len(f.readlines())
                print(f"✓ {file_path} ({num_lines} lines)")
        else:
            print(f"○ {file_path} - Not found (add your data here)")
    
    return True


def check_gpu():
    """Check GPU availability."""
    print("\n" + "=" * 80)
    print("Checking GPU Availability...")
    print("=" * 80)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            print(f"✓ CUDA is available")
            print(f"✓ Number of GPUs: {num_gpus}")
            
            for i in range(num_gpus):
                gpu_name = torch.cuda.get_device_name(i)
                gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
            
            return True
        else:
            print("✗ CUDA is not available")
            print("  Training will be very slow without GPU")
            return False
    except Exception as e:
        print(f"✗ Error checking GPU: {e}")
        return False


def check_scripts():
    """Check if training scripts are executable."""
    print("\n" + "=" * 80)
    print("Checking Training Scripts...")
    print("=" * 80)
    
    scripts = [
        "train.py",
        "train_ar.py",
        "train_dit.py",
    ]
    
    all_good = True
    for script in scripts:
        if Path(script).exists():
            print(f"✓ {script}")
        else:
            print(f"✗ {script} - NOT FOUND")
            all_good = False
    
    return all_good


def main():
    """Run all checks."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "GLM-Training Installation Test" + " " * 28 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    results = {
        "Imports": check_imports(),
        "GLM-Training Package": check_glm_training(),
        "Configuration Files": check_configs(),
        "Data Structure": check_data_structure(),
        "Training Scripts": check_scripts(),
        "GPU": check_gpu(),
    }
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    all_passed = True
    for check_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{check_name}: {status}")
        if not passed and check_name in ["Imports", "GLM-Training Package", "Configuration Files", "Training Scripts"]:
            all_passed = False
    
    print("=" * 80)
    
    if all_passed:
        print("\n✓ All critical checks passed! You're ready to train.")
        print("\nNext steps:")
        print("1. Prepare your training data in data/t2i/ or data/i2i/")
        print("2. Review and customize configs/t2i_training.yaml or configs/i2i_training.yaml")
        print("3. Run: python train.py --config configs/t2i_training.yaml")
        print("\nSee QUICKSTART.md and examples/USAGE.md for more details.")
        return 0
    else:
        print("\n✗ Some critical checks failed. Please fix the issues above.")
        print("\nInstallation instructions:")
        print("pip install -e .")
        print("pip install git+https://github.com/huggingface/transformers.git")
        print("pip install git+https://github.com/huggingface/diffusers.git")
        return 1


if __name__ == "__main__":
    sys.exit(main())
