# GitHub Copilot Instructions for GLM-Training

## Project Overview

This is a training framework for GLM-Image, a hybrid autoregressive + diffusion decoder architecture for image generation. The framework supports:
- Reward-based training using GRPO (Group Relative Policy Optimization)
- Multi-GPU distributed training with PyTorch DDP
- Separate or joint training of autoregressive (9B params) and DiT decoder (7B params) components
- Text-to-Image (t2i) and Image-to-Image (i2i) modes

## Architecture Components

- **Autoregressive Generator (AR)**: 9B-parameter model for generating compact visual token encodings
- **Diffusion Decoder (DiT)**: 7B-parameter DiT architecture for latent-space image decoding with Glyph Encoder
- **Reward System**: Computes rewards using LPIPS, SSIM, aesthetic scores, and OCR-based text accuracy

## Code Style and Conventions

### Python Style
- Follow **PEP 8** style guidelines strictly
- Use **black** for code formatting: `black glm_training/ train.py train_ar.py train_dit.py`
- Run **flake8** for linting: `flake8 glm_training/ train.py train_ar.py train_dit.py`
- Use type hints for function parameters and return values

### Documentation
- Use **Google-style docstrings** for all functions and classes
- Include Args, Returns, and Raises sections
- Example format:
  ```python
  def function_name(arg1: str, arg2: int) -> bool:
      """
      Brief description of function.
      
      More detailed description if needed.
      
      Args:
          arg1: Description of arg1
          arg2: Description of arg2
          
      Returns:
          Description of return value
          
      Raises:
          ValueError: When conditions are not met
      """
  ```

### Import Organization
- Standard library imports first
- Third-party imports second (torch, transformers, etc.)
- Local imports last (glm_training.*)
- Use absolute imports for glm_training modules

## Project Structure

```
glm_training/
├── trainers/         # Training logic (base_trainer, ar_trainer, dit_trainer, reward_trainer)
├── data/            # Data loading utilities (t2i_dataset, i2i_dataset)
├── models/          # Model wrappers (glm_wrapper)
├── rewards/         # Reward computation (reward_calculator)
└── utils/           # Utility functions (distributed, logging)
```

## Training Scripts

- **train.py**: Main script for training both components or end-to-end training
- **train_ar.py**: Autoregressive-only training
- **train_dit.py**: DiT decoder-only training

## Configuration System

- All configs are YAML-based in `configs/` directory
- Key config sections: model, training, distributed, reward
- Always validate config loading before making changes to config handling

## Testing Practices

### Running Tests
- Test files follow pattern: `test_*.py`
- Key tests: `test_grpo_implementation.py`, `test_gradient_fix.py`, `test_distributed_fix.py`, `test_installation.py`
- Run tests before committing changes

### Testing Checklist
- Test T2I and I2I modes separately
- Verify multi-GPU functionality if touching distributed code
- Check both AR and DiT training paths
- Validate reward computation changes with unit tests

## Development Workflow

### Setup
```bash
pip install -e .
pip install black flake8 pytest
```

### Before Committing
1. Format code: `black glm_training/ train.py train_ar.py train_dit.py`
2. Check linting: `flake8 glm_training/ train.py train_ar.py train_dit.py`
3. Run relevant tests
4. Verify imports work correctly
5. Test config loading if changed

## Common Patterns

### Distributed Training
- Use `torch.distributed` for DDP setup
- Check `utils/distributed.py` for helper functions
- Always handle rank 0 logging separately
- Use `torch.nn.parallel.DistributedDataParallel` for model wrapping

### Reward Computation
- Rewards are computed by comparing generated vs target images
- Key metrics: LPIPS (perceptual similarity), SSIM, aesthetic scores, OCR text accuracy
- Extend `RewardCalculator` class for custom rewards

### Model Loading
- Models are loaded from "zai-org/GLM-Image" on HuggingFace
- Support separate loading of AR and DiT components
- Use `safetensors` for checkpoint saving/loading

### Memory Management
- Support gradient checkpointing via config: `training.gradient_checkpointing: true`
- Use mixed precision training: `training.mixed_precision: "bf16"`
- Minimum 40GB VRAM for AR, 50GB for DiT, 80GB for both

## Dependencies

### Core Dependencies
- PyTorch >= 2.0.0 (with CUDA support)
- transformers (git version from HuggingFace)
- diffusers (git version from HuggingFace)
- accelerate >= 0.20.0
- peft >= 0.7.0 (for LoRA support)

### Reward/Metric Dependencies
- lpips >= 0.1.4
- scikit-image >= 0.21.0
- pytesseract >= 0.3.10
- opencv-python >= 4.8.0

### Optional Dependencies
- bitsandbytes (may not work on Windows)
- Listed in requirements-optional.txt

## Hardware Considerations

- **Minimum**: Single GPU with 80GB VRAM (A100)
- **Recommended**: 4x GPUs with 80GB VRAM for multi-GPU training
- AR only: ~40GB VRAM
- DiT only: ~50GB VRAM
- Both components: ~80GB VRAM

## Common Tasks

### Adding New Reward Metrics
1. Extend `RewardCalculator` in `glm_training/rewards/reward_calculator.py`
2. Add metric weight to config schema
3. Update documentation
4. Add unit tests

### Supporting New Training Modes
1. Create new trainer class in `glm_training/trainers/`
2. Inherit from `BaseTrainer`
3. Implement required methods
4. Add configuration support
5. Update training scripts

### Modifying Data Loading
1. Update dataset classes in `glm_training/data/`
2. Handle both t2i and i2i modes
3. Ensure proper distributed sampling
4. Test with various batch sizes

## Error Handling

- Use descriptive error messages
- Log errors with appropriate severity levels
- Handle CUDA OOM gracefully with suggestions
- Validate config parameters early
- Provide helpful suggestions for common issues (e.g., Windows compatibility)

## Performance Optimization

- Use gradient accumulation for large batch sizes
- Enable gradient checkpointing for memory efficiency
- Use mixed precision (bf16) by default
- Profile code for bottlenecks before optimizing
- Consider data loading parallelism

## Security Considerations

- Never commit API keys or tokens
- Validate user-provided file paths
- Sanitize config inputs
- Be cautious with eval() or exec()

## Backward Compatibility

- Don't break existing config files
- Maintain compatibility with existing checkpoints
- Add deprecation warnings before removing features
- Document breaking changes clearly

## Best Practices for Code Changes

1. Make minimal, focused changes
2. Update documentation alongside code
3. Add tests for new features
4. Follow existing patterns in the codebase
5. Consider memory and performance impact
6. Test on both single-GPU and multi-GPU setups
7. Verify both t2i and i2i modes still work
8. Check Windows compatibility for non-optional features
