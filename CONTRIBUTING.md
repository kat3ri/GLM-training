# Contributing to GLM-Training

Thank you for your interest in contributing to the GLM-Image training framework!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR_USERNAME/GLM-training.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Run tests (if available)
6. Commit: `git commit -m "Add your feature"`
7. Push: `git push origin feature/your-feature-name`
8. Open a Pull Request

## Development Setup

```bash
# Install in development mode
pip install -e .

# Install development dependencies
pip install black flake8 pytest
```

## Code Style

We follow PEP 8 style guidelines. Format your code with:

```bash
black glm_training/ train.py train_ar.py train_dit.py
```

Check for issues with:

```bash
flake8 glm_training/ train.py train_ar.py train_dit.py
```

## Areas for Contribution

### High Priority

- **Testing**: Add unit tests and integration tests
- **Documentation**: Improve docstrings and examples
- **Performance**: Optimize training speed and memory usage
- **Reward Metrics**: Implement additional reward functions

### Features to Add

- [ ] Support for LoRA training
- [ ] Quantization support (8-bit, 4-bit)
- [ ] Additional data augmentation strategies
- [ ] Better checkpoint management
- [ ] Visualization tools for generated images
- [ ] Integration with more logging platforms
- [ ] Support for other reward models
- [ ] Hyperparameter search utilities

### Bug Fixes

- Report bugs via GitHub Issues
- Include reproduction steps
- Provide system information (GPU, CUDA version, etc.)

## Pull Request Guidelines

1. **Small, focused changes**: One feature or fix per PR
2. **Descriptive title**: Clearly state what the PR does
3. **Documentation**: Update README.md and docstrings as needed
4. **Tests**: Add tests for new features (when test framework exists)
5. **Backward compatibility**: Don't break existing functionality

## Testing Your Changes

Before submitting a PR:

1. Test basic functionality:
```bash
# Test T2I training script loads
python train.py --config configs/t2i_training.yaml --help

# Test AR training script loads
python train_ar.py --config configs/t2i_training.yaml --help

# Test DiT training script loads
python train_dit.py --config configs/t2i_training.yaml --help
```

2. Verify imports work:
```python
from glm_training.trainers import RewardTrainer, ARTrainer, DiTTrainer
from glm_training.data import T2IDataset, I2IDataset
from glm_training.rewards import RewardCalculator
```

3. Check configuration loading:
```python
import yaml
with open("configs/t2i_training.yaml") as f:
    config = yaml.safe_load(f)
```

## Reporting Issues

When reporting issues, include:

1. **Environment**:
   - Python version
   - PyTorch version
   - GPU model and CUDA version
   - Operating system

2. **Issue description**:
   - What you expected to happen
   - What actually happened
   - Steps to reproduce

3. **Code/Configuration**:
   - Minimal code example
   - Configuration file (if relevant)
   - Error messages and stack traces

## Documentation

- Use clear, concise language
- Include code examples
- Update README.md for user-facing changes
- Add docstrings to new functions/classes
- Follow Google-style docstring format

Example:
```python
def my_function(arg1: str, arg2: int) -> bool:
    """
    Brief description of function.
    
    More detailed description if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ValueError: When arg2 is negative
    """
    pass
```

## Community

- Be respectful and inclusive
- Help others in issues and discussions
- Share your use cases and results
- Provide constructive feedback

## License

By contributing, you agree that your contributions will be licensed under the same license as the project.
