#!/usr/bin/env python3
"""
Simple test to verify the training attribute fix.
This test verifies that the code uses self.model.training instead of self.training.
"""
import re
import sys


def test_training_attribute_usage():
    """Test that reward_trainer.py uses self.model.training instead of self.training."""
    print("\n" + "=" * 80)
    print("Testing Training Attribute Fix")
    print("=" * 80)
    
    # Read the reward_trainer.py file
    with open('/home/runner/work/GLM-training/GLM-training/glm_training/trainers/reward_trainer.py', 'r') as f:
        content = f.read()
    
    # Check for the problematic pattern: self.training (but not self.model.training)
    # We need to find uses of self.training that are NOT preceded by 'model.'
    lines = content.split('\n')
    
    problematic_lines = []
    for i, line in enumerate(lines, 1):
        # Skip comments
        if line.strip().startswith('#'):
            continue
        
        # Look for self.training but not self.model.training
        if 'self.training' in line and 'self.model.training' not in line:
            # Make sure it's not in a comment
            if '#' in line:
                code_part = line.split('#')[0]
                if 'self.training' in code_part:
                    problematic_lines.append((i, line.strip()))
            else:
                problematic_lines.append((i, line.strip()))
    
    if problematic_lines:
        print("\n✗ FAILED: Found problematic uses of self.training:")
        for line_num, line in problematic_lines:
            print(f"  Line {line_num}: {line}")
        return False
    
    # Check that self.model.training is used
    if 'self.model.training' in content:
        print("\n✓ PASSED: Code correctly uses self.model.training")
        
        # Show where it's used
        for i, line in enumerate(lines, 1):
            if 'self.model.training' in line and not line.strip().startswith('#'):
                print(f"  Line {i}: {line.strip()}")
        return True
    else:
        print("\n⚠ WARNING: Could not find self.model.training in the file")
        return False


def test_mock_has_training_attribute():
    """Test that the mock model in test file has training attribute."""
    print("\n" + "=" * 80)
    print("Testing Mock Model Training Attribute")
    print("=" * 80)
    
    # Read the test file
    with open('/home/runner/work/GLM-training/GLM-training/test_grpo_implementation.py', 'r') as f:
        content = f.read()
    
    # Check that the mock sets model.training
    if 'model.training = True' in content or 'model.training = False' in content:
        print("\n✓ PASSED: Mock model has training attribute")
        
        # Find the relevant lines
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'model.training' in line and '=' in line:
                print(f"  Line {i}: {line.strip()}")
        return True
    else:
        print("\n✗ FAILED: Mock model does not set training attribute")
        return False


def test_no_trainer_training_workaround():
    """Test that the test file doesn't set trainer.training as a workaround."""
    print("\n" + "=" * 80)
    print("Testing No Trainer Training Workaround")
    print("=" * 80)
    
    # Read the test file
    with open('/home/runner/work/GLM-training/GLM-training/test_grpo_implementation.py', 'r') as f:
        content = f.read()
    
    # Check that trainer.training is not set
    if 'trainer.training' in content:
        print("\n✗ FAILED: Found trainer.training workaround in test file")
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            if 'trainer.training' in line:
                print(f"  Line {i}: {line.strip()}")
        return False
    else:
        print("\n✓ PASSED: No trainer.training workaround found")
        return True


if __name__ == "__main__":
    results = []
    
    results.append(test_training_attribute_usage())
    results.append(test_mock_has_training_attribute())
    results.append(test_no_trainer_training_workaround())
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ All tests passed!")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
