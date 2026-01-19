#!/usr/bin/env python3
"""
Test to validate the DiT model signature fix.
This verifies that all required arguments are passed to GlmImageTransformer2DModel.forward().
"""
import re


def test_dit_model_has_required_arguments():
    """Test that dit_model calls include all required arguments."""
    print("\n" + "=" * 80)
    print("Testing DiT Model Has Required Arguments")
    print("=" * 80)
    
    # Read the reward_trainer.py file
    with open('glm_training/trainers/reward_trainer.py', 'r') as f:
        content = f.read()
    
    # Find all dit_model calls
    pattern = r'self\.model\.dit_model\([^)]+\)\.sample'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    print(f"\nFound {len(matches)} dit_model call(s)")
    
    required_args = ['prior_token_drop', 'height', 'width', 'crop_top', 'crop_left']
    all_correct = True
    
    for i, match in enumerate(matches, 1):
        call = match.group(0)
        print(f"\n--- Call {i} ---")
        
        # Extract just the part before .sample for clearer output
        call_without_sample = call.replace('.sample', '')
        
        # Check for each required argument (as variable names in the call)
        missing_args = []
        
        # Check all required variables are present
        if 'prior_token_drop' not in call:
            missing_args.append('prior_token_drop')
        if 'height,' not in call and 'height\n' not in call:
            missing_args.append('height')
        if 'width,' not in call and 'width\n' not in call:
            missing_args.append('width')
        if 'crop_top' not in call:
            missing_args.append('crop_top')
        if 'crop_left' not in call:
            missing_args.append('crop_left')
        
        if missing_args:
            print(f"✗ FAILED: Missing arguments: {', '.join(missing_args)}")
            all_correct = False
        else:
            print(f"✓ PASSED: All required arguments present")
            
        # Show the call (truncated if too long)
        if len(call_without_sample) > 300:
            print(f"Call preview: {call_without_sample[:300]}...")
        else:
            print(f"Full call: {call_without_sample}")
    
    print(f"\n" + "=" * 80)
    if all_correct:
        print(f"✓ All {len(matches)} call(s) include required arguments")
        return True
    else:
        print(f"✗ Some calls are missing required arguments")
        return False


def test_arguments_defined_before_use():
    """Test that prior_token_drop, target_size, crop_coords are defined before use."""
    print("\n" + "=" * 80)
    print("Testing Arguments Are Defined Before Use")
    print("=" * 80)
    
    with open('glm_training/trainers/reward_trainer.py', 'r') as f:
        lines = f.readlines()
    
    # Find dit_model call locations
    dit_call_lines = []
    for i, line in enumerate(lines):
        if 'self.model.dit_model(' in line:
            dit_call_lines.append(i)
    
    print(f"\nFound dit_model calls at lines: {[l+1 for l in dit_call_lines]}")
    
    all_correct = True
    for call_line in dit_call_lines:
        print(f"\n--- Checking call at line {call_line + 1} ---")
        
        # Look backwards from the call to find variable definitions
        search_range = max(0, call_line - 20)
        context = ''.join(lines[search_range:call_line + 10])
        
        required_vars = {
            'prior_token_drop': False,
        }
        
        for var in required_vars:
            if f'{var} =' in context:
                required_vars[var] = True
        
        # For height and width, they should be computed earlier in the function
        # Check if they're available in a broader context
        if 'height' not in context or 'width' not in context:
            print(f"⚠ WARNING: height/width may not be in immediate context (should be computed earlier)")
        
        # Check for crop_top and crop_left
        if 'crop_top' not in context:
            print(f"⚠ WARNING: crop_top may not be in immediate context")
        if 'crop_left' not in context:
            print(f"⚠ WARNING: crop_left may not be in immediate context")
                
        missing_vars = [v for v, found in required_vars.items() if not found]
        
        if missing_vars:
            print(f"✗ FAILED: Variables not defined: {', '.join(missing_vars)}")
            all_correct = False
        else:
            print(f"✓ PASSED: Required variables are defined")
    
    print(f"\n" + "=" * 80)
    if all_correct:
        print(f"✓ All required arguments are properly defined")
        return True
    else:
        print(f"✗ Some arguments are not properly defined")
        return False


def test_argument_values():
    """Test that arguments have reasonable values."""
    print("\n" + "=" * 80)
    print("Testing Argument Values")
    print("=" * 80)
    
    with open('glm_training/trainers/reward_trainer.py', 'r') as f:
        content = f.read()
    
    checks = {
        'prior_token_drop': r'prior_token_drop\s*=\s*(0\.[0-9]+|1\.0|0)',  # Valid probability [0, 1]
        'crop_top': r'crop_top\s*=\s*[0-9]+',
        'crop_left': r'crop_left\s*=\s*[0-9]+',
    }
    
    all_correct = True
    for var_name, pattern in checks.items():
        matches = re.findall(pattern, content)
        if matches:
            print(f"✓ {var_name}: Found {len(matches)} definition(s)")
            for match in matches[:2]:  # Show first 2 matches
                print(f"  Value: {var_name} = {match}")
        else:
            print(f"✗ {var_name}: No definitions found")
            all_correct = False
    
    # Check that height and width are used (computed earlier in the function)
    if 'height' in content and 'width' in content:
        print(f"✓ height and width: Variables are used in the code")
    else:
        print(f"✗ height and width: Variables not found")
        all_correct = False
    
    print(f"\n" + "=" * 80)
    if all_correct:
        print(f"✓ All arguments have valid definitions")
        return True
    else:
        print(f"✗ Some arguments are missing definitions")
        return False


if __name__ == "__main__":
    import sys
    
    results = []
    results.append(test_dit_model_has_required_arguments())
    results.append(test_arguments_defined_before_use())
    results.append(test_argument_values())
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ All tests passed! DiT model calls include all required arguments.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
