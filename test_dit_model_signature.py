#!/usr/bin/env python3
"""
Test to validate the DiT model call signature fix.
This verifies that the DiT model is called with correct argument order.
"""
import re


def test_dit_model_call_signature():
    """Test that dit_model is called with correct argument order."""
    print("\n" + "=" * 80)
    print("Testing DiT Model Call Signature")
    print("=" * 80)
    
    # Read the reward_trainer.py file
    with open('/home/runner/work/GLM-training/GLM-training/glm_training/trainers/reward_trainer.py', 'r') as f:
        content = f.read()
    
    # Find all dit_model calls
    pattern = r'self\.model\.dit_model\([^)]+\)'
    matches = list(re.finditer(pattern, content, re.DOTALL))
    
    print(f"\nFound {len(matches)} dit_model call(s)")
    
    all_correct = True
    for i, match in enumerate(matches, 1):
        call = match.group(0)
        print(f"\n--- Call {i} ---")
        print(call[:200] + "..." if len(call) > 200 else call)
        
        # Check if it uses the wrong pattern (encoder_hidden_states as keyword arg)
        if 'encoder_hidden_states=' in call:
            print("✗ FAILED: Uses encoder_hidden_states as keyword argument")
            all_correct = False
        else:
            print("✓ PASSED: Does not use encoder_hidden_states as keyword argument")
    
    # Check for the correct pattern: positional arguments in order
    # The call should look like: dit_model(latents, prompt_embeds, timestep)
    correct_pattern = r'self\.model\.dit_model\(\s*latents,\s*prompt_embeds,\s*(?:t|timestep)'
    correct_matches = list(re.finditer(correct_pattern, content, re.DOTALL))
    
    print(f"\n" + "=" * 80)
    if len(correct_matches) == len(matches):
        print(f"✓ All {len(matches)} call(s) use correct positional argument order")
        print("  Expected: dit_model(latents, prompt_embeds, timestep)")
        return True
    else:
        print(f"✗ {len(correct_matches)}/{len(matches)} call(s) use correct argument order")
        return False


def test_no_duplicate_argument_error():
    """Test that there's no pattern that would cause 'multiple values' error."""
    print("\n" + "=" * 80)
    print("Testing for Duplicate Argument Pattern")
    print("=" * 80)
    
    with open('/home/runner/work/GLM-training/GLM-training/glm_training/trainers/reward_trainer.py', 'r') as f:
        content = f.read()
    
    # Pattern that would cause "multiple values" error:
    # dit_model(arg1, arg2, encoder_hidden_states=...)
    # where arg2 is treated as positional encoder_hidden_states
    bad_pattern = r'dit_model\([^,]+,\s*(?:t|timestep)[^,]*,\s*encoder_hidden_states='
    
    if re.search(bad_pattern, content):
        print("✗ FAILED: Found pattern that could cause 'multiple values' error")
        return False
    else:
        print("✓ PASSED: No patterns that would cause 'multiple values' error")
        return True


if __name__ == "__main__":
    import sys
    
    results = []
    results.append(test_dit_model_call_signature())
    results.append(test_no_duplicate_argument_error())
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\nTests passed: {passed}/{total}")
    
    if all(results):
        print("\n✓ All tests passed! DiT model calls are correct.")
        sys.exit(0)
    else:
        print("\n✗ Some tests failed!")
        sys.exit(1)
