"""
Tests for GRPO loss functions.

Day 1: Test the minimal GRPO implementation.
"""
import torch
import pytest
from glm_training.grpo.loss import compute_advantages, compute_grpo_loss


def test_compute_advantages_basic():
    """Test advantage computation with basic example."""
    # Two groups: one with variance, one constant
    rewards = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 5.0, 5.0, 5.0])
    advantages = compute_advantages(rewards, group_size=4)
    
    # Check shape
    assert advantages.shape == rewards.shape
    
    # Group 1: [1,2,3,4] should have variance
    group1_adv = advantages[:4]
    assert group1_adv.std() > 0, "Group 1 should have non-zero std"
    assert torch.allclose(group1_adv.mean(), torch.tensor(0.0), atol=1e-6), \
        "Group 1 advantages should have zero mean"
    
    # Group 2: [5,5,5,5] should be all zeros (constant rewards)
    group2_adv = advantages[4:]
    assert torch.allclose(group2_adv, torch.zeros(4), atol=1e-5), \
        "Group 2 advantages should be zero (constant rewards)"
    
    print("✅ test_compute_advantages_basic passed")


def test_compute_advantages_normalization():
    """Test that advantages are properly normalized."""
    rewards = torch.tensor([10.0, 20.0, 30.0, 40.0])
    advantages = compute_advantages(rewards, group_size=4)
    
    # Should have zero mean
    assert torch.allclose(advantages.mean(), torch.tensor(0.0), atol=1e-6)
    
    # Should have unit variance (approximately)
    assert torch.allclose(advantages.std(), torch.tensor(1.0), atol=0.2)
    
    print("✅ test_compute_advantages_normalization passed")


def test_grpo_loss_basic():
    """Test GRPO loss computation."""
    # Create dummy data
    new_lp = torch.randn(8, requires_grad=True)
    old_lp = torch.randn(8)
    advantages = torch.randn(8)
    
    loss, metrics = compute_grpo_loss(new_lp, old_lp, advantages)
    
    # Check loss has gradients
    assert loss.requires_grad, "Loss should have gradients"
    
    # Check can backprop
    loss.backward()
    assert new_lp.grad is not None, "Should be able to compute gradients"
    
    # Check metrics exist
    assert "loss" in metrics
    assert "ratio_mean" in metrics
    assert "advantages_mean" in metrics
    
    print("✅ test_grpo_loss_basic passed")


def test_grpo_loss_clipping():
    """Test that PPO clipping works correctly."""
    # Create scenario where ratio > 1.2 (should be clipped)
    new_lp = torch.tensor([2.0], requires_grad=True)  # ratio = exp(2.0) ≈ 7.4
    old_lp = torch.tensor([0.0])
    advantages = torch.tensor([1.0])
    
    loss, metrics = compute_grpo_loss(new_lp, old_lp, advantages, clip_range=0.2)
    
    # Ratio should be large
    assert metrics["ratio_mean"] > 1.2, "Ratio should exceed clip range"
    
    # Loss should be less than unclipped (due to clipping)
    ratio = torch.exp(new_lp - old_lp).item()
    unclipped_loss = -(ratio * advantages).mean().item()
    assert loss.item() > unclipped_loss, "Loss should be clipped (less negative)"
    
    print("✅ test_grpo_loss_clipping passed")


def test_grpo_loss_negative_advantage():
    """Test GRPO loss with negative advantages."""
    # Negative advantage should penalize policy when ratio > 1
    new_lp = torch.tensor([1.0], requires_grad=True)
    old_lp = torch.tensor([0.0])
    advantages = torch.tensor([-1.0])  # Negative advantage
    
    loss, metrics = compute_grpo_loss(new_lp, old_lp, advantages)
    
    # Loss should be positive (we're penalizing)
    assert loss.item() > 0, "Loss should be positive for negative advantage"
    
    print("✅ test_grpo_loss_negative_advantage passed")


def test_advantages_multiple_groups():
    """Test advantages with multiple groups."""
    # 3 groups of 4 samples each
    rewards = torch.tensor([
        # Group 1: ascending
        1.0, 2.0, 3.0, 4.0,
        # Group 2: descending
        4.0, 3.0, 2.0, 1.0,
        # Group 3: constant
        5.0, 5.0, 5.0, 5.0,
    ])
    
    advantages = compute_advantages(rewards, group_size=4)
    
    # Each group should have zero mean
    for i in range(3):
        group_adv = advantages[i*4:(i+1)*4]
        assert torch.allclose(group_adv.mean(), torch.tensor(0.0), atol=1e-6), \
            f"Group {i+1} should have zero mean"
    
    # Group 3 (constant) should be all zeros
    group3_adv = advantages[8:]
    assert torch.allclose(group3_adv, torch.zeros(4), atol=1e-5), \
        "Group 3 should be zero (constant rewards)"
    
    print("✅ test_advantages_multiple_groups passed")


if __name__ == "__main__":
    print("Running GRPO loss tests...")
    print()
    
    test_compute_advantages_basic()
    test_compute_advantages_normalization()
    test_grpo_loss_basic()
    test_grpo_loss_clipping()
    test_grpo_loss_negative_advantage()
    test_advantages_multiple_groups()
    
    print()
    print("=" * 50)
    print("All tests passed! ✅")
    print("=" * 50)
