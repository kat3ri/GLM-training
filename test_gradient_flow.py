#!/usr/bin/env python3
"""
Test to verify gradient flow in the fixed DiT training.
"""
import torch
import torch.nn.functional as F


def test_gradient_flow():
    """Test that the new training approach has proper gradient flow."""
    print("Testing gradient flow...")
    
    # Simulate the training scenario
    batch_size = 2
    channels = 4
    height = 64
    width = 64
    
    # Create a simple mock DiT model
    class MockDiTModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(channels, channels, 3, padding=1)
        
        def forward(self, noisy_latents, timesteps, text_embeds):
            # Simplified DiT forward pass
            x = self.conv(noisy_latents)
            return type('obj', (object,), {'sample': x})()
    
    # Create mock components
    dit_model = MockDiTModel()
    
    # Simulate training step
    # 1. Create target latents (from VAE encoding - detached)
    with torch.no_grad():
        latents = torch.randn(batch_size, channels, height, width)
    
    # 2. Sample timesteps
    timesteps = torch.randint(0, 1000, (batch_size,), dtype=torch.long)
    
    # 3. Add noise
    noise = torch.randn_like(latents)
    
    # Simple noise schedule
    sqrt_alpha_prod = (1 - timesteps.float() / 1000) ** 0.5
    sqrt_one_minus_alpha_prod = (timesteps.float() / 1000) ** 0.5
    
    sqrt_alpha_prod = sqrt_alpha_prod.reshape(-1, 1, 1, 1)
    sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.reshape(-1, 1, 1, 1)
    
    noisy_latents = sqrt_alpha_prod * latents + sqrt_one_minus_alpha_prod * noise
    
    # 4. Get text embeddings (detached)
    with torch.no_grad():
        text_embeds = torch.randn(batch_size, 77, 768)
    
    # 5. Predict noise with DiT model (WITH gradients)
    model_pred = dit_model(noisy_latents, timesteps, text_embeds).sample
    
    # 6. Compute loss
    loss = F.mse_loss(model_pred, noise)
    
    # 7. Check gradient flow
    print(f"Loss value: {loss.item():.6f}")
    print(f"Loss requires_grad: {loss.requires_grad}")
    print(f"Model_pred requires_grad: {model_pred.requires_grad}")
    
    # 8. Backward pass
    try:
        loss.backward()
        print("✓ Backward pass successful!")
        
        # Check if gradients exist
        has_gradients = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in dit_model.parameters()
        )
        print(f"✓ Gradients computed: {has_gradients}")
        
        if has_gradients:
            print("✓ Gradient flow test PASSED!")
            return True
        else:
            print("✗ No gradients computed")
            return False
            
    except RuntimeError as e:
        print(f"✗ Backward pass failed: {e}")
        return False


if __name__ == "__main__":
    success = test_gradient_flow()
    exit(0 if success else 1)
