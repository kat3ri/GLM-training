"""
Tests for GRPO trainer.

Day 2: Test the minimal GRPO trainer.
"""
import torch
import torch.nn as nn
from glm_training.grpo import MinimalGRPOTrainer, simple_reward_function


class DummyModel(nn.Module):
    """Dummy model for testing."""
    
    def __init__(self, vocab_size=1000, hidden_size=128, image_size=256):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size)
        self.image_size = image_size
        
    def forward(self, input_ids):
        """Forward pass."""
        hidden = self.embedding(input_ids)
        logits = self.lm_head(hidden)
        
        # Return object with logits attribute
        class Output:
            def __init__(self, logits):
                self.logits = logits
        
        return Output(logits)
    
    def generate(self, input_ids, max_new_tokens=10, **kwargs):
        """Dummy generation."""
        batch_size = input_ids.size(0)
        # Just append some random tokens
        new_tokens = torch.randint(0, 1000, (batch_size, max_new_tokens), device=input_ids.device)
        return torch.cat([input_ids, new_tokens], dim=1)


class DummyTokenizer:
    """Dummy tokenizer for testing."""
    
    def __call__(self, texts, return_tensors=None, padding=False):
        """Tokenize texts."""
        # Return dictionary with token IDs
        batch_size = len(texts)
        input_ids = torch.randint(0, 1000, (batch_size, 10))
        
        return {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


class DummyDataLoader:
    """Dummy dataloader for testing."""
    
    def __init__(self, num_batches=5, image_size=256):
        self.num_batches = num_batches
        self.image_size = image_size
        
    def __iter__(self):
        for i in range(self.num_batches):
            yield {
                "prompts": ["test prompt 1", "test prompt 2"],
                "target_images": torch.randn(2, 3, self.image_size, self.image_size),
            }


# Monkey-patch _ids_to_images to return correct size
def patch_trainer_images(trainer, image_size=256):
    """Patch trainer to return correct image size."""
    original_method = trainer._ids_to_images
    
    def _ids_to_images_patched(generated_ids):
        batch_size = generated_ids.size(0)
        return torch.randn(batch_size, 3, image_size, image_size, device=trainer.device)
    
    trainer._ids_to_images = _ids_to_images_patched


def test_trainer_initialization():
    """Test that trainer initializes correctly."""
    model = DummyModel()
    tokenizer = DummyTokenizer()
    dataloader = DummyDataLoader()
    
    trainer = MinimalGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,
        clip_range=0.2,
        learning_rate=5e-6,
    )
    
    # Check attributes
    assert trainer.model is model
    assert trainer.group_size == 4
    assert trainer.clip_range == 0.2
    assert trainer.step_count == 0
    
    print("✅ test_trainer_initialization passed")


def test_generate_samples():
    """Test sample generation."""
    model = DummyModel(image_size=256)
    tokenizer = DummyTokenizer()
    dataloader = DummyDataLoader(image_size=256)
    
    trainer = MinimalGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,
    )
    patch_trainer_images(trainer, 256)
    
    prompts = ["test prompt 1", "test prompt 2"]
    images, old_logprobs, token_ids = trainer.generate_samples(prompts)
    
    # Check shapes
    batch_size = len(prompts)
    group_size = trainer.group_size
    assert images.shape[0] == batch_size * group_size
    assert old_logprobs.shape[0] == batch_size * group_size
    assert token_ids.shape[0] == batch_size * group_size
    
    print("✅ test_generate_samples passed")


def test_train_step():
    """Test single training step."""
    model = DummyModel(image_size=256)
    tokenizer = DummyTokenizer()
    dataloader = DummyDataLoader(image_size=256)
    
    trainer = MinimalGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,
    )
    patch_trainer_images(trainer, 256)
    
    batch = {
        "prompts": ["test prompt"],
        "target_images": torch.randn(1, 3, 256, 256),
    }
    
    # Run training step
    metrics = trainer.train_step(batch)
    
    # Check metrics exist
    assert "loss" in metrics
    assert "reward_mean" in metrics
    assert "ratio_mean" in metrics
    assert "advantages_mean" in metrics
    
    # Check step count increased
    assert trainer.step_count == 1
    
    print("✅ test_train_step passed")


def test_train_loop():
    """Test training loop."""
    model = DummyModel(image_size=256)
    tokenizer = DummyTokenizer()
    dataloader = DummyDataLoader(num_batches=5, image_size=256)
    
    trainer = MinimalGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,
    )
    patch_trainer_images(trainer, 256)
    
    # Train for a few steps
    trainer.train(num_steps=3)
    
    # Check step count
    assert trainer.step_count == 3
    
    print("✅ test_train_loop passed")


def test_checkpoint_save_load():
    """Test checkpoint saving and loading."""
    import tempfile
    import os
    
    model = DummyModel(image_size=256)
    tokenizer = DummyTokenizer()
    dataloader = DummyDataLoader(image_size=256)
    
    trainer1 = MinimalGRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,
    )
    patch_trainer_images(trainer1, 256)
    
    # Train for a few steps
    batch = {
        "prompts": ["test prompt"],
        "target_images": torch.randn(1, 3, 256, 256),
    }
    trainer1.train_step(batch)
    trainer1.train_step(batch)
    
    # Save checkpoint
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        checkpoint_path = f.name
    
    trainer1.save_checkpoint(checkpoint_path)
    
    # Create new trainer and load
    model2 = DummyModel(image_size=256)
    trainer2 = MinimalGRPOTrainer(
        model=model2,
        tokenizer=tokenizer,
        train_dataloader=dataloader,
        reward_function=simple_reward_function,
        group_size=4,
    )
    
    trainer2.load_checkpoint(checkpoint_path)
    
    # Check step count matches
    assert trainer2.step_count == trainer1.step_count
    
    # Clean up
    os.unlink(checkpoint_path)
    
    print("✅ test_checkpoint_save_load passed")


if __name__ == "__main__":
    print("Running GRPO trainer tests...")
    print()
    
    test_trainer_initialization()
    test_generate_samples()
    test_train_step()
    test_train_loop()
    test_checkpoint_save_load()
    
    print()
    print("=" * 50)
    print("All trainer tests passed! ✅")
    print("=" * 50)
