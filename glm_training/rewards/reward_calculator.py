"""
Reward calculation for GLM-Image training.
Supports multiple reward metrics for assessing generated image quality.
"""
from typing import Dict, Optional, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SSIM_AVAILABLE = True
except ImportError:
    SSIM_AVAILABLE = False

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


class RewardCalculator:
    """Calculate rewards for generated images."""
    
    def __init__(
        self,
        metrics: Dict[str, float],
        device: str = "cuda",
    ):
        """
        Initialize reward calculator.
        
        Args:
            metrics: Dictionary of metric names and their weights
            device: Device to run calculations on
        """
        self.metrics = metrics
        self.device = device
        
        # Initialize LPIPS if available and needed
        self.lpips_model = None
        if "lpips" in metrics and LPIPS_AVAILABLE:
            self.lpips_model = lpips.LPIPS(net="alex").to(device)
            self.lpips_model.eval()
        
        # Initialize aesthetic predictor (simplified version)
        # In production, use a proper aesthetic scoring model
        self.aesthetic_enabled = "aesthetic" in metrics
        
        # Text accuracy uses OCR
        self.text_accuracy_enabled = "text_accuracy" in metrics and OCR_AVAILABLE
        
        # Structure preservation (for i2i)
        self.structure_enabled = "structure_preservation" in metrics
    
    def compute_reward(
        self,
        generated_images: torch.Tensor,
        target_images: torch.Tensor,
        prompts: Optional[List[str]] = None,
        source_images: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute reward for generated images.
        
        Args:
            generated_images: Generated images tensor (B, C, H, W)
            target_images: Target images tensor (B, C, H, W)
            prompts: List of text prompts (for text accuracy)
            source_images: Source images for i2i (B, C, H, W)
            
        Returns:
            Dictionary of reward components and total reward
        """
        rewards = {}
        
        # LPIPS (perceptual similarity) - lower is better
        if "lpips" in self.metrics and self.lpips_model is not None:
            lpips_score = self._compute_lpips(generated_images, target_images)
            # Convert to reward (invert and normalize)
            lpips_reward = (1.0 - lpips_score.clamp(0, 1)) * self.metrics["lpips"]
            rewards["lpips"] = lpips_reward
        
        # Aesthetic quality
        if self.aesthetic_enabled:
            aesthetic_score = self._compute_aesthetic(generated_images)
            aesthetic_reward = aesthetic_score * self.metrics["aesthetic"]
            rewards["aesthetic"] = aesthetic_reward
        
        # Text rendering accuracy
        if self.text_accuracy_enabled and prompts is not None:
            text_accuracy = self._compute_text_accuracy(generated_images, prompts)
            text_reward = text_accuracy * self.metrics["text_accuracy"]
            rewards["text_accuracy"] = text_reward
        
        # Structure preservation (for i2i)
        if self.structure_enabled and source_images is not None:
            structure_score = self._compute_structure_preservation(
                generated_images, source_images
            )
            structure_reward = structure_score * self.metrics["structure_preservation"]
            rewards["structure_preservation"] = structure_reward
        
        # Compute total reward
        total_reward = sum(rewards.values())
        rewards["total"] = total_reward
        
        return rewards
    
    def _compute_lpips(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LPIPS perceptual distance.
        
        Args:
            images1: First set of images (B, C, H, W)
            images2: Second set of images (B, C, H, W)
            
        Returns:
            LPIPS scores (B,)
        """
        # Normalize to [-1, 1] if needed
        if images1.max() > 1.0:
            images1 = images1 / 255.0
        if images2.max() > 1.0:
            images2 = images2 / 255.0
        
        images1 = images1 * 2.0 - 1.0
        images2 = images2 * 2.0 - 1.0
        
        with torch.no_grad():
            lpips_scores = self.lpips_model(images1, images2)
        
        return lpips_scores.squeeze()
    
    def _compute_aesthetic(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute aesthetic quality score.
        This is a simplified version. In production, use a trained aesthetic predictor.
        
        Args:
            images: Images tensor (B, C, H, W)
            
        Returns:
            Aesthetic scores (B,)
        """
        # Simple heuristic based on color diversity and contrast
        batch_size = images.shape[0]
        scores = []
        
        for i in range(batch_size):
            img = images[i]
            
            # Color diversity (std across channels)
            color_std = img.std(dim=[1, 2]).mean()
            
            # Contrast (std within each channel)
            contrast = img.std(dim=[1, 2]).mean()
            
            # Simple combined score
            score = (color_std * 0.5 + contrast * 0.5).clamp(0, 1)
            scores.append(score)
        
        return torch.stack(scores)
    
    def _compute_text_accuracy(
        self,
        images: torch.Tensor,
        prompts: List[str],
    ) -> torch.Tensor:
        """
        Compute text rendering accuracy using OCR.
        
        Args:
            images: Generated images (B, C, H, W)
            prompts: Text prompts
            
        Returns:
            Text accuracy scores (B,)
        """
        if not self.text_accuracy_enabled:
            return torch.zeros(images.shape[0], device=images.device)
        
        scores = []
        for i, prompt in enumerate(prompts):
            img = images[i]
            
            # Convert to PIL Image
            img_np = img.cpu().numpy()
            if img_np.max() <= 1.0:
                img_np = (img_np * 255).astype(np.uint8)
            img_np = np.transpose(img_np, (1, 2, 0))
            pil_img = Image.fromarray(img_np)
            
            try:
                # Extract text using OCR
                extracted_text = pytesseract.image_to_string(pil_img).lower()
                
                # Extract quoted text from prompt
                import re
                quoted_texts = re.findall(r'"([^"]*)"', prompt)
                
                if quoted_texts:
                    # Calculate how many quoted texts appear in the image
                    matches = sum(1 for text in quoted_texts if text.lower() in extracted_text)
                    score = matches / len(quoted_texts)
                else:
                    # No quoted text to check
                    score = 1.0
                
                scores.append(torch.tensor(score, device=images.device))
            except Exception:
                # OCR failed, assign neutral score
                scores.append(torch.tensor(0.5, device=images.device))
        
        return torch.stack(scores)
    
    def _compute_structure_preservation(
        self,
        generated_images: torch.Tensor,
        source_images: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute structure preservation score for i2i.
        Uses SSIM to measure structural similarity.
        
        Args:
            generated_images: Generated images (B, C, H, W)
            source_images: Source images (B, C, H, W)
            
        Returns:
            Structure preservation scores (B,)
        """
        if not SSIM_AVAILABLE:
            # Fallback to simple MSE-based metric
            mse = F.mse_loss(generated_images, source_images, reduction="none")
            mse = mse.mean(dim=[1, 2, 3])
            # Convert to similarity score
            return torch.exp(-mse)
        
        batch_size = generated_images.shape[0]
        scores = []
        
        # Convert to numpy for SSIM calculation
        gen_np = generated_images.cpu().numpy()
        src_np = source_images.cpu().numpy()
        
        for i in range(batch_size):
            # Calculate SSIM for each channel and average
            ssim_scores = []
            for c in range(3):
                score = ssim(
                    gen_np[i, c],
                    src_np[i, c],
                    data_range=gen_np[i, c].max() - gen_np[i, c].min()
                )
                ssim_scores.append(score)
            
            avg_ssim = np.mean(ssim_scores)
            scores.append(torch.tensor(avg_ssim, device=generated_images.device))
        
        return torch.stack(scores)
    
    def compute_grpo_rewards(
        self,
        samples: List[torch.Tensor],
        target_images: torch.Tensor,
        prompts: Optional[List[str]] = None,
        source_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute GRPO (Group Relative Policy Optimization) rewards.
        
        Args:
            samples: List of sample tensors, each (B, C, H, W)
            target_images: Target images (B, C, H, W)
            prompts: Text prompts
            source_images: Source images for i2i
            
        Returns:
            Relative rewards for each sample (num_samples, B)
        """
        num_samples = len(samples)
        batch_size = samples[0].shape[0]
        
        # Compute rewards for all samples
        all_rewards = []
        for sample in samples:
            rewards = self.compute_reward(
                sample,
                target_images,
                prompts,
                source_images,
            )
            all_rewards.append(rewards["total"])
        
        # Stack rewards (num_samples, B)
        all_rewards = torch.stack(all_rewards)
        
        # Compute group relative rewards (normalize within each group)
        mean_rewards = all_rewards.mean(dim=0, keepdim=True)
        std_rewards = all_rewards.std(dim=0, keepdim=True) + 1e-8
        
        relative_rewards = (all_rewards - mean_rewards) / std_rewards
        
        return relative_rewards
