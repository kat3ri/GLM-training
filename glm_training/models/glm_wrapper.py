"""
GLM-Image model wrapper for training.
"""
import torch
import torch.nn as nn
from typing import Optional, List, Union
from PIL import Image

try:
    from diffusers.pipelines.glm_image import GlmImagePipeline
    DIFFUSERS_AVAILABLE = True
except ImportError:
    DIFFUSERS_AVAILABLE = False


class GLMImageWrapper(nn.Module):
    """Wrapper for GLM-Image model to facilitate training."""
    
    def __init__(
        self,
        model_name: str = "zai-org/GLM-Image",
        component: str = "both",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
    ):
        """
        Initialize GLM-Image model wrapper.
        
        Args:
            model_name: Model name or path
            component: Which component to train ("ar", "dit", "both")
            torch_dtype: Data type for model weights
            device_map: Device mapping strategy. Supported values: "balanced", "cuda", "cpu".
                       "auto" is automatically mapped to "balanced" for flexibility.
        """
        super().__init__()
        
        if not DIFFUSERS_AVAILABLE:
            raise ImportError(
                "diffusers is not available. Please install with: "
                "pip install git+https://github.com/huggingface/diffusers.git"
            )
        
        self.component = component
        self.torch_dtype = torch_dtype
        
        # Map unsupported device_map values to supported ones
        # GlmImagePipeline.from_pretrained() supports: "balanced", "cuda", "cpu"
        device_map_mapping = {
            "auto": "balanced",  # "balanced" is more flexible for large models
        }
        resolved_device_map = device_map_mapping.get(device_map, device_map)
        
        # "balanced" works with single or multiple GPUs and automatically
        # balances memory across available devices. This is safer for
        # large models like GLM-Image (16B parameters total).
        
        # Load the pipeline
        self.pipe = GlmImagePipeline.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=resolved_device_map,
        )
        
        # Extract components
        self.ar_model = self.pipe.vision_language_encoder  # Autoregressive model
        self.dit_model = self.pipe.transformer  # DiT decoder
        self.vae = self.pipe.vae
        self.tokenizer = self.pipe.tokenizer
        
        # Freeze components based on what we're training
        self._setup_trainable_parameters()
    
    def _setup_trainable_parameters(self):
        """Setup which parameters are trainable based on component."""
        # Freeze everything by default
        self.vae.requires_grad_(False)
        
        if self.component == "ar":
            # Train only autoregressive model
            self.ar_model.requires_grad_(True)
            self.dit_model.requires_grad_(False)
        elif self.component == "dit":
            # Train only DiT decoder
            self.ar_model.requires_grad_(False)
            self.dit_model.requires_grad_(True)
        elif self.component == "both":
            # Train both components
            self.ar_model.requires_grad_(True)
            self.dit_model.requires_grad_(True)
        else:
            raise ValueError(f"Unknown component: {self.component}")
    
    def get_trainable_parameters(self):
        """Get trainable parameters."""
        params = []
        
        if self.component in ["ar", "both"]:
            params.extend(
                [p for p in self.ar_model.parameters() if p.requires_grad]
            )
        
        if self.component in ["dit", "both"]:
            params.extend(
                [p for p in self.dit_model.parameters() if p.requires_grad]
            )
        
        return params
    
    def get_ar_parameters(self):
        """Get autoregressive model parameters."""
        return [p for p in self.ar_model.parameters() if p.requires_grad]
    
    def get_dit_parameters(self):
        """Get DiT decoder parameters."""
        return [p for p in self.dit_model.parameters() if p.requires_grad]
    
    @torch.no_grad()
    def generate(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        generator: Optional[torch.Generator] = None,
        **kwargs,
    ) -> List[Image.Image]:
        """
        Generate images using the pipeline.
        
        Args:
            prompts: Text prompts
            images: Source images for i2i (optional)
            height: Image height
            width: Image width
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            generator: Random generator
            **kwargs: Additional arguments
            
        Returns:
            List of generated images
        """
        outputs = self.pipe(
            prompt=prompts,
            image=images,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            **kwargs,
        )
        
        return outputs.images
    
    def forward(
        self,
        prompts: List[str],
        images: Optional[List[Image.Image]] = None,
        height: int = 1024,
        width: int = 1024,
        num_inference_steps: int = 50,
        guidance_scale: float = 1.5,
        return_latents: bool = False,
        **kwargs,
    ):
        """
        Forward pass for training.
        
        Args:
            prompts: Text prompts
            images: Source images for i2i
            height: Image height
            width: Image width
            num_inference_steps: Number of inference steps
            guidance_scale: Guidance scale
            return_latents: Whether to return latents
            **kwargs: Additional arguments
            
        Returns:
            Generated images or latents
        """
        # This is a simplified version. In practice, you'd need to implement
        # the forward pass with gradient tracking for the specific components
        # being trained.
        
        # For now, we'll use the generate method
        # In production, you'd want to implement proper forward passes
        # for each component separately
        
        if self.training:
            # Training mode - implement custom forward pass
            # This would involve:
            # 1. Encode text prompts
            # 2. If i2i, encode source images
            # 3. Run AR model to generate tokens
            # 4. Run DiT decoder to generate image
            # 5. Return appropriate outputs for loss calculation
            raise NotImplementedError(
                "Training forward pass needs to be implemented based on "
                "specific training objectives"
            )
        else:
            # Inference mode
            return self.generate(
                prompts,
                images,
                height,
                width,
                num_inference_steps,
                guidance_scale,
                **kwargs,
            )
    
    def save_pretrained(self, save_path: str):
        """Save the model."""
        self.pipe.save_pretrained(save_path)
    
    def load_state_dict(self, state_dict, strict=True):
        """Load state dict."""
        # Load only the components we're training
        if self.component in ["ar", "both"]:
            ar_state_dict = {
                k.replace("ar_model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("ar_model.")
            }
            if ar_state_dict:
                self.ar_model.load_state_dict(ar_state_dict, strict=False)
        
        if self.component in ["dit", "both"]:
            dit_state_dict = {
                k.replace("dit_model.", ""): v
                for k, v in state_dict.items()
                if k.startswith("dit_model.")
            }
            if dit_state_dict:
                self.dit_model.load_state_dict(dit_state_dict, strict=False)
    
    def state_dict(self):
        """Get state dict."""
        state_dict = {}
        
        if self.component in ["ar", "both"]:
            ar_state = self.ar_model.state_dict()
            state_dict.update({f"ar_model.{k}": v for k, v in ar_state.items()})
        
        if self.component in ["dit", "both"]:
            dit_state = self.dit_model.state_dict()
            state_dict.update({f"dit_model.{k}": v for k, v in dit_state.items()})
        
        return state_dict
