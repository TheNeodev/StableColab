import os
import torch
from diffusers import DiffusionPipeline, AutoencoderKL
from typing import Optional

class SDXLInference:
    def __init__(self, 
                 model_id: str = "stabilityai/stable-diffusion-xl-base-1.0",
                 vae_model_id: str = "madebyollin/sdxl-vae-fp16-fix",
                 lora_path: Optional[str] = None,
                 use_cpu: bool = False,
                 use_tpu: bool = False):
        """
        Initialize SDXL inference pipeline
        
        Args:
            model_id: ID of the base SDXL model
            vae_model_id: ID of the VAE model
            lora_path: Path to LoRA weights file (.safetensors)
            use_cpu: Force CPU usage
            use_tpu: Use TPU accelerator
        """
        self.model_id = model_id
        self.vae_model_id = vae_model_id
        self.lora_path = lora_path
        
        # Determine device
        self.device = self._get_device(use_cpu, use_tpu)
        
        # Initialize pipeline
        self.pipeline = self._initialize_pipeline()
        
    def _get_device(self, use_cpu: bool, use_tpu: bool) -> torch.device:
        """Determine and configure the compute device"""
        if use_tpu:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            print(f"Using TPU: {device}")
        elif use_cpu:
            device = torch.device("cpu")
            print("Using CPU")
        else:
            if torch.cuda.is_available():
                device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                device = torch.device("cpu")
                print("Falling back to CPU (no CUDA available)")
                
        return device
    
    def _initialize_pipeline(self) -> DiffusionPipeline:
        """Initialize the diffusion pipeline with model and LoRA weights"""
        vae = AutoencoderKL.from_pretrained(
            self.vae_model_id,
            torch_dtype=torch.float32
        )   
        pipe = DiffusionPipeline.from_pretrained(
            self.model_id,
            vae=vae,
            torch_dtype=torch.float32,
            use_safetensors=True
        )
        if self.lora_path:
            pipe.load_lora_weights(self.lora_path)
           
        pipe.to(self.device)
    return pipe
    
    def generate(self, 
                prompt: str,
                num_inference_steps: int = 50,
                guidance_scale: float = 7.5,
                **kwargs) -> torch.Tensor:
        """Generate an image based on the prompt"""
        result = self.pipeline(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            **kwargs
        )
        return result.images[0]
