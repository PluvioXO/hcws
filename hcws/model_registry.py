"""
Model Registry for HCWS

This module defines supported models and their configurations for HCWS steering.
Includes model-specific parameters and architecture detection.
"""

from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for a specific model."""
    
    def __init__(
        self,
        name: str,
        model_id: str,
        architecture: str,
        hidden_dim: int,
        num_layers: int,
        layer_attr_path: List[str],
        default_steering_strength: float = 3.0,
        requires_trust_remote_code: bool = False,
        torch_dtype: Optional[str] = None,
        description: str = ""
    ):
        """
        Initialize model configuration.
        
        Args:
            name: Human-readable model name
            model_id: HuggingFace model identifier
            architecture: Model architecture type
            hidden_dim: Hidden dimension size
            num_layers: Number of transformer layers
            layer_attr_path: Path to access transformer layers (e.g., ['model', 'layers'])
            default_steering_strength: Default steering strength for this model
            requires_trust_remote_code: Whether model requires trust_remote_code=True
            torch_dtype: Recommended torch dtype (e.g., "float16", "bfloat16")
            description: Model description
        """
        self.name = name
        self.model_id = model_id
        self.architecture = architecture
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.layer_attr_path = layer_attr_path
        self.default_steering_strength = default_steering_strength
        self.requires_trust_remote_code = requires_trust_remote_code
        self.torch_dtype = torch_dtype
        self.description = description


# Model registry with predefined configurations
MODEL_REGISTRY = {
    # GPT-2 family
    "gpt2": ModelConfig(
        name="GPT-2 Small",
        model_id="gpt2",
        architecture="gpt2",
        hidden_dim=768,
        num_layers=12,
        layer_attr_path=["transformer", "h"],
        default_steering_strength=3.0,
        description="OpenAI GPT-2 Small (117M parameters)"
    ),
    "gpt2-medium": ModelConfig(
        name="GPT-2 Medium",
        model_id="gpt2-medium",
        architecture="gpt2",
        hidden_dim=1024,
        num_layers=24,
        layer_attr_path=["transformer", "h"],
        default_steering_strength=3.0,
        description="OpenAI GPT-2 Medium (345M parameters)"
    ),
    "gpt2-large": ModelConfig(
        name="GPT-2 Large",
        model_id="gpt2-large",
        architecture="gpt2",
        hidden_dim=1280,
        num_layers=36,
        layer_attr_path=["transformer", "h"],
        default_steering_strength=3.0,
        description="OpenAI GPT-2 Large (762M parameters)"
    ),
    "gpt2-xl": ModelConfig(
        name="GPT-2 XL",
        model_id="gpt2-xl",
        architecture="gpt2",
        hidden_dim=1600,
        num_layers=48,
        layer_attr_path=["transformer", "h"],
        default_steering_strength=3.0,
        description="OpenAI GPT-2 XL (1.5B parameters)"
    ),
    
    # DeepSeek family
    "deepseek-v3": ModelConfig(
        name="DeepSeek-V3-0324",
        model_id="deepseek-ai/DeepSeek-V3-0324",
        architecture="deepseek",
        hidden_dim=7168,
        num_layers=61,
        layer_attr_path=["model", "layers"],
        default_steering_strength=2.0,
        requires_trust_remote_code=True,
        torch_dtype="bfloat16",
        description="DeepSeek-V3-0324 (671B parameters, MoE)"
    ),
    "deepseek-v2.5": ModelConfig(
        name="DeepSeek-V2.5",
        model_id="deepseek-ai/DeepSeek-V2.5",
        architecture="deepseek",
        hidden_dim=5120,
        num_layers=60,
        layer_attr_path=["model", "layers"],
        default_steering_strength=2.0,
        requires_trust_remote_code=True,
        torch_dtype="bfloat16",
        description="DeepSeek-V2.5 (236B parameters, MoE)"
    ),
    "deepseek-coder-v2": ModelConfig(
        name="DeepSeek-Coder-V2",
        model_id="deepseek-ai/DeepSeek-Coder-V2-Instruct",
        architecture="deepseek",
        hidden_dim=5120,
        num_layers=60,
        layer_attr_path=["model", "layers"],
        default_steering_strength=2.5,
        requires_trust_remote_code=True,
        torch_dtype="bfloat16",
        description="DeepSeek-Coder-V2 Instruct (236B parameters)"
    ),
    
    # Llama family
    "llama2-7b": ModelConfig(
        name="Llama 2 7B",
        model_id="meta-llama/Llama-2-7b-hf",
        architecture="llama",
        hidden_dim=4096,
        num_layers=32,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.5,
        description="Meta Llama 2 7B"
    ),
    "llama2-13b": ModelConfig(
        name="Llama 2 13B",
        model_id="meta-llama/Llama-2-13b-hf",
        architecture="llama",
        hidden_dim=5120,
        num_layers=40,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.5,
        description="Meta Llama 2 13B"
    ),
    "llama3-8b": ModelConfig(
        name="Llama 3 8B",
        model_id="meta-llama/Meta-Llama-3-8B",
        architecture="llama",
        hidden_dim=4096,
        num_layers=32,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.5,
        description="Meta Llama 3 8B"
    ),
    "llama3.1-8b": ModelConfig(
        name="Llama 3.1 8B",
        model_id="meta-llama/Llama-3.1-8B",
        architecture="llama",
        hidden_dim=4096,
        num_layers=32,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.5,
        description="Meta Llama 3.1 8B"
    ),
    
    # Mistral family
    "mistral-7b": ModelConfig(
        name="Mistral 7B",
        model_id="mistralai/Mistral-7B-v0.1",
        architecture="mistral",
        hidden_dim=4096,
        num_layers=32,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.5,
        description="Mistral 7B v0.1"
    ),
    "mixtral-8x7b": ModelConfig(
        name="Mixtral 8x7B",
        model_id="mistralai/Mixtral-8x7B-v0.1",
        architecture="mixtral",
        hidden_dim=4096,
        num_layers=32,
        layer_attr_path=["model", "layers"],
        default_steering_strength=2.5,
        description="Mixtral 8x7B MoE"
    ),
    
    # Qwen family
    "qwen2-7b": ModelConfig(
        name="Qwen2 7B Instruct",
        model_id="Qwen/Qwen2-7B-Instruct",
        architecture="qwen2",
        hidden_dim=3584,
        num_layers=28,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.0,
        description="Qwen2 7B Instruct"
    ),
    "qwen2.5-7b": ModelConfig(
        name="Qwen2.5 7B Instruct",
        model_id="Qwen/Qwen2.5-7B-Instruct",
        architecture="qwen2",
        hidden_dim=3584,
        num_layers=28,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.0,
        description="Qwen2.5 7B Instruct"
    ),
    "qwen2.5-3b": ModelConfig(
        name="Qwen2.5 3B Instruct",
        model_id="Qwen/Qwen2.5-3B-Instruct",
        architecture="qwen2",
        hidden_dim=2048,
        num_layers=36,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.5,
        description="Qwen2.5 3B Instruct"
    ),
    "qwen2.5-1.5b": ModelConfig(
        name="Qwen2.5 1.5B Instruct",
        model_id="Qwen/Qwen2.5-1.5B-Instruct",
        architecture="qwen2",
        hidden_dim=1536,
        num_layers=28,
        layer_attr_path=["model", "layers"],
        default_steering_strength=4.0,
        description="Qwen2.5 1.5B Instruct"
    ),
    "qwen2.5-0.5b": ModelConfig(
        name="Qwen2.5 0.5B Instruct",
        model_id="Qwen/Qwen2.5-0.5B-Instruct",
        architecture="qwen2",
        hidden_dim=896,
        num_layers=24,
        layer_attr_path=["model", "layers"],
        default_steering_strength=4.5,
        description="Qwen2.5 0.5B Instruct"
    ),
    
    # Gemma family
    "gemma-2b": ModelConfig(
        name="Gemma 2B",
        model_id="google/gemma-2b",
        architecture="gemma",
        hidden_dim=2048,
        num_layers=18,
        layer_attr_path=["model", "layers"],
        default_steering_strength=4.0,
        description="Google Gemma 2B"
    ),
    "gemma-7b": ModelConfig(
        name="Gemma 7B",
        model_id="google/gemma-7b",
        architecture="gemma",
        hidden_dim=3072,
        num_layers=28,
        layer_attr_path=["model", "layers"],
        default_steering_strength=3.5,
        description="Google Gemma 7B"
    ),
}


def get_model_config(model_key: str) -> ModelConfig:
    """
    Get model configuration by key.
    
    Args:
        model_key: Model key from registry
        
    Returns:
        Model configuration
        
    Raises:
        ValueError: If model key not found
    """
    if model_key not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Model '{model_key}' not found. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_key]


def list_available_models() -> Dict[str, str]:
    """
    List all available models with descriptions.
    
    Returns:
        Dictionary mapping model keys to descriptions
    """
    return {key: config.description for key, config in MODEL_REGISTRY.items()}


def get_models_by_architecture(architecture: str) -> List[str]:
    """
    Get all model keys for a specific architecture.
    
    Args:
        architecture: Architecture name
        
    Returns:
        List of model keys
    """
    return [key for key, config in MODEL_REGISTRY.items() if config.architecture == architecture]


def detect_model_config(model_name_or_path: str) -> Optional[ModelConfig]:
    """
    Try to detect model configuration from model name/path.
    
    Args:
        model_name_or_path: Model name or path
        
    Returns:
        Detected model configuration or None
    """
    # Direct match
    if model_name_or_path in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name_or_path]
    
    # Try partial matches
    model_lower = model_name_or_path.lower()
    
    for key, config in MODEL_REGISTRY.items():
        if key in model_lower or model_lower in config.model_id.lower():
            logger.info(f"Detected model configuration for '{model_name_or_path}' as '{key}'")
            return config
    
    # Try to infer from common patterns
    if "gpt2" in model_lower:
        return MODEL_REGISTRY["gpt2"]
    elif "deepseek-v3" in model_lower:
        return MODEL_REGISTRY["deepseek-v3"]
    elif "deepseek" in model_lower and "v2.5" in model_lower:
        return MODEL_REGISTRY["deepseek-v2.5"]
    elif "deepseek" in model_lower and "coder" in model_lower:
        return MODEL_REGISTRY["deepseek-coder-v2"]
    elif "llama" in model_lower and "3.1" in model_lower:
        return MODEL_REGISTRY["llama3.1-8b"]
    elif "llama" in model_lower and "3" in model_lower:
        return MODEL_REGISTRY["llama3-8b"]
    elif "llama" in model_lower and ("2" in model_lower or "13b" in model_lower):
        return MODEL_REGISTRY["llama2-13b"] if "13b" in model_lower else MODEL_REGISTRY["llama2-7b"]
    elif "mistral" in model_lower and "mixtral" not in model_lower:
        return MODEL_REGISTRY["mistral-7b"]
    elif "mixtral" in model_lower:
        return MODEL_REGISTRY["mixtral-8x7b"]
    elif "qwen2.5" in model_lower:
        return MODEL_REGISTRY["qwen2.5-7b"]
    elif "qwen2" in model_lower:
        return MODEL_REGISTRY["qwen2-7b"]
    elif "gemma" in model_lower and ("2b" in model_lower or "2B" in model_lower):
        return MODEL_REGISTRY["gemma-2b"]
    elif "gemma" in model_lower:
        return MODEL_REGISTRY["gemma-7b"]
    
    logger.warning(f"Could not detect model configuration for '{model_name_or_path}'")
    return None


def print_model_info(model_key: str):
    """
    Print detailed information about a model.
    
    Args:
        model_key: Model key from registry
    """
    config = get_model_config(model_key)
    
    print(f"Model: {config.name}")
    print(f"ID: {config.model_id}")
    print(f"Architecture: {config.architecture}")
    print(f"Hidden Dim: {config.hidden_dim}")
    print(f"Layers: {config.num_layers}")
    print(f"Default Steering Strength: {config.default_steering_strength}")
    print(f"Requires Trust Remote Code: {config.requires_trust_remote_code}")
    if config.torch_dtype:
        print(f"Recommended Dtype: {config.torch_dtype}")
    print(f"Description: {config.description}")


def print_available_models():
    """Print all available models organized by architecture."""
    architectures = {}
    for key, config in MODEL_REGISTRY.items():
        if config.architecture not in architectures:
            architectures[config.architecture] = []
        architectures[config.architecture].append((key, config))
    
    print("Available Models:")
    print("=" * 50)
    
    for arch, models in sorted(architectures.items()):
        print(f"\n{arch.upper()}:")
        for key, config in models:
            print(f"  {key:20} - {config.description}") 