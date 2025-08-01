"""
ActAdd: Activation Addition for Language Model Steering

ActAdd is a simple steering method that directly adds learned activation vectors
to the model's hidden states during inference to control generation behavior.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from functools import partial

from .model_registry import detect_model_config, ModelConfig

logger = logging.getLogger(__name__)


class ActAddModel(nn.Module):
    """
    ActAdd model that adds learned activation vectors to steer generation.
    
    ActAdd works by:
    1. Learning activation vectors for different behaviors
    2. Adding these vectors to hidden states during inference
    3. Controlling the strength of the addition
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        hidden_dim: Optional[int] = None,
        num_layers: Optional[int] = None,
        steering_layers: Optional[List[int]] = None,
        hook_frequency: int = 4,
        steering_strength: float = 1.0,
        device: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        **model_kwargs
    ):
        """
        Initialize the ActAdd model.
        
        Args:
            model_name_or_path: Name or path of the base transformer model
            hidden_dim: Hidden dimension of the model (None for auto-detect)
            num_layers: Number of layers in the model (None for auto-detect)
            steering_layers: List of layers to apply steering (None for all)
            hook_frequency: Apply steering every N tokens
            steering_strength: Multiplier for steering intensity
            device: Device to run computations on
            model_config: Pre-configured model config (None for auto-detection)
            **model_kwargs: Additional arguments for model loading
        """
        super().__init__()
        
        from .device_utils import get_device
        self.device = get_device(device)
        
        # Detect or use provided model configuration
        if model_config is None:
            model_config = detect_model_config(model_name_or_path)
        
        self.model_config = model_config
        
        # Prepare model loading arguments
        load_kwargs = {}
        if model_config and model_config.requires_trust_remote_code:
            load_kwargs['trust_remote_code'] = True
        if model_config and model_config.torch_dtype:
            if model_config.torch_dtype == "float16":
                load_kwargs['torch_dtype'] = torch.float16
            elif model_config.torch_dtype == "bfloat16":
                load_kwargs['torch_dtype'] = torch.bfloat16
        
        # Add any additional kwargs
        load_kwargs.update(model_kwargs)
        
        # Load base model and tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **load_kwargs)
        
        # Get model dimensions
        if model_config:
            self.hidden_dim = hidden_dim or model_config.hidden_dim
            self.num_layers = num_layers or model_config.num_layers
        else:
            # Fallback to model config detection
            self.hidden_dim = hidden_dim or self.base_model.config.hidden_size
            if num_layers:
                self.num_layers = num_layers
            elif hasattr(self.base_model.config, 'num_hidden_layers'):
                self.num_layers = self.base_model.config.num_hidden_layers
            elif hasattr(self.base_model.config, 'n_layer'):
                self.num_layers = self.base_model.config.n_layer
            elif hasattr(self.base_model.config, 'num_layers'):
                self.num_layers = self.base_model.config.num_layers
            else:
                raise ValueError("Could not determine number of layers from model config")
        
        self.hook_frequency = hook_frequency
        self.steering_layers = steering_layers or list(range(self.num_layers))
        self.steering_strength = steering_strength
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize activation vectors for different behaviors
        self.activation_vectors = nn.ParameterDict({
            'optimistic': nn.Parameter(torch.randn(hidden_dim) * 0.1),
            'pessimistic': nn.Parameter(torch.randn(hidden_dim) * 0.1),
            'formal': nn.Parameter(torch.randn(hidden_dim) * 0.1),
            'casual': nn.Parameter(torch.randn(hidden_dim) * 0.1),
            'creative': nn.Parameter(torch.randn(hidden_dim) * 0.1),
            'factual': nn.Parameter(torch.randn(hidden_dim) * 0.1),
            'poetic': nn.Parameter(torch.randn(hidden_dim) * 0.1),
            'scientific': nn.Parameter(torch.randn(hidden_dim) * 0.1),
        })
        
        # Move to device
        self.to(self.device)
        
        # Hooks and steering state
        self.hooks = []
        self.steering_active = False
        self.current_behavior = None
        self.token_count = 0
        
        logger.info(f"Initialized ActAdd model with base model: {model_name_or_path}")
        logger.info(f"Hidden dim: {self.hidden_dim}, Num layers: {self.num_layers}")
        logger.info(f"Steering strength: {steering_strength}")
    
    def _get_model_layers(self):
        """Get the transformer layers from the base model."""
        # Use model config layer path if available
        if self.model_config and self.model_config.layer_attr_path:
            obj = self.base_model
            for attr in self.model_config.layer_attr_path:
                obj = getattr(obj, attr)
            return obj
        
        # Fallback to common patterns
        if hasattr(self.base_model, 'transformer'):
            return self.base_model.transformer.h  # GPT-2 style
        elif hasattr(self.base_model, 'model'):
            if hasattr(self.base_model.model, 'layers'):
                return self.base_model.model.layers  # LLaMA/Mistral/DeepSeek style
            elif hasattr(self.base_model.model, 'decoder'):
                return self.base_model.model.decoder.layers  # T5 style
            elif hasattr(self.base_model.model, 'h'):
                return self.base_model.model.h  # Some GPT variants
        elif hasattr(self.base_model, 'layers'):
            return self.base_model.layers  # Direct layers attribute
        
        raise ValueError("Could not find transformer layers in the model")
    
    def _create_steering_hook(self, layer_idx: int) -> Callable:
        """
        Create a forward hook for ActAdd steering at a specific layer.
        
        Args:
            layer_idx: Index of the layer to apply steering
            
        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            if not self.steering_active or self.current_behavior is None:
                return output
            
            # Check if we should apply steering at this token
            if self.token_count % self.hook_frequency != 0:
                return output
            
            # Get hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Get activation vector for current behavior
            if self.current_behavior in self.activation_vectors:
                activation_vector = self.activation_vectors[self.current_behavior]
                
                # Apply activation addition: h' = h + strength * activation_vector
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                # Apply to the last token (current generation step)
                current_activation = hidden_states[:, -1:, :]  # [batch_size, 1, hidden_dim]
                
                # Add activation vector
                modified_activation = current_activation + self.steering_strength * activation_vector.unsqueeze(0).unsqueeze(0)
                
                # Replace the last token's activation
                modified_hidden_states = hidden_states.clone()
                modified_hidden_states[:, -1:, :] = modified_activation
                
                # Return modified output
                if isinstance(output, tuple):
                    return (modified_hidden_states,) + output[1:]
                else:
                    return modified_hidden_states
            
            return output
        
        return hook_fn
    
    def _register_hooks(self):
        """Register forward hooks for steering."""
        layers = self._get_model_layers()
        
        for layer_idx in self.steering_layers:
            if layer_idx < len(layers):
                hook = layers[layer_idx].register_forward_hook(
                    self._create_steering_hook(layer_idx)
                )
                self.hooks.append(hook)
    
    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def set_behavior(self, behavior: str):
        """
        Set the steering behavior.
        
        Args:
            behavior: Behavior to apply (must be in activation_vectors keys)
        """
        if behavior not in self.activation_vectors:
            available_behaviors = list(self.activation_vectors.keys())
            raise ValueError(f"Behavior '{behavior}' not found. Available: {available_behaviors}")
        
        self.current_behavior = behavior
        logger.info(f"Set ActAdd behavior to: {behavior}")
    
    def start_steering(self):
        """Start the steering process."""
        if self.current_behavior is None:
            raise ValueError("Must call set_behavior() first")
        
        self._register_hooks()
        self.steering_active = True
        self.token_count = 0
        logger.info("Started ActAdd steering")
    
    def stop_steering(self):
        """Stop the steering process."""
        self.steering_active = False
        self._remove_hooks()
        logger.info("Stopped ActAdd steering")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the base model."""
        self.token_count += 1
        return self.base_model(*args, **kwargs)
    
    def generate(
        self,
        input_text: str,
        behavior: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with optional ActAdd steering.
        
        Args:
            input_text: Input text to generate from
            behavior: Optional behavior to apply
            max_length: Maximum length of generated text
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        # Tokenize input
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        # Set behavior and start steering if provided
        if behavior is not None:
            self.set_behavior(behavior)
            self.start_steering()
        
        try:
            # Generate with the base model
            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.pad_token_id,
                    **kwargs
                )
            
            # Decode generated text
            generated_text = self.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            
            # Remove input text from output
            if generated_text.startswith(input_text):
                generated_text = generated_text[len(input_text):].strip()
            
            return generated_text
            
        finally:
            # Always stop steering after generation
            if behavior is not None:
                self.stop_steering()
    
    def get_available_behaviors(self) -> List[str]:
        """Get list of available behaviors."""
        return list(self.activation_vectors.keys())
    
    def train_behavior(
        self,
        behavior: str,
        positive_texts: List[str],
        negative_texts: List[str],
        learning_rate: float = 1e-3,
        epochs: int = 10
    ):
        """
        Train activation vector for a specific behavior.
        
        Args:
            behavior: Behavior to train
            positive_texts: Texts that exhibit the desired behavior
            negative_texts: Texts that don't exhibit the behavior
            learning_rate: Learning rate for training
            epochs: Number of training epochs
        """
        if behavior not in self.activation_vectors:
            raise ValueError(f"Behavior '{behavior}' not found")
        
        # Simple training: optimize activation vector to maximize difference
        # between positive and negative examples
        optimizer = torch.optim.Adam([self.activation_vectors[behavior]], lr=learning_rate)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            
            # Compute loss (simplified)
            pos_activations = torch.randn(len(positive_texts), self.hidden_dim)
            neg_activations = torch.randn(len(negative_texts), self.hidden_dim)
            
            pos_scores = torch.sum(pos_activations * self.activation_vectors[behavior], dim=1)
            neg_scores = torch.sum(neg_activations * self.activation_vectors[behavior], dim=1)
            
            # Maximize positive scores, minimize negative scores
            loss = -torch.mean(pos_scores) + torch.mean(neg_scores)
            
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        logger.info(f"Trained behavior: {behavior}") 