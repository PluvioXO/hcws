"""
Main HCWS model that implements the steering mechanism with residual stream hooks.

This module provides the complete HCWS implementation that can be wrapped around
any transformer model to provide conceptor-based steering during inference.
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Optional, Tuple, Any, Callable
import logging
from functools import partial

from .encoder import InstructionEncoder
from .simple_encoder import SimpleInstructionEncoder
from .hyper_network import HyperNetwork
from .conceptors import ConceptorBank
from .controller import SteeringController
from .model_registry import get_model_config, detect_model_config, ModelConfig

logger = logging.getLogger(__name__)


class HCWSModel(nn.Module):
    """
    Main HCWS model that wraps a transformer and provides steering functionality.
    
    The model implements the complete HCWS pipeline:
    1. Instruction encoding with T5
    2. Conceptor generation with hyper-network
    3. Real-time steering with controller
    4. Activation modification via residual stream hooks
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        instruction_encoder_name: str = "t5-small",
        conceptor_rank: int = 32,
        controller_dim: int = 128,
        steering_layers: Optional[List[int]] = None,
        hook_frequency: int = 4,
        steering_strength: Optional[float] = None,
        device: Optional[str] = None,
        model_config: Optional[ModelConfig] = None,
        **model_kwargs
    ):
        """
        Initialize the HCWS model.
        
        Args:
            model_name_or_path: Name or path of the base transformer model
            instruction_encoder_name: Name of the T5 encoder for instructions
            conceptor_rank: Rank of conceptor matrices
            controller_dim: Controller hidden dimension
            steering_layers: List of layers to apply steering (None for all)
            hook_frequency: Apply steering every N tokens
            steering_strength: Multiplier for steering intensity (None for default)
            device: Device to run computations on
            model_config: Pre-configured model config (None for auto-detection)
            **model_kwargs: Additional arguments for model loading
        """
        super().__init__()
        
        from .device_utils import get_device
        self.device = get_device(device)
        self.hook_frequency = hook_frequency
        
        # Detect or use provided model configuration
        if model_config is None:
            model_config = detect_model_config(model_name_or_path)
        
        self.model_config = model_config
        
        # Set steering strength (use config default if not provided)
        if steering_strength is None and model_config:
            steering_strength = model_config.default_steering_strength
        elif steering_strength is None:
            steering_strength = 1.0
        
        self.steering_strength = steering_strength
        self.steering_layers = steering_layers
        
        # Prepare model loading arguments
        load_kwargs = {}
        if model_config and model_config.requires_trust_remote_code:
            load_kwargs['trust_remote_code'] = True
        if model_config and model_config.torch_dtype:
            import torch
            if model_config.torch_dtype == "float16":
                load_kwargs['torch_dtype'] = torch.float16
            elif model_config.torch_dtype == "bfloat16":
                load_kwargs['torch_dtype'] = torch.bfloat16
        
        # Add any additional kwargs
        load_kwargs.update(model_kwargs)
        
        # Determine the actual model path/ID to use
        if model_config and model_config.model_id:
            actual_model_path = model_config.model_id
        else:
            actual_model_path = model_name_or_path
        
        # Load base model and tokenizer
        self.base_model = AutoModelForCausalLM.from_pretrained(actual_model_path, **load_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path, **load_kwargs)
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Get model dimensions (use config if available, otherwise detect)
        if model_config:
            self.hidden_dim = model_config.hidden_dim
            self.num_layers = model_config.num_layers
        else:
            # Fallback to model config detection
            self.hidden_dim = self.base_model.config.hidden_size
            if hasattr(self.base_model.config, 'num_hidden_layers'):
                self.num_layers = self.base_model.config.num_hidden_layers
            elif hasattr(self.base_model.config, 'n_layer'):
                self.num_layers = self.base_model.config.n_layer
            elif hasattr(self.base_model.config, 'num_layers'):
                self.num_layers = self.base_model.config.num_layers
            else:
                raise ValueError("Could not determine number of layers from model config")
        
        # Set default steering layers
        if steering_layers is None:
            self.steering_layers = list(range(self.num_layers))
        
        # Initialize HCWS components
        try:
            self.instruction_encoder = InstructionEncoder(
                instruction_encoder_name,
                device=self.device
            )
        except Exception as e:
            logger.warning(f"Failed to load T5 encoder: {e}")
            logger.info("Falling back to simple BERT-based encoder")
            self.instruction_encoder = SimpleInstructionEncoder(device=self.device)
        
        self.hyper_network = HyperNetwork(
            instruction_dim=self.instruction_encoder.get_embedding_dim(),
            num_layers=self.num_layers,
            hidden_dim=self.hidden_dim,
            conceptor_rank=conceptor_rank,
            device=self.device
        )
        
        self.controller = SteeringController(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            controller_dim=controller_dim,
            steering_strength=steering_strength,
            device=self.device
        )
        
        # Move to device
        self.to(self.device)
        
        # Hooks and steering state
        self.hooks = []
        self.steering_active = False
        self.current_conceptor_bank = None
        self.token_count = 0
        
        logger.info(f"Initialized HCWS model with base model: {model_name_or_path}")
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
        Create a forward hook for steering at a specific layer.
        
        Args:
            layer_idx: Index of the layer to apply steering
            
        Returns:
            Hook function
        """
        def hook_fn(module, input, output):
            if not self.steering_active or self.current_conceptor_bank is None:
                return output
            
            # Check if we should apply steering at this token
            if self.token_count % self.hook_frequency != 0:
                return output
            
            # Get hidden states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            
            # Get steering parameters from controller
            gain, layer_weights = self.controller(
                hidden_states,
                layer_idx
            )
            
            # Get layer-specific weight
            layer_weight = layer_weights[:, layer_idx]
            
            # Apply conceptor steering
            conceptor = self.current_conceptor_bank.get_conceptor(layer_idx)
            
            # Implement: h'_{t,ℓ} = h_{t,ℓ} - g_t * w_{ℓ,t} * C_ℓ * h_{t,ℓ}
            batch_size, seq_len, hidden_dim = hidden_states.shape
            
            # Apply steering to the last token (current generation step)
            current_activation = hidden_states[:, -1:, :]  # [batch_size, 1, hidden_dim]
            
            # Contract activation toward conceptor subspace
            steering_strength = gain * layer_weight.unsqueeze(1)  # [batch_size, 1]
            contracted_activation = conceptor.contract_activation(
                current_activation,
                steering_strength.unsqueeze(2)
            )
            
            # Replace the last token's activation
            modified_hidden_states = hidden_states.clone()
            modified_hidden_states[:, -1:, :] = contracted_activation
            
            # Return modified output
            if isinstance(output, tuple):
                return (modified_hidden_states,) + output[1:]
            else:
                return modified_hidden_states
        
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
    
    def prepare_steering(self, instruction: str):
        """
        Prepare the model for steering with the given instruction.
        
        Args:
            instruction: Plain-language steering instruction
        """
        # Encode instruction
        instruction_embedding = self.instruction_encoder(instruction)
        
        # Generate conceptor bank
        self.current_conceptor_bank = self.hyper_network.generate_conceptor_bank(
            instruction_embedding.squeeze(0)
        )
        
        # Reset controller state
        self.controller.reset_state()
        
        # Reset token count
        self.token_count = 0
        
        logger.info(f"Prepared steering for instruction: {instruction}")
    
    def start_steering(self):
        """Start the steering process."""
        if self.current_conceptor_bank is None:
            raise ValueError("Must call prepare_steering() first")
        
        self._register_hooks()
        self.steering_active = True
        logger.info("Started steering")
    
    def stop_steering(self):
        """Stop the steering process."""
        self.steering_active = False
        self._remove_hooks()
        logger.info("Stopped steering")
    
    def forward(self, *args, **kwargs):
        """Forward pass through the base model."""
        # Increment token count for hook frequency
        self.token_count += 1
        
        return self.base_model(*args, **kwargs)
    
    def generate(
        self,
        input_text: str,
        steering_instruction: Optional[str] = None,
        max_length: int = 100,
        temperature: float = 1.0,
        top_p: float = 0.9,
        do_sample: bool = True,
        **kwargs
    ) -> str:
        """
        Generate text with optional steering.
        
        Args:
            input_text: Input text to generate from
            steering_instruction: Optional steering instruction
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
        
        # Prepare steering if instruction is provided
        if steering_instruction is not None:
            self.prepare_steering(steering_instruction)
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
            if steering_instruction is not None:
                self.stop_steering()
    
    def generate_with_multiple_instructions(
        self,
        input_text: str,
        steering_instructions: List[str],
        max_length: int = 100,
        **kwargs
    ) -> List[str]:
        """
        Generate multiple outputs with different steering instructions.
        
        Args:
            input_text: Input text to generate from
            steering_instructions: List of steering instructions
            max_length: Maximum length of generated text
            **kwargs: Additional generation parameters
            
        Returns:
            List of generated texts
        """
        results = []
        
        for instruction in steering_instructions:
            result = self.generate(
                input_text,
                steering_instruction=instruction,
                max_length=max_length,
                **kwargs
            )
            results.append(result)
        
        return results
    
    def compute_steering_strength(self, instruction: str) -> Dict[str, Any]:
        """
        Compute steering strength metrics for an instruction.
        
        Args:
            instruction: Steering instruction
            
        Returns:
            Dictionary with steering metrics
        """
        # Encode instruction
        instruction_embedding = self.instruction_encoder(instruction)
        
        # Generate conceptor bank
        conceptor_bank = self.hyper_network.generate_conceptor_bank(
            instruction_embedding.squeeze(0)
        )
        
        # Compute metrics
        apertures = conceptor_bank.get_apertures()
        
        return {
            'mean_aperture': sum(apertures) / len(apertures),
            'max_aperture': max(apertures),
            'min_aperture': min(apertures),
            'aperture_std': torch.std(torch.tensor(apertures)).item(),
            'layer_apertures': apertures
        }
    
    def save_steering_components(self, path: str):
        """
        Save the steering components (hyper-network and controller).
        
        Args:
            path: Path to save components
        """
        state = {
            'hyper_network': self.hyper_network.state_dict(),
            'controller': self.controller.state_dict(),
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'conceptor_rank': self.hyper_network.conceptor_rank,
                'controller_dim': self.controller.controller_dim
            }
        }
        
        torch.save(state, path)
        logger.info(f"Saved steering components to {path}")
    
    def load_steering_components(self, path: str):
        """
        Load steering components from file.
        
        Args:
            path: Path to load components from
        """
        state = torch.load(path, map_location=self.device)
        
        self.hyper_network.load_state_dict(state['hyper_network'])
        self.controller.load_state_dict(state['controller'])
        
        logger.info(f"Loaded steering components from {path}")
    
    def __del__(self):
        """Cleanup hooks on deletion."""
        self._remove_hooks()


class HCWSTrainer:
    """
    Trainer for HCWS components.
    
    This class handles training the hyper-network and controller components
    using instruction-response pairs.
    """
    
    def __init__(
        self,
        model: HCWSModel,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5
    ):
        """
        Initialize the trainer.
        
        Args:
            model: HCWS model to train
            learning_rate: Learning rate for optimization
            weight_decay: Weight decay for regularization
        """
        self.model = model
        
        # Only train HCWS components, not the base model
        hcws_params = list(model.hyper_network.parameters()) + list(model.controller.parameters())
        
        self.optimizer = torch.optim.AdamW(
            hcws_params,
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.criterion = nn.MSELoss()
        
    def train_step(
        self,
        instructions: List[str],
        input_texts: List[str],
        target_texts: List[str]
    ) -> Dict[str, float]:
        """
        Perform a single training step.
        
        Args:
            instructions: List of steering instructions
            input_texts: List of input texts
            target_texts: List of target texts
            
        Returns:
            Dictionary with loss values
        """
        self.optimizer.zero_grad()
        
        total_loss = 0.0
        batch_size = len(instructions)
        
        for instruction, input_text, target_text in zip(instructions, input_texts, target_texts):
            # Encode instruction
            instruction_embedding = self.model.instruction_encoder(instruction)
            
            # Generate conceptor bank
            conceptor_bank = self.model.hyper_network.generate_conceptor_bank(
                instruction_embedding.squeeze(0)
            )
            
            # Compute regularization loss
            reg_loss = conceptor_bank.compute_total_regularization()
            reg_loss += self.model.hyper_network.compute_regularization_loss()
            
            total_loss += reg_loss
        
        # Backward pass
        loss = total_loss / batch_size
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            list(self.model.hyper_network.parameters()) + 
            list(self.model.controller.parameters()),
            max_norm=1.0
        )
        
        self.optimizer.step()
        
        return {
            'total_loss': loss.item(),
            'regularization_loss': loss.item()
        }


def load_hcws_model(
    base_model_path: str,
    steering_components_path: str,
    device: Optional[str] = None
) -> HCWSModel:
    """
    Load a complete HCWS model with pre-trained steering components.
    
    Args:
        base_model_path: Path to base transformer model
        steering_components_path: Path to steering components
        device: Device to load model on
        
    Returns:
        Loaded HCWS model
    """
    # Load base model
    model = HCWSModel(base_model_path, device=device)
    
    # Load steering components
    model.load_steering_components(steering_components_path)
    
    return model 