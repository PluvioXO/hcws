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
        
        # Load base model and tokenizer with special handling for GPT-OSS
        from transformers import AutoModelForCausalLM, AutoTokenizer  # Import at top level
        
        try:
            self.base_model = AutoModelForCausalLM.from_pretrained(actual_model_path, **load_kwargs)
            self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path, **load_kwargs)
        except Exception as e:
            if "gpt_oss" in str(e).lower() or "does not recognize this architecture" in str(e):
                print(f"GPT-OSS architecture detected. Implementing proper GPT-OSS support...")
                
                success = False
                
                # Method 1: Install bleeding-edge transformers with GPT-OSS support
                if not success:
                    try:
                        import subprocess
                        import sys
                        import os
                        
                        print("Installing bleeding-edge transformers with GPT-OSS support...")
                        
                        # Install the absolute latest transformers from main branch
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", 
                            "git+https://github.com/huggingface/transformers.git@main",
                            "--force-reinstall", "--no-deps", "--quiet"
                        ])
                        
                        # Install specific transformers version that supports GPT-OSS
                        try:
                            subprocess.check_call([
                                sys.executable, "-m", "pip", "install", 
                                "transformers>=4.56.0.dev0", "--pre", "--quiet"
                            ])
                        except:
                            pass  # Continue with what we have
                        
                        # Force restart Python modules
                        import importlib
                        import sys
                        
                        # Remove transformers from cache
                        modules_to_reload = [name for name in sys.modules.keys() if name.startswith('transformers')]
                        for module_name in modules_to_reload:
                            if module_name in sys.modules:
                                del sys.modules[module_name]
                        
                        # Re-import transformers
                        import transformers
                        from transformers import AutoModelForCausalLM, AutoTokenizer
                        
                        print(f"Transformers version: {transformers.__version__}")
                        
                        # Set up GPT-OSS specific parameters
                        gpt_oss_kwargs = load_kwargs.copy()
                        gpt_oss_kwargs.update({
                            'trust_remote_code': True,
                            'torch_dtype': 'auto',
                            'device_map': 'auto',
                            'attn_implementation': 'flash_attention_2' if 'cuda' in str(device).lower() else 'eager'
                        })
                        
                        self.base_model = AutoModelForCausalLM.from_pretrained(actual_model_path, **gpt_oss_kwargs)
                        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path, **gpt_oss_kwargs)
                        print("✓ GPT-OSS-20B model loaded successfully with bleeding-edge transformers!")
                        success = True
                        
                    except Exception as e1:
                        print(f"Method 1 failed: {str(e1)[:150]}...")
                
                # Method 2: Manual model registration approach
                if not success:
                    try:
                        print("Method 2: Attempting manual GPT-OSS model registration...")
                        
                        # Try to manually register the GPT-OSS model type
                        from transformers import AutoConfig
                        from transformers.models.auto import configuration_auto
                        
                        # Register a temporary GPT-OSS config mapping to GPT-2 for now
                        if hasattr(configuration_auto, 'CONFIG_MAPPING'):
                            try:
                                from transformers.models.gpt2.configuration_gpt2 import GPT2Config
                                configuration_auto.CONFIG_MAPPING._extra_content['gpt_oss'] = GPT2Config
                                print("Registered GPT-OSS as GPT-2 variant")
                            except:
                                pass
                        
                        # Try loading with trust_remote_code and custom config
                        special_kwargs = load_kwargs.copy()
                        special_kwargs.update({
                            'trust_remote_code': True,
                            'torch_dtype': 'auto',
                            'device_map': 'auto',
                            'use_safetensors': True
                        })
                        
                        self.base_model = AutoModelForCausalLM.from_pretrained(actual_model_path, **special_kwargs)
                        self.tokenizer = AutoTokenizer.from_pretrained(actual_model_path, **special_kwargs)
                        print("✓ GPT-OSS loaded with manual registration!")
                        success = True
                        
                    except Exception as e2:
                        print(f"Method 2 failed: {str(e2)[:150]}...")
                
                # Method 3: Try with pipeline and custom loading
                if not success:
                    try:
                        print("Method 3: Attempting custom pipeline loading...")
                        
                        # Set environment variables that might help
                        os.environ['TRANSFORMERS_TRUST_REMOTE_CODE'] = 'true'
                        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'
                        
                        from transformers import pipeline
                        
                        pipeline_kwargs = {
                            'trust_remote_code': True,
                            'torch_dtype': 'auto',
                            'device_map': 'auto',
                            'model_kwargs': {
                                'trust_remote_code': True,
                                'torch_dtype': 'auto'
                            }
                        }
                        
                        self._pipeline = pipeline("text-generation", model=actual_model_path, **pipeline_kwargs)
                        self.base_model = self._pipeline.model
                        self.tokenizer = self._pipeline.tokenizer
                        print("✓ GPT-OSS loaded via custom pipeline!")
                        success = True
                        
                    except Exception as e3:
                        print(f"Method 3 failed: {str(e3)[:150]}...")
                
                if not success:
                    print("\n" + "="*80)
                    print("❌ UNABLE TO LOAD GPT-OSS-20B")
                    print("="*80)
                    print("The GPT-OSS-20B model architecture is not yet supported in the current")
                    print("transformers library. This is a limitation of the transformers library,")
                    print("not the HCWS framework.")
                    print()
                    print("POSSIBLE SOLUTIONS:")
                    print("1. Wait for official GPT-OSS support in transformers")
                    print("2. Use OpenAI's official GPT-OSS tools from their repository")
                    print("3. Run this on a system with Python 3.12+ and the gpt-oss package")
                    print()
                    print("For now, the script will need to use a compatible model.")
                    print("="*80)
                    
                    raise ValueError(f"GPT-OSS-20B architecture not supported in current transformers library. "
                                   f"Original error: {str(e)[:200]}... "
                                   f"This is a limitation of transformers, not HCWS. "
                                   f"Please use a compatible model or wait for official GPT-OSS support.")
            else:
                raise e
        
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
        
        # Training instruction tracking for automatic retraining
        self.trained_instructions = set()  # Track unique instructions this model has been trained on
        
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
            # Check if retraining is needed for this instruction
            if hasattr(self, 'needs_retraining') and self.needs_retraining([steering_instruction]):
                print(f"\n⚠️  WARNING: Instruction '{steering_instruction}' not in trained set!")
                print(f"   🧠 Hypernetwork may not steer effectively for this instruction.")
                if self.trained_instructions:
                    print(f"   📝 Trained instructions: {list(self.trained_instructions)}")
                else:
                    print(f"   📝 No instructions have been trained yet.")
                print(f"   🔄 Consider retraining: model.retrain_for_instructions(['{steering_instruction}'])")
                print(f"   📊 Or use: train_hcws_model_with_instruction_check(model, training_data)")
            
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
            'trained_instructions': list(self.trained_instructions),  # Save trained instructions
            'config': {
                'hidden_dim': self.hidden_dim,
                'num_layers': self.num_layers,
                'conceptor_rank': self.hyper_network.conceptor_rank,
                'controller_dim': self.controller.controller_dim
            }
        }
        
        torch.save(state, path)
        print(f"\n💾 Saved hypernetwork to: {path}")
        if hasattr(self, 'trained_instructions') and self.trained_instructions:
            print(f"📝 Includes training for {len(self.trained_instructions)} instructions: {list(self.trained_instructions)}")
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
        
        # Load trained instructions if available
        if 'trained_instructions' in state:
            self.trained_instructions = set(state['trained_instructions'])
        else:
            # For compatibility with older saved models
            self.trained_instructions = set()
        
        logger.info(f"Loaded steering components from {path}")
        if hasattr(self, 'trained_instructions') and self.trained_instructions:
            print(f"\n📝 Loaded hypernetwork trained on {len(self.trained_instructions)} instructions:")
            for i, inst in enumerate(sorted(self.trained_instructions), 1):
                print(f"   {i}. '{inst}'")
        else:
            print(f"\n⚠️  Loaded hypernetwork with no recorded training instructions")
            print(f"   Consider training with: model.retrain_for_instructions(['your_instruction'])")
    
    def needs_retraining(self, new_instructions: List[str]) -> bool:
        """
        Check if the model needs retraining based on new instructions.
        
        Args:
            new_instructions: List of instructions to check
            
        Returns:
            True if retraining is needed, False otherwise
        """
        new_instruction_set = set(new_instructions)
        return not new_instruction_set.issubset(self.trained_instructions)
    
    def update_trained_instructions(self, instructions: List[str]):
        """
        Update the set of trained instructions.
        
        Args:
            instructions: List of instructions that the model has been trained on
        """
        self.trained_instructions.update(instructions)
    
    def retrain_for_instructions(
        self,
        new_instructions: List[str],
        training_data_path: Optional[str] = None,
        output_path: Optional[str] = None,
        learning_rate: float = 1e-4,
        epochs: int = 10,
        batch_size: int = 4,
        force_retrain: bool = False,
        verbose: bool = True,
        **training_kwargs
    ):
        """
        Retrain the hypernetwork if new instructions are detected.
        
        This method will automatically check if the provided instructions require
        retraining and will only train if new instructions are found.
        
        Args:
            new_instructions: List of instructions to check/train for
            training_data_path: Path to training data (if None, will use default)
            output_path: Path to save retrained model
            learning_rate: Learning rate for training
            epochs: Number of training epochs
            batch_size: Training batch size
            force_retrain: Force retraining even if instructions are known
            verbose: Whether to print progress
            **training_kwargs: Additional training arguments
            
        Returns:
            True if retraining was performed, False otherwise
        """
        # Import here to avoid circular imports
        from .training import train_hcws_model_with_instruction_check
        
        if not force_retrain and not self.needs_retraining(new_instructions):
            if verbose:
                print("\n✅ All instructions already trained. No retraining needed.")
                print(f"   📝 Known instructions: {list(self.trained_instructions)}")
            return False
        
        if verbose:
            if force_retrain:
                print("\n🔄 Force retraining requested.")
            else:
                new_inst = set(new_instructions) - self.trained_instructions
                print(f"\n📝 Retraining for new instructions: {new_inst}")
            print("🧠 Starting hypernetwork training for instruction-based steering...")
        
        # Create training data with the new instructions
        training_data = None
        if training_data_path is None:
            # Create default training data including the new instructions
            from .training import create_default_training_data
            training_data = create_default_training_data()
            
            # Add simple examples for new instructions if they're not covered
            for instruction in new_instructions:
                # Check if instruction is already in training data
                found = False
                for example in training_data['positive'] + training_data['negative']:
                    if example.get('instruction') == instruction:
                        found = True
                        break
                
                if not found:
                    # Add a basic positive example for this instruction
                    training_data['positive'].append({
                        'instruction': instruction,
                        'input': 'Please help me with this task.',
                        'output': f'I will help you with {instruction} in mind.'
                    })
        
        # Perform retraining
        history = train_hcws_model_with_instruction_check(
            model=self,
            training_data=training_data,
            data_path=training_data_path,
            force_retrain=True,  # We've already checked
            learning_rate=learning_rate,
            epochs=epochs,
            batch_size=batch_size,
            save_path=output_path,
            verbose=verbose,
            **training_kwargs
        )
        
        if verbose:
            print("\n🎉 Hypernetwork retraining completed!")
            print("✅ Model now optimized for the new instructions.")
            if output_path:
                print(f"💾 Updated model saved to: {output_path}")
        
        return True
    
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