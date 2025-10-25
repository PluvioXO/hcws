"""
Steering controller for real-time activation modification during inference.

The controller produces scalar gains g_t and per-layer weights w_{ℓ,t} that
determine how strongly to apply the conceptor-based steering at each step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, TYPE_CHECKING
import logging

logger = logging.getLogger(__name__)


class SteeringController(nn.Module):
    """
    Controller that determines steering parameters during inference.
    
    The controller takes the current hidden state and produces:
    - g_t: scalar gain for overall steering strength
    - w_{ℓ,t}: per-layer weights for layer-specific steering
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        controller_dim: int = 128,
        steering_strength: float = 1.0,
        use_attention: bool = True,
        use_temporal_smoothing: bool = True,
        smoothing_factor: float = 0.9,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the steering controller.
        
        Args:
            hidden_dim: Hidden dimension of the target model
            num_layers: Number of layers in the target model
            controller_dim: Internal dimension of the controller
            steering_strength: Multiplier for steering intensity (default: 1.0)
            use_attention: Whether to use attention mechanism
            use_temporal_smoothing: Whether to smooth gains over time
            smoothing_factor: Smoothing factor for temporal smoothing
            device: Device to run computations on
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.controller_dim = controller_dim
        self.steering_strength = steering_strength
        self.use_attention = use_attention
        self.use_temporal_smoothing = use_temporal_smoothing
        self.smoothing_factor = smoothing_factor
        self.dtype = dtype
        from .device_utils import get_device
        raw_device = get_device(device)
        
        # Convert to torch.device for consistent handling
        if isinstance(raw_device, str):
            self.device = torch.device(raw_device)
        else:
            self.device = raw_device
        
        # Input projection
        self.input_proj = nn.Linear(hidden_dim, controller_dim)
        
        # Controller layers
        self.controller_layers = nn.Sequential(
            nn.Linear(controller_dim, controller_dim),
            nn.LayerNorm(controller_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(controller_dim, controller_dim),
            nn.LayerNorm(controller_dim),
            nn.ReLU()
        )
        
        # Output heads
        self.gain_head = nn.Linear(controller_dim, 1)
        self.weight_head = nn.Linear(controller_dim, num_layers)
        
        # Attention mechanism for contextual steering
        if use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=controller_dim,
                num_heads=4,
                dropout=0.1,
                batch_first=True
            )
        
        # Temporal smoothing state
        if use_temporal_smoothing:
            self.register_buffer('prev_gain', torch.tensor(0.0))
            self.register_buffer('prev_weights', torch.zeros(num_layers))
        
        # Initialize weights
        self._initialize_weights()
        
        # Move controller to the requested device/dtype
        self.to(self.device, dtype=dtype)
        
        logger.info(f"Initialized SteeringController with {num_layers} layers")
        logger.info(f"Using dtype: {dtype}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute steering parameters for the current step.
        
        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_dim]
            layer_idx: Current layer index
            context: Optional context for attention
            
        Returns:
            Tuple of (gain, layer_weights)
            - gain: Scalar gain [batch_size, 1]
            - layer_weights: Per-layer weights [batch_size, num_layers]
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Use the last token's hidden state for steering decisions
        current_hidden = hidden_states[:, -1, :]  # [batch_size, hidden_dim]
        
        # Ensure dtype consistency - convert to controller's dtype for processing
        original_dtype = current_hidden.dtype
        if current_hidden.dtype != self.dtype:
            current_hidden = current_hidden.to(self.dtype)
        
        # Project to controller dimension
        controller_input = self.input_proj(current_hidden)  # [batch_size, controller_dim]
        
        # Apply attention if enabled
        if self.use_attention and context is not None:
            controller_input_expanded = controller_input.unsqueeze(1)  # [batch_size, 1, controller_dim]
            attended, _ = self.attention(
                controller_input_expanded,
                context,
                context
            )
            controller_input = attended.squeeze(1)  # [batch_size, controller_dim]
        
        # Pass through controller layers
        controller_output = self.controller_layers(controller_input)
        
        # Generate gain and weights
        gain_raw = self.gain_head(controller_output)  # [batch_size, 1]
        gain = torch.sigmoid(gain_raw) * self.steering_strength  # Scale up gain to [0, 5] range
        
        weight_raw = self.weight_head(controller_output)  # [batch_size, num_layers]
        layer_weights = torch.softmax(weight_raw, dim=1) * self.steering_strength  # Scale up weights to [0, 2] range
        
        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing and self.training:
            gain = self.smoothing_factor * self.prev_gain + (1 - self.smoothing_factor) * gain
            layer_weights = self.smoothing_factor * self.prev_weights + (1 - self.smoothing_factor) * layer_weights
            
            # Update previous values
            self.prev_gain.copy_(gain.mean().detach())
            self.prev_weights.copy_(layer_weights.mean(dim=0).detach())
        
        # Convert back to original dtype for compatibility with hidden_states
        if original_dtype != self.dtype:
            gain = gain.to(original_dtype)
            layer_weights = layer_weights.to(original_dtype)
        
        return gain, layer_weights
    
    def reset_state(self):
        """Reset temporal smoothing state."""
        if self.use_temporal_smoothing:
            self.prev_gain.fill_(0.0)
            self.prev_weights.fill_(0.0)


class AdaptiveSteeringController(SteeringController):
    """
    Adaptive controller that adjusts steering based on generation quality.
    
    This controller monitors the quality of generated text and adjusts
    steering strength accordingly to maintain fluency while achieving
    the desired behavior.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        controller_dim: int = 128,
        quality_threshold: float = 0.8,
        adaptation_rate: float = 0.1,
        use_attention: bool = True,
        use_temporal_smoothing: bool = True,
        smoothing_factor: float = 0.9,
        device: Optional[str] = None
    ):
        """
        Initialize the adaptive steering controller.
        
        Args:
            quality_threshold: Threshold for quality-based adaptation
            adaptation_rate: Rate of adaptation
            (other args same as SteeringController)
        """
        super().__init__(
            hidden_dim, num_layers, controller_dim, use_attention,
            use_temporal_smoothing, smoothing_factor, device
        )
        
        self.quality_threshold = quality_threshold
        self.adaptation_rate = adaptation_rate
        
        # Quality estimation network
        self.quality_estimator = nn.Sequential(
            nn.Linear(controller_dim, controller_dim // 2),
            nn.ReLU(),
            nn.Linear(controller_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # Adaptation state
        self.register_buffer('adaptation_factor', torch.tensor(1.0))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute adaptive steering parameters.
        
        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_dim]
            layer_idx: Current layer index
            context: Optional context for attention
            
        Returns:
            Tuple of (gain, layer_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get base steering parameters
        gain, layer_weights = super().forward(hidden_states, layer_idx, context)
        
        # Estimate quality
        current_hidden = hidden_states[:, -1, :]
        controller_input = self.input_proj(current_hidden)
        controller_output = self.controller_layers(controller_input)
        
        quality = self.quality_estimator(controller_output)  # [batch_size, 1]
        
        # Adapt steering strength based on quality
        if self.training:
            # Update adaptation factor
            mean_quality = quality.mean()
            if mean_quality < self.quality_threshold:
                # Reduce steering if quality is low
                self.adaptation_factor = torch.clamp(
                    self.adaptation_factor - self.adaptation_rate,
                    min=0.1
                )
            else:
                # Increase steering if quality is high
                self.adaptation_factor = torch.clamp(
                    self.adaptation_factor + self.adaptation_rate,
                    max=2.0
                )
        
        # Apply adaptation
        gain = gain * self.adaptation_factor
        
        return gain, layer_weights
    
    def get_quality_scores(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get quality scores for the current hidden states.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, hidden_dim]
            
        Returns:
            Quality scores [batch_size, 1]
        """
        current_hidden = hidden_states[:, -1, :]
        controller_input = self.input_proj(current_hidden)
        controller_output = self.controller_layers(controller_input)
        
        return self.quality_estimator(controller_output)


class MultiModalController(SteeringController):
    """
    Controller that handles multiple steering modalities simultaneously.
    
    This controller can manage different types of steering instructions
    (e.g., style, sentiment, factuality) and balance them appropriately.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        num_modalities: int,
        controller_dim: int = 128,
        use_attention: bool = True,
        use_temporal_smoothing: bool = True,
        smoothing_factor: float = 0.9,
        device: Optional[str] = None
    ):
        """
        Initialize the multi-modal controller.
        
        Args:
            num_modalities: Number of steering modalities
            (other args same as SteeringController)
        """
        super().__init__(
            hidden_dim, num_layers, controller_dim, use_attention,
            use_temporal_smoothing, smoothing_factor, device
        )
        
        self.num_modalities = num_modalities
        
        # Modality-specific heads
        self.modality_gains = nn.ModuleList([
            nn.Linear(controller_dim, 1) for _ in range(num_modalities)
        ])
        
        self.modality_weights = nn.ModuleList([
            nn.Linear(controller_dim, num_layers) for _ in range(num_modalities)
        ])
        
        # Modality balancing
        self.modality_balancer = nn.Linear(controller_dim, num_modalities)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        layer_idx: int,
        context: Optional[torch.Tensor] = None,
        modality_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-modal steering parameters.
        
        Args:
            hidden_states: Current hidden states [batch_size, seq_len, hidden_dim]
            layer_idx: Current layer index
            context: Optional context for attention
            modality_mask: Mask for active modalities [batch_size, num_modalities]
            
        Returns:
            Tuple of (gain, layer_weights)
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Get controller output
        current_hidden = hidden_states[:, -1, :]
        controller_input = self.input_proj(current_hidden)
        
        if self.use_attention and context is not None:
            controller_input_expanded = controller_input.unsqueeze(1)
            attended, _ = self.attention(
                controller_input_expanded,
                context,
                context
            )
            controller_input = attended.squeeze(1)
        
        controller_output = self.controller_layers(controller_input)
        
        # Compute modality balance
        modality_balance = torch.softmax(
            self.modality_balancer(controller_output), dim=1
        )  # [batch_size, num_modalities]
        
        # Apply modality mask if provided
        if modality_mask is not None:
            modality_balance = modality_balance * modality_mask
            modality_balance = modality_balance / (modality_balance.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute modality-specific parameters
        modality_gains = []
        modality_layer_weights = []
        
        for i in range(self.num_modalities):
            gain = torch.sigmoid(self.modality_gains[i](controller_output))
            weights = torch.softmax(self.modality_weights[i](controller_output), dim=1)
            
            modality_gains.append(gain)
            modality_layer_weights.append(weights)
        
        # Combine modalities
        combined_gain = torch.zeros_like(modality_gains[0])
        combined_weights = torch.zeros_like(modality_layer_weights[0])
        
        for i in range(self.num_modalities):
            balance = modality_balance[:, i:i+1]
            combined_gain += balance * modality_gains[i]
            combined_weights += balance.unsqueeze(2) * modality_layer_weights[i]
        
        return combined_gain, combined_weights


def create_controller_from_config(config: Dict[str, Any]) -> SteeringController:
    """
    Create a steering controller from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized steering controller
    """
    controller_type = config.get('type', 'basic')
    
    if controller_type == 'adaptive':
        return AdaptiveSteeringController(**config.get('params', {}))
    elif controller_type == 'multimodal':
        return MultiModalController(**config.get('params', {}))
    else:
        return SteeringController(**config.get('params', {})) 


class BOHBSteeringStrengthOptimizer:
    """
    Bayesian Optimization-based Hyperband (BOHB) Steering Strength Optimizer.
    
    Uses Bayesian optimization to find the optimal steering strength
    for a given instruction and model.
    """
    
    def __init__(
        self, 
        model: 'HCWSModel', 
        instruction: str,
        initial_strength_range: Tuple[float, float] = (0.5, 10.0),
        num_trials: int = 20,
        num_iterations: int = 5
    ):
        """
        Initialize the BOHB steering strength optimizer.
        
        Args:
            model: HCWS model to optimize
            instruction: Steering instruction
            initial_strength_range: Initial range for steering strength
            num_trials: Number of trials for optimization
            num_iterations: Number of iterations per trial
        """
        # Lazy import to break circular dependency
        from .model import HCWSModel
        
        if not isinstance(model, HCWSModel):
            raise TypeError("model must be an instance of HCWSModel")
        
        self.model = model
        self.instruction = instruction
        self.initial_strength_range = initial_strength_range
        self.num_trials = num_trials
        self.num_iterations = num_iterations
        
        # Import optimization libraries
        try:
            import ConfigSpace as CS
            import ConfigSpace.hyperparameters as CSH
            import hpbandster.core.nameserver as hpns
            import hpbandster.core.result as hpres
            import hpbandster.optimizers.bohb as bohb
            
            self.CS = CS
            self.CSH = CSH
            self.hpns = hpns
            self.hpres = hpres
            self.bohb = bohb
        except ImportError:
            raise ImportError("Please install ConfigSpace, hpbandster for BOHB optimization") 