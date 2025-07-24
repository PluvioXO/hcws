"""
Steering controller for real-time activation modification during inference.

The controller produces scalar gains g_t and per-layer weights w_{ℓ,t} that
determine how strongly to apply the conceptor-based steering at each step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
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
        device: Optional[str] = None
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
        from .device_utils import get_device
        self.device = get_device(device)
        
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
        
        logger.info(f"Initialized SteeringController with {num_layers} layers")
    
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
    BOHB-inspired optimizer for finding optimal steering strength.
    
    Combines Bayesian Optimization with Hyperband-like resource allocation.
    """
    
    def __init__(
        self, 
        model: HCWSModel, 
        instruction: str,
        max_iterations: int = 30,
        eta: float = 3.0,  # Hyperband scaling parameter
        min_budget: float = 1.0,
        max_budget: float = 10.0
    ):
        """
        Initialize the BOHB-inspired steering strength optimizer.
        
        Args:
            model: HCWS model to optimize
            instruction: Steering instruction
            max_iterations: Maximum number of optimization iterations
            eta: Hyperband scaling parameter
            min_budget: Minimum steering strength to explore
            max_budget: Maximum steering strength to explore
        """
        self.model = model
        self.instruction = instruction
        self.max_iterations = max_iterations
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        
        # Bayesian optimization components
        from skopt import Optimizer
        from skopt.space import Real
        
        self.optimizer = Optimizer(
            dimensions=[Real(min_budget, max_budget, 'log-uniform')],
            base_estimator='GP',  # Gaussian Process
            acq_func='EI',  # Expected Improvement
            n_initial_points=5
        )
        
        # Tracking optimization history
        self.trials = []
        self.performance_history = []
    
    def _compute_quality_metrics(self, steering_strength: float) -> Dict[str, float]:
        """
        Compute multiple quality metrics for a given steering strength.
        
        Args:
            steering_strength: Steering strength to evaluate
            
        Returns:
            Dictionary of quality metrics
        """
        # Temporarily set steering strength
        original_strength = self.model.steering_strength
        self.model.steering_strength = steering_strength
        
        try:
            # Generate multiple samples with different evaluation metrics
            samples = self.model.generate_with_multiple_instructions(
                input_text="The future of AI is",
                steering_instructions=[self.instruction] * 5,
                max_length=50
            )
            
            # Compute multiple quality metrics
            metrics = {
                'perplexity': self._compute_perplexity(samples),
                'coherence': self._compute_coherence(samples),
                'diversity': self._compute_sample_diversity(samples)
            }
            
            # Combine metrics into a single score
            combined_score = (
                -metrics['perplexity'] +  # Lower perplexity is better
                metrics['coherence'] + 
                metrics['diversity']
            )
            
            return {
                'combined_score': combined_score,
                **metrics
            }
        
        finally:
            # Restore original steering strength
            self.model.steering_strength = original_strength
    
    def _compute_perplexity(self, samples: List[str]) -> float:
        """Compute average perplexity for generated samples."""
        perplexities = []
        for sample in samples:
            inputs = self.model.tokenizer(
                sample, 
                return_tensors='pt', 
                truncation=True
            ).to(self.model.device)
            
            with torch.no_grad():
                outputs = self.model.base_model(
                    input_ids=inputs.input_ids, 
                    labels=inputs.input_ids
                )
                perplexity = torch.exp(outputs.loss).item()
                perplexities.append(perplexity)
        
        return np.mean(perplexities)
    
    def _compute_coherence(self, samples: List[str]) -> float:
        """Compute semantic coherence between samples."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(samples)
        
        # Compute pairwise cosine similarities
        similarities = [
            np.dot(embeddings[i], embeddings[j]) / 
            (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))
            for i in range(len(embeddings))
            for j in range(i+1, len(embeddings))
        ]
        
        return np.mean(similarities)
    
    def _compute_sample_diversity(self, samples: List[str]) -> float:
        """Compute semantic diversity between samples."""
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('all-MiniLM-L6-v2')
        embeddings = model.encode(samples)
        
        # Compute pairwise distances
        distances = [
            np.linalg.norm(embeddings[i] - embeddings[j])
            for i in range(len(embeddings))
            for j in range(i+1, len(embeddings))
        ]
        
        return np.mean(distances)
    
    def _successive_halving(self, configurations: List[float]) -> List[float]:
        """
        Implement Successive Halving for steering strength optimization.
        
        Args:
            configurations: List of steering strengths to evaluate
            
        Returns:
            Reduced list of most promising configurations
        """
        # Compute performance for all configurations
        performances = [
            self._compute_quality_metrics(config)['combined_score'] 
            for config in configurations
        ]
        
        # Sort configurations by performance
        sorted_configs = [
            x for _, x in sorted(
                zip(performances, configurations), 
                key=lambda pair: pair[0], 
                reverse=True
            )
        ]
        
        # Keep top 1/eta configurations
        return sorted_configs[:max(1, len(sorted_configs) // int(self.eta))]
    
    def find_optimal_strength(self) -> float:
        """
        Find the optimal steering strength using BOHB-inspired approach.
        
        Returns:
            Optimal steering strength
        """
        # Initial random sampling
        initial_configs = [
            self.min_budget + (self.max_budget - self.min_budget) * np.random.random()
            for _ in range(max(5, self.max_iterations // 2))
        ]
        
        # Track best configuration
        best_strength = None
        best_score = float('-inf')
        
        # Iterative optimization with Successive Halving
        for iteration in range(self.max_iterations):
            # Evaluate current configurations
            performances = [
                self._compute_quality_metrics(config)['combined_score'] 
                for config in initial_configs
            ]
            
            # Update best configuration
            current_best_idx = np.argmax(performances)
            if performances[current_best_idx] > best_score:
                best_score = performances[current_best_idx]
                best_strength = initial_configs[current_best_idx]
            
            # Bayesian optimization suggestion
            x = self.optimizer.ask()
            y = self._compute_quality_metrics(x[0])['combined_score']
            self.optimizer.tell(x, y)
            
            # Successive Halving
            initial_configs = self._successive_halving(initial_configs)
            
            # Add Bayesian optimization suggestion to configurations
            initial_configs.append(x[0])
        
        # Final optimization using Bayesian optimization
        result = self.optimizer.get_result()
        final_best_strength = result.x[0]
        
        # Compare with previous best
        final_score = self._compute_quality_metrics(final_best_strength)['combined_score']
        if final_score > best_score:
            best_strength = final_best_strength
        
        return best_strength 