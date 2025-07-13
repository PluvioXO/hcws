"""
Hyper-network that maps instruction vectors to layer-specific conceptors.

The hyper-network takes dense instruction vectors z from the T5 encoder and
generates the parameters for conceptor matrices C_ℓ = U_ℓ diag(s_ℓ) U_ℓᵀ
for each layer ℓ of the target model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import logging

from .conceptors import ConceptorBank

logger = logging.getLogger(__name__)


class HyperNetwork(nn.Module):
    """
    Hyper-network that maps instruction embeddings to conceptor parameters.
    
    Given an instruction embedding z, generates layer-specific conceptor
    parameters (U_ℓ, s_ℓ) for each layer ℓ of the target model.
    """
    
    def __init__(
        self,
        instruction_dim: int,
        num_layers: int,
        hidden_dim: int,
        conceptor_rank: int,
        hyper_hidden_dim: int = 256,
        num_hyper_layers: int = 3,
        dropout: float = 0.1,
        use_layer_embedding: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the hyper-network.
        
        Args:
            instruction_dim: Dimension of instruction embeddings
            num_layers: Number of target model layers
            hidden_dim: Hidden dimension of target model
            conceptor_rank: Rank of conceptor matrices
            hyper_hidden_dim: Hidden dimension of hyper-network
            num_hyper_layers: Number of layers in hyper-network
            dropout: Dropout rate
            use_layer_embedding: Whether to use layer embeddings
            device: Device to run computations on
        """
        super().__init__()
        
        self.instruction_dim = instruction_dim
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.conceptor_rank = conceptor_rank
        self.hyper_hidden_dim = hyper_hidden_dim
        self.use_layer_embedding = use_layer_embedding
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Layer embeddings for layer-specific generation
        if use_layer_embedding:
            self.layer_embedding = nn.Embedding(num_layers, hyper_hidden_dim)
            input_dim = instruction_dim + hyper_hidden_dim
        else:
            input_dim = instruction_dim
        
        # Shared hyper-network layers
        layers = []
        prev_dim = input_dim
        
        for i in range(num_hyper_layers):
            layers.append(nn.Linear(prev_dim, hyper_hidden_dim))
            layers.append(nn.LayerNorm(hyper_hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hyper_hidden_dim
            
        self.shared_layers = nn.Sequential(*layers)
        
        # Separate heads for U and s generation
        self.u_head = nn.Linear(hyper_hidden_dim, hidden_dim * conceptor_rank)
        self.s_head = nn.Linear(hyper_hidden_dim, conceptor_rank)
        
        # Initialize weights
        self._initialize_weights()
        
        logger.info(f"Initialized HyperNetwork with {num_layers} layers")
        logger.info(f"Conceptor rank: {conceptor_rank}, Hidden dim: {hidden_dim}")
    
    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=0.02)
    
    def forward(self, instruction_embedding: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generate conceptor parameters for all layers.
        
        Args:
            instruction_embedding: Instruction embedding [batch_size, instruction_dim]
            
        Returns:
            Dictionary containing 'U' and 's' parameters for all layers
        """
        batch_size = instruction_embedding.shape[0]
        
        # Generate parameters for each layer
        U_params = torch.zeros(
            batch_size, self.num_layers, self.hidden_dim, self.conceptor_rank,
            device=self.device
        )
        s_params = torch.zeros(
            batch_size, self.num_layers, self.conceptor_rank,
            device=self.device
        )
        
        for layer_idx in range(self.num_layers):
            # Prepare input
            if self.use_layer_embedding:
                layer_emb = self.layer_embedding(
                    torch.tensor([layer_idx], device=self.device)
                ).expand(batch_size, -1)
                hyper_input = torch.cat([instruction_embedding, layer_emb], dim=1)
            else:
                hyper_input = instruction_embedding
            
            # Pass through shared layers
            hidden = self.shared_layers(hyper_input)
            
            # Generate U parameters
            u_flat = self.u_head(hidden)
            U_params[:, layer_idx] = u_flat.view(
                batch_size, self.hidden_dim, self.conceptor_rank
            )
            
            # Generate s parameters
            s_raw = self.s_head(hidden)
            s_params[:, layer_idx] = torch.sigmoid(s_raw)  # Ensure s ∈ [0, 1]
        
        return {
            'U': U_params,
            's': s_params
        }
    
    def generate_conceptor_bank(
        self,
        instruction_embedding: torch.Tensor,
        regularization: float = 1e-3
    ) -> ConceptorBank:
        """
        Generate a conceptor bank from instruction embedding.
        
        Args:
            instruction_embedding: Instruction embedding [instruction_dim]
            regularization: Regularization parameter
            
        Returns:
            ConceptorBank with generated parameters
        """
        if instruction_embedding.dim() == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0)
        
        # Generate parameters
        params = self.forward(instruction_embedding)
        U_params = params['U'][0]  # Take first batch element
        s_params = params['s'][0]
        
        # Create conceptor bank
        bank = ConceptorBank(
            self.num_layers,
            self.hidden_dim,
            self.conceptor_rank,
            regularization,
            self.device
        )
        
        # Set generated parameters
        with torch.no_grad():
            for layer_idx in range(self.num_layers):
                conceptor = bank.get_conceptor(layer_idx)
                
                # Orthogonalize U
                U_layer = U_params[layer_idx]
                U_ortho, _ = torch.qr(U_layer)
                
                conceptor.U.data.copy_(U_ortho)
                conceptor.s.data.copy_(s_params[layer_idx])
        
        return bank
    
    def generate_layer_conceptor(
        self,
        instruction_embedding: torch.Tensor,
        layer_idx: int,
        regularization: float = 1e-3
    ) -> torch.Tensor:
        """
        Generate conceptor matrix for a specific layer.
        
        Args:
            instruction_embedding: Instruction embedding [instruction_dim]
            layer_idx: Target layer index
            regularization: Regularization parameter
            
        Returns:
            Conceptor matrix [hidden_dim, hidden_dim]
        """
        if instruction_embedding.dim() == 1:
            instruction_embedding = instruction_embedding.unsqueeze(0)
        
        # Prepare input
        if self.use_layer_embedding:
            layer_emb = self.layer_embedding(
                torch.tensor([layer_idx], device=self.device)
            ).expand(1, -1)
            hyper_input = torch.cat([instruction_embedding, layer_emb], dim=1)
        else:
            hyper_input = instruction_embedding
        
        # Pass through shared layers
        hidden = self.shared_layers(hyper_input)
        
        # Generate parameters
        u_flat = self.u_head(hidden)
        U = u_flat.view(1, self.hidden_dim, self.conceptor_rank)[0]
        s = torch.sigmoid(self.s_head(hidden))[0]
        
        # Orthogonalize U
        U_ortho, _ = torch.qr(U)
        
        # Compute conceptor matrix C = U diag(s) U^T
        C = torch.mm(U_ortho * s.unsqueeze(0), U_ortho.t())
        
        return C
    
    def compute_regularization_loss(self) -> torch.Tensor:
        """
        Compute regularization loss for the hyper-network.
        
        Returns:
            Regularization loss
        """
        reg_loss = torch.tensor(0.0, device=self.device)
        
        # L2 regularization on all parameters
        for param in self.parameters():
            reg_loss += torch.norm(param) ** 2
        
        return reg_loss * 1e-4


class AdaptiveHyperNetwork(HyperNetwork):
    """
    Adaptive hyper-network that can condition on context during generation.
    
    This variant can adapt the conceptor generation based on additional
    context such as the current token position or activation statistics.
    """
    
    def __init__(
        self,
        instruction_dim: int,
        num_layers: int,
        hidden_dim: int,
        conceptor_rank: int,
        context_dim: int = 0,
        hyper_hidden_dim: int = 256,
        num_hyper_layers: int = 3,
        dropout: float = 0.1,
        use_layer_embedding: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize the adaptive hyper-network.
        
        Args:
            context_dim: Dimension of context vectors
            (other args same as HyperNetwork)
        """
        self.context_dim = context_dim
        
        # Modify instruction dim to include context
        effective_instruction_dim = instruction_dim + context_dim
        
        super().__init__(
            effective_instruction_dim,
            num_layers,
            hidden_dim,
            conceptor_rank,
            hyper_hidden_dim,
            num_hyper_layers,
            dropout,
            use_layer_embedding,
            device
        )
    
    def forward_with_context(
        self,
        instruction_embedding: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Generate conceptor parameters with context conditioning.
        
        Args:
            instruction_embedding: Instruction embedding [batch_size, instruction_dim]
            context: Context vector [batch_size, context_dim]
            
        Returns:
            Dictionary containing 'U' and 's' parameters for all layers
        """
        if context is not None:
            # Concatenate instruction and context
            combined_input = torch.cat([instruction_embedding, context], dim=1)
        else:
            # Pad with zeros if no context
            batch_size = instruction_embedding.shape[0]
            zero_context = torch.zeros(batch_size, self.context_dim, device=self.device)
            combined_input = torch.cat([instruction_embedding, zero_context], dim=1)
        
        return super().forward(combined_input)


def create_hyper_network_from_config(config: Dict) -> HyperNetwork:
    """
    Create a hyper-network from configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized hyper-network
    """
    if config.get('adaptive', False):
        return AdaptiveHyperNetwork(**config)
    else:
        return HyperNetwork(**config) 