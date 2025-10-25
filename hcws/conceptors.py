"""
Conceptor operations for defining ellipsoidal subspaces in activation space.

Conceptors are low-rank matrices C = U diag(s) U^T that define ellipsoidal
subspaces aligned with desired behaviors or styles.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging

logger = logging.getLogger(__name__)


class Conceptor(nn.Module):
    """
    A single conceptor matrix C = U diag(s) U^T defining an ellipsoidal subspace.
    
    The conceptor defines a subspace that captures the essential directions
    for a particular behavior or style. During steering, activations are
    contracted toward this subspace.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        rank: int,
        regularization: float = 1e-3,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize a conceptor.
        
        Args:
            hidden_dim: Dimension of the hidden space
            rank: Rank of the conceptor matrix
            regularization: Regularization parameter for stability
            device: Device to run computations on
            dtype: Data type for parameters
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.regularization = regularization
        self.dtype = dtype
        from .device_utils import get_device
        raw_device = get_device(device)
        
        # Convert to torch.device for consistent handling
        if isinstance(raw_device, str):
            self.device = torch.device(raw_device)
        else:
            self.device = raw_device
        
        # Initialize U and s parameters with specified dtype
        self.U = nn.Parameter(torch.randn(hidden_dim, rank, device=self.device, dtype=dtype))
        self.s = nn.Parameter(torch.ones(rank, device=self.device, dtype=dtype) * 0.1)  # Start with smaller values
        
        # Initialize U with orthogonal columns
        self._initialize_orthogonal()
        
    def _initialize_orthogonal(self):
        """Initialize U with orthogonal columns."""
        with torch.no_grad():
            # Use float32 for QR decomposition on CPU (geqrf_cpu doesn't support float16)
            compute_dtype = torch.float32 if self.device.type == 'cpu' and self.dtype == torch.float16 else self.dtype
            U_init = torch.randn(self.hidden_dim, self.rank, device=self.device, dtype=compute_dtype)
            U_init, _ = torch.qr(U_init)
            # Convert back to target dtype if needed
            if compute_dtype != self.dtype:
                U_init = U_init.to(self.dtype)
            self.U.data.copy_(U_init)
    
    def get_matrix(self) -> torch.Tensor:
        """
        Get the full conceptor matrix C = U diag(s) U^T.
        
        Returns:
            Conceptor matrix [hidden_dim, hidden_dim]
        """
        # Use float32 for QR decomposition on CPU (geqrf_cpu doesn't support float16)
        compute_dtype = torch.float32 if self.device.type == 'cpu' and self.U.dtype == torch.float16 else self.U.dtype
        
        # Convert to compute dtype if needed
        U_compute = self.U.to(compute_dtype) if compute_dtype != self.U.dtype else self.U
        
        # Ensure U has orthogonal columns
        U_normalized, _ = torch.qr(U_compute)
        
        # Clamp singular values to ensure stability and reduce aperture
        s_compute = self.s.to(compute_dtype) if compute_dtype != self.s.dtype else self.s
        s_clamped = torch.clamp(s_compute, min=0.0, max=0.5)  # Reduced from 1.0 to 0.5
        
        # Compute C = U diag(s) U^T
        C = torch.mm(U_normalized * s_clamped.unsqueeze(0), U_normalized.t())
        
        # Convert back to original dtype if needed
        if compute_dtype != self.U.dtype:
            C = C.to(self.U.dtype)
        
        return C
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the conceptor to input tensor: C @ x.
        
        Args:
            x: Input tensor [..., hidden_dim]
            
        Returns:
            Transformed tensor [..., hidden_dim]
        """
        original_shape = x.shape
        original_dtype = x.dtype
        x_flat = x.view(-1, self.hidden_dim)
        
        # Get conceptor matrix
        C = self.get_matrix()
        
        # Ensure dtype consistency
        if C.dtype != x_flat.dtype:
            C = C.to(x_flat.dtype)
        
        # Apply conceptor
        output = torch.mm(x_flat, C.t())
        
        return output.view(original_shape)
    
    def contract_activation(self, x: torch.Tensor, strength: float = 1.0) -> torch.Tensor:
        """
        Contract activation toward the conceptor subspace.
        
        Implements: x' = x - strength * C @ x
        
        Args:
            x: Input activation [..., hidden_dim]
            strength: Contraction strength
            
        Returns:
            Contracted activation [..., hidden_dim]
        """
        projection = self.forward(x)
        return x - strength * projection
    
    def get_subspace_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project activation onto the conceptor subspace.
        
        Args:
            x: Input activation [..., hidden_dim]
            
        Returns:
            Projected activation [..., hidden_dim]
        """
        return self.forward(x)
    
    def get_orthogonal_projection(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project activation onto the orthogonal complement of the conceptor subspace.
        
        Args:
            x: Input activation [..., hidden_dim]
            
        Returns:
            Orthogonal projection [..., hidden_dim]
        """
        subspace_proj = self.get_subspace_projection(x)
        return x - subspace_proj
    
    def compute_aperture(self) -> float:
        """
        Compute the aperture (sum of singular values) of the conceptor.
        
        Returns:
            Aperture value
        """
        return torch.sum(self.s).item()
    
    def regularize(self) -> torch.Tensor:
        """
        Compute regularization loss for the conceptor.
        
        Returns:
            Regularization loss
        """
        # Orthogonality regularization for U
        U_gram = torch.mm(self.U.t(), self.U)
        orthogonal_loss = torch.norm(U_gram - torch.eye(self.rank, device=self.device))
        
        # Sparsity regularization for s
        sparsity_loss = torch.sum(self.s ** 2)
        
        return self.regularization * (orthogonal_loss + sparsity_loss)


class ConceptorBank(nn.Module):
    """
    A collection of conceptors for different layers of a transformer model.
    
    The bank manages layer-specific conceptors and provides efficient
    operations for multi-layer steering.
    """
    
    def __init__(
        self,
        num_layers: int,
        hidden_dim: int,
        rank: int,
        regularization: float = 1e-3,
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float32
    ):
        """
        Initialize a conceptor bank.
        
        Args:
            num_layers: Number of transformer layers
            hidden_dim: Hidden dimension for each layer
            rank: Rank for each conceptor
            regularization: Regularization parameter
            device: Device to run computations on
            dtype: Data type for parameters
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.dtype = dtype
        from .device_utils import get_device
        raw_device = get_device(device)
        
        # Convert to torch.device for consistent handling
        if isinstance(raw_device, str):
            self.device = torch.device(raw_device)
        else:
            self.device = raw_device
        
        # Create conceptors for each layer with specified dtype
        self.conceptors = nn.ModuleList([
            Conceptor(hidden_dim, rank, regularization, self.device, dtype)
            for _ in range(num_layers)
        ])

        # Ensure entire bank is on the desired device/dtype
        self.to(self.device, dtype=dtype)
        
        logger.info(f"Initialized ConceptorBank with {num_layers} layers")
    
    def get_conceptor(self, layer_idx: int) -> Conceptor:
        """
        Get conceptor for a specific layer.
        
        Args:
            layer_idx: Layer index
            
        Returns:
            Conceptor for the layer
        """
        return self.conceptors[layer_idx]
    
    def get_all_matrices(self) -> List[torch.Tensor]:
        """
        Get all conceptor matrices.
        
        Returns:
            List of conceptor matrices
        """
        return [conceptor.get_matrix() for conceptor in self.conceptors]
    
    def contract_activations(
        self,
        activations: Dict[int, torch.Tensor],
        strengths: Dict[int, float]
    ) -> Dict[int, torch.Tensor]:
        """
        Contract activations for multiple layers.
        
        Args:
            activations: Dictionary mapping layer_idx to activation tensors
            strengths: Dictionary mapping layer_idx to contraction strengths
            
        Returns:
            Dictionary of contracted activations
        """
        contracted = {}
        
        for layer_idx, activation in activations.items():
            if layer_idx < len(self.conceptors):
                strength = strengths.get(layer_idx, 1.0)
                contracted[layer_idx] = self.conceptors[layer_idx].contract_activation(
                    activation, strength
                )
            else:
                contracted[layer_idx] = activation
                
        return contracted
    
    def compute_total_regularization(self) -> torch.Tensor:
        """
        Compute total regularization loss across all conceptors.
        
        Returns:
            Total regularization loss
        """
        total_loss = torch.tensor(0.0, device=self.device)
        
        for conceptor in self.conceptors:
            total_loss += conceptor.regularize()
            
        return total_loss
    
    def get_apertures(self) -> List[float]:
        """
        Get apertures for all conceptors.
        
        Returns:
            List of aperture values
        """
        return [conceptor.compute_aperture() for conceptor in self.conceptors]
    
    def save_state(self, path: str):
        """
        Save conceptor bank state.
        
        Args:
            path: Path to save state
        """
        state = {
            'num_layers': self.num_layers,
            'hidden_dim': self.hidden_dim,
            'rank': self.rank,
            'state_dict': self.state_dict()
        }
        torch.save(state, path)
        logger.info(f"Saved ConceptorBank state to {path}")
    
    def load_state(self, path: str):
        """
        Load conceptor bank state.
        
        Args:
            path: Path to load state from
        """
        state = torch.load(path, map_location=self.device)
        self.load_state_dict(state['state_dict'])
        logger.info(f"Loaded ConceptorBank state from {path}")


def create_conceptor_from_activations(
    activations: torch.Tensor,
    rank: int,
    regularization: float = 1e-3
) -> Conceptor:
    """
    Create a conceptor from a collection of activations using SVD.
    
    Args:
        activations: Activation tensor [num_samples, hidden_dim]
        rank: Desired rank for the conceptor
        regularization: Regularization parameter
        
    Returns:
        Fitted conceptor
    """
    hidden_dim = activations.shape[1]
    device = activations.device
    
    # Center the activations
    mean_activation = torch.mean(activations, dim=0, keepdim=True)
    centered_activations = activations - mean_activation
    
    # Compute SVD
    U, s, V = torch.svd(centered_activations)
    
    # Take top-k components
    U_k = V[:, :rank]  # V contains right singular vectors
    s_k = s[:rank]
    
    # Normalize singular values
    s_k_normalized = s_k / (s_k.max() + regularization)
    
    # Create conceptor
    conceptor = Conceptor(hidden_dim, rank, regularization, device)
    
    # Initialize with SVD results
    with torch.no_grad():
        conceptor.U.data.copy_(U_k)
        conceptor.s.data.copy_(s_k_normalized)
    
    return conceptor 