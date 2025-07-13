"""
Unit tests for the conceptors module.
"""

import pytest
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hcws.conceptors import Conceptor, ConceptorBank, create_conceptor_from_activations


class TestConceptor:
    """Test cases for Conceptor."""
    
    @pytest.fixture
    def conceptor(self):
        """Create a conceptor for testing."""
        return Conceptor(hidden_dim=128, rank=16, device="cpu")
    
    def test_initialization(self, conceptor):
        """Test conceptor initialization."""
        assert conceptor.hidden_dim == 128
        assert conceptor.rank == 16
        assert conceptor.regularization == 1e-3
        assert conceptor.device == "cpu"
        assert conceptor.U.shape == (128, 16)
        assert conceptor.s.shape == (16,)
    
    def test_get_matrix(self, conceptor):
        """Test getting the full conceptor matrix."""
        C = conceptor.get_matrix()
        
        assert C.shape == (128, 128)
        assert not torch.isnan(C).any()
        assert not torch.isinf(C).any()
        
        # Check symmetry
        assert torch.allclose(C, C.t(), atol=1e-6)
    
    def test_forward(self, conceptor):
        """Test forward pass of conceptor."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, conceptor.hidden_dim)
        
        output = conceptor.forward(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_contract_activation(self, conceptor):
        """Test activation contraction."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, conceptor.hidden_dim)
        
        # Contract with different strengths
        contracted_weak = conceptor.contract_activation(x, strength=0.1)
        contracted_strong = conceptor.contract_activation(x, strength=1.0)
        
        assert contracted_weak.shape == x.shape
        assert contracted_strong.shape == x.shape
        assert not torch.isnan(contracted_weak).any()
        assert not torch.isnan(contracted_strong).any()
        
        # Stronger contraction should be more different from original
        diff_weak = torch.norm(x - contracted_weak)
        diff_strong = torch.norm(x - contracted_strong)
        assert diff_strong >= diff_weak
    
    def test_get_subspace_projection(self, conceptor):
        """Test subspace projection."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, conceptor.hidden_dim)
        
        projection = conceptor.get_subspace_projection(x)
        
        assert projection.shape == x.shape
        assert not torch.isnan(projection).any()
        assert not torch.isinf(projection).any()
    
    def test_get_orthogonal_projection(self, conceptor):
        """Test orthogonal projection."""
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, conceptor.hidden_dim)
        
        subspace_proj = conceptor.get_subspace_projection(x)
        orthogonal_proj = conceptor.get_orthogonal_projection(x)
        
        # Check that subspace + orthogonal = original
        reconstructed = subspace_proj + orthogonal_proj
        assert torch.allclose(x, reconstructed, atol=1e-5)
    
    def test_compute_aperture(self, conceptor):
        """Test aperture computation."""
        aperture = conceptor.compute_aperture()
        
        assert isinstance(aperture, float)
        assert aperture >= 0
        assert aperture <= conceptor.rank  # Aperture should be <= rank
    
    def test_regularize(self, conceptor):
        """Test regularization loss computation."""
        reg_loss = conceptor.regularize()
        
        assert isinstance(reg_loss, torch.Tensor)
        assert reg_loss.numel() == 1
        assert reg_loss.item() >= 0


class TestConceptorBank:
    """Test cases for ConceptorBank."""
    
    @pytest.fixture
    def conceptor_bank(self):
        """Create a conceptor bank for testing."""
        return ConceptorBank(
            num_layers=12,
            hidden_dim=128,
            rank=16,
            device="cpu"
        )
    
    def test_initialization(self, conceptor_bank):
        """Test conceptor bank initialization."""
        assert conceptor_bank.num_layers == 12
        assert conceptor_bank.hidden_dim == 128
        assert conceptor_bank.rank == 16
        assert len(conceptor_bank.conceptors) == 12
    
    def test_get_conceptor(self, conceptor_bank):
        """Test getting a specific conceptor."""
        conceptor = conceptor_bank.get_conceptor(0)
        
        assert isinstance(conceptor, Conceptor)
        assert conceptor.hidden_dim == 128
        assert conceptor.rank == 16
    
    def test_get_all_matrices(self, conceptor_bank):
        """Test getting all conceptor matrices."""
        matrices = conceptor_bank.get_all_matrices()
        
        assert len(matrices) == 12
        for matrix in matrices:
            assert matrix.shape == (128, 128)
            assert not torch.isnan(matrix).any()
            assert not torch.isinf(matrix).any()
    
    def test_contract_activations(self, conceptor_bank):
        """Test contracting activations for multiple layers."""
        batch_size, seq_len = 2, 10
        
        # Create activations for layers 0, 5, 10
        activations = {
            0: torch.randn(batch_size, seq_len, 128),
            5: torch.randn(batch_size, seq_len, 128),
            10: torch.randn(batch_size, seq_len, 128)
        }
        
        strengths = {0: 0.5, 5: 0.8, 10: 0.3}
        
        contracted = conceptor_bank.contract_activations(activations, strengths)
        
        assert len(contracted) == 3
        assert 0 in contracted
        assert 5 in contracted
        assert 10 in contracted
        
        for layer_idx, activation in contracted.items():
            assert activation.shape == (batch_size, seq_len, 128)
            assert not torch.isnan(activation).any()
            assert not torch.isinf(activation).any()
    
    def test_compute_total_regularization(self, conceptor_bank):
        """Test total regularization computation."""
        total_reg = conceptor_bank.compute_total_regularization()
        
        assert isinstance(total_reg, torch.Tensor)
        assert total_reg.numel() == 1
        assert total_reg.item() >= 0
    
    def test_get_apertures(self, conceptor_bank):
        """Test getting apertures for all conceptors."""
        apertures = conceptor_bank.get_apertures()
        
        assert len(apertures) == 12
        for aperture in apertures:
            assert isinstance(aperture, float)
            assert aperture >= 0
    
    def test_save_load_state(self, conceptor_bank, tmp_path):
        """Test saving and loading conceptor bank state."""
        # Save state
        save_path = tmp_path / "conceptor_bank.pt"
        conceptor_bank.save_state(str(save_path))
        assert save_path.exists()
        
        # Create new bank and load state
        new_bank = ConceptorBank(
            num_layers=12,
            hidden_dim=128,
            rank=16,
            device="cpu"
        )
        new_bank.load_state(str(save_path))
        
        # Check that states are equivalent
        original_matrices = conceptor_bank.get_all_matrices()
        loaded_matrices = new_bank.get_all_matrices()
        
        for orig, loaded in zip(original_matrices, loaded_matrices):
            assert torch.allclose(orig, loaded, atol=1e-6)


class TestConceptorCreation:
    """Test cases for conceptor creation utilities."""
    
    def test_create_conceptor_from_activations(self):
        """Test creating conceptor from activation data."""
        # Create sample activations
        num_samples, hidden_dim = 100, 128
        activations = torch.randn(num_samples, hidden_dim)
        
        # Create conceptor
        conceptor = create_conceptor_from_activations(
            activations,
            rank=16,
            regularization=1e-3
        )
        
        assert isinstance(conceptor, Conceptor)
        assert conceptor.hidden_dim == hidden_dim
        assert conceptor.rank == 16
        
        # Test that conceptor can process activations
        output = conceptor.forward(activations[:10])
        assert output.shape == (10, hidden_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_create_conceptor_from_activations_different_ranks(self):
        """Test creating conceptors with different ranks."""
        num_samples, hidden_dim = 50, 64
        activations = torch.randn(num_samples, hidden_dim)
        
        ranks = [4, 8, 16, 32]
        
        for rank in ranks:
            conceptor = create_conceptor_from_activations(
                activations,
                rank=rank,
                regularization=1e-3
            )
            
            assert conceptor.rank == rank
            assert conceptor.hidden_dim == hidden_dim
            
            # Test forward pass
            output = conceptor.forward(activations[:5])
            assert output.shape == (5, hidden_dim)
            assert not torch.isnan(output).any()
    
    def test_create_conceptor_edge_cases(self):
        """Test edge cases for conceptor creation."""
        # Test with very few samples
        activations = torch.randn(5, 32)
        conceptor = create_conceptor_from_activations(
            activations,
            rank=4,
            regularization=1e-3
        )
        assert conceptor.rank == 4
        assert conceptor.hidden_dim == 32
        
        # Test with rank equal to dimension
        activations = torch.randn(20, 16)
        conceptor = create_conceptor_from_activations(
            activations,
            rank=16,
            regularization=1e-3
        )
        assert conceptor.rank == 16
        assert conceptor.hidden_dim == 16
    
    def test_conceptor_matrix_properties(self):
        """Test mathematical properties of conceptor matrices."""
        # Create sample activations
        activations = torch.randn(100, 64)
        conceptor = create_conceptor_from_activations(
            activations,
            rank=16,
            regularization=1e-3
        )
        
        # Get conceptor matrix
        C = conceptor.get_matrix()
        
        # Check symmetry
        assert torch.allclose(C, C.t(), atol=1e-6)
        
        # Check positive semi-definiteness (eigenvalues >= 0)
        eigenvalues = torch.linalg.eigvals(C)
        assert torch.all(eigenvalues.real >= -1e-6)  # Allow small numerical errors
        
        # Check that matrix is bounded (eigenvalues <= 1)
        assert torch.all(eigenvalues.real <= 1 + 1e-6)


if __name__ == "__main__":
    pytest.main([__file__]) 