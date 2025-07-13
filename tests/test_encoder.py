"""
Unit tests for the instruction encoder module.
"""

import pytest
import torch
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from hcws.encoder import InstructionEncoder


class TestInstructionEncoder:
    """Test cases for InstructionEncoder."""
    
    @pytest.fixture
    def encoder(self):
        """Create an instruction encoder for testing."""
        return InstructionEncoder("t5-small", device="cpu")
    
    def test_initialization(self, encoder):
        """Test encoder initialization."""
        assert encoder.model_name == "t5-small"
        assert encoder.max_length == 128
        assert encoder.pooling == "mean"
        assert encoder.embedding_dim > 0
        assert encoder.device == "cpu"
    
    def test_tokenize_single_instruction(self, encoder):
        """Test tokenizing a single instruction."""
        instruction = "answer in Shakespearean English"
        tokens = encoder.tokenize(instruction)
        
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 1
        assert tokens["attention_mask"].shape[0] == 1
    
    def test_tokenize_multiple_instructions(self, encoder):
        """Test tokenizing multiple instructions."""
        instructions = [
            "answer in Shakespearean English",
            "be positive and optimistic",
            "respond with factual accuracy"
        ]
        tokens = encoder.tokenize(instructions)
        
        assert "input_ids" in tokens
        assert "attention_mask" in tokens
        assert tokens["input_ids"].shape[0] == 3
        assert tokens["attention_mask"].shape[0] == 3
    
    def test_pool_embeddings_mean(self, encoder):
        """Test mean pooling of embeddings."""
        batch_size, seq_len, hidden_dim = 2, 10, encoder.embedding_dim
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        
        pooled = encoder.pool_embeddings(hidden_states, attention_mask)
        
        assert pooled.shape == (batch_size, hidden_dim)
        assert not torch.isnan(pooled).any()
        assert not torch.isinf(pooled).any()
    
    def test_pool_embeddings_max(self):
        """Test max pooling of embeddings."""
        encoder = InstructionEncoder("t5-small", pooling="max", device="cpu")
        batch_size, seq_len, hidden_dim = 2, 10, encoder.embedding_dim
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        
        pooled = encoder.pool_embeddings(hidden_states, attention_mask)
        
        assert pooled.shape == (batch_size, hidden_dim)
        assert not torch.isnan(pooled).any()
        assert not torch.isinf(pooled).any()
    
    def test_pool_embeddings_cls(self):
        """Test CLS pooling of embeddings."""
        encoder = InstructionEncoder("t5-small", pooling="cls", device="cpu")
        batch_size, seq_len, hidden_dim = 2, 10, encoder.embedding_dim
        
        hidden_states = torch.randn(batch_size, seq_len, hidden_dim)
        attention_mask = torch.ones(batch_size, seq_len)
        
        pooled = encoder.pool_embeddings(hidden_states, attention_mask)
        
        assert pooled.shape == (batch_size, hidden_dim)
        assert torch.allclose(pooled, hidden_states[:, 0])
    
    def test_forward_single_instruction(self, encoder):
        """Test forward pass with single instruction."""
        instruction = "answer in Shakespearean English"
        embedding = encoder.forward(instruction)
        
        assert embedding.shape == (1, encoder.embedding_dim)
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()
    
    def test_forward_multiple_instructions(self, encoder):
        """Test forward pass with multiple instructions."""
        instructions = [
            "answer in Shakespearean English",
            "be positive and optimistic",
            "respond with factual accuracy"
        ]
        embeddings = encoder.forward(instructions)
        
        assert embeddings.shape == (3, encoder.embedding_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_encode_batch(self, encoder):
        """Test batch encoding of instructions."""
        instructions = [
            "answer in Shakespearean English",
            "be positive and optimistic",
            "respond with factual accuracy",
            "use creative language",
            "be formal and professional"
        ]
        
        embeddings = encoder.encode_batch(instructions, batch_size=2)
        
        assert embeddings.shape == (5, encoder.embedding_dim)
        assert not torch.isnan(embeddings).any()
        assert not torch.isinf(embeddings).any()
    
    def test_different_instructions_different_embeddings(self, encoder):
        """Test that different instructions produce different embeddings."""
        instruction1 = "answer in Shakespearean English"
        instruction2 = "be positive and optimistic"
        
        embedding1 = encoder.forward(instruction1)
        embedding2 = encoder.forward(instruction2)
        
        # Embeddings should be different
        assert not torch.allclose(embedding1, embedding2, atol=1e-6)
    
    def test_same_instruction_same_embedding(self, encoder):
        """Test that same instruction produces same embedding."""
        instruction = "answer in Shakespearean English"
        
        embedding1 = encoder.forward(instruction)
        embedding2 = encoder.forward(instruction)
        
        # Embeddings should be identical
        assert torch.allclose(embedding1, embedding2)
    
    def test_get_embedding_dim(self, encoder):
        """Test getting embedding dimension."""
        dim = encoder.get_embedding_dim()
        assert dim == encoder.embedding_dim
        assert dim > 0
    
    def test_invalid_pooling_strategy(self):
        """Test invalid pooling strategy raises error."""
        with pytest.raises(ValueError, match="Unknown pooling strategy"):
            encoder = InstructionEncoder("t5-small", pooling="invalid", device="cpu")
            hidden_states = torch.randn(1, 10, 512)
            attention_mask = torch.ones(1, 10)
            encoder.pool_embeddings(hidden_states, attention_mask)
    
    def test_empty_instruction(self, encoder):
        """Test handling of empty instruction."""
        instruction = ""
        embedding = encoder.forward(instruction)
        
        assert embedding.shape == (1, encoder.embedding_dim)
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()
    
    def test_long_instruction(self, encoder):
        """Test handling of long instruction that exceeds max_length."""
        # Create a very long instruction
        long_instruction = "answer in Shakespearean English " * 20
        embedding = encoder.forward(long_instruction)
        
        assert embedding.shape == (1, encoder.embedding_dim)
        assert not torch.isnan(embedding).any()
        assert not torch.isinf(embedding).any()


if __name__ == "__main__":
    pytest.main([__file__]) 