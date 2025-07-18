"""
Simple Instruction Encoder using BERT instead of T5.

This is an alternative encoder that avoids the SentencePiece dependency
by using BERT-based models for instruction encoding.
"""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class SimpleInstructionEncoder(nn.Module):
    """
    Encodes plain-language instructions into dense vectors using BERT.
    
    This is a simpler alternative to the T5-based encoder that avoids
    the SentencePiece dependency.
    """
    
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 128,
        pooling: str = "mean",
        device: Optional[str] = None
    ):
        """
        Initialize the simple instruction encoder.
        
        Args:
            model_name: BERT model name/path
            max_length: Maximum sequence length for tokenization
            pooling: Pooling strategy ("mean", "max", "cls")
            device: Device to run the model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name)
            
            # Freeze the model
            for param in self.model.parameters():
                param.requires_grad = False
                
            self.model.eval()
            self.model.to(self.device)
            
            # Get embedding dimension
            self.embedding_dim = self.model.config.hidden_size
            
            logger.info(f"Initialized SimpleInstructionEncoder with {model_name}")
            logger.info(f"Embedding dimension: {self.embedding_dim}")
            
        except Exception as e:
            logger.error(f"Failed to load BERT model {model_name}: {e}")
            logger.info("Falling back to dummy encoder")
            self._init_dummy_encoder()
    
    def _init_dummy_encoder(self):
        """Initialize a dummy encoder for testing purposes."""
        self.tokenizer = None
        self.model = None
        self.embedding_dim = 512
        self.is_dummy = True
        logger.warning("Using dummy encoder - instruction encoding will be random!")
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim
    
    def forward(self, instruction: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode instruction(s) into dense vectors.
        
        Args:
            instruction: Single instruction string or list of instructions
            
        Returns:
            Tensor of shape (batch_size, embedding_dim)
        """
        if hasattr(self, 'is_dummy') and self.is_dummy:
            # Return random embeddings for dummy encoder
            if isinstance(instruction, str):
                batch_size = 1
            else:
                batch_size = len(instruction)
            
            return torch.randn(batch_size, self.embedding_dim, device=self.device)
        
        # Convert single string to list
        if isinstance(instruction, str):
            instructions = [instruction]
        else:
            instructions = instruction
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                instructions,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                hidden_states = outputs.last_hidden_state
                
                # Apply pooling
                if self.pooling == "mean":
                    # Mean pooling over sequence length
                    attention_mask = inputs["attention_mask"].unsqueeze(-1)
                    masked_hidden = hidden_states * attention_mask
                    summed = masked_hidden.sum(dim=1)
                    lengths = attention_mask.sum(dim=1)
                    embeddings = summed / lengths
                elif self.pooling == "max":
                    # Max pooling over sequence length
                    embeddings = hidden_states.max(dim=1)[0]
                elif self.pooling == "cls":
                    # Use [CLS] token
                    embeddings = hidden_states[:, 0]
                else:
                    raise ValueError(f"Unknown pooling strategy: {self.pooling}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Error encoding instruction: {e}")
            # Fallback to random embedding
            batch_size = len(instructions)
            return torch.randn(batch_size, self.embedding_dim, device=self.device)
    
    def encode(self, instruction: Union[str, List[str]]) -> torch.Tensor:
        """Alias for forward method."""
        return self.forward(instruction) 