"""
Instruction Encoder using frozen T5 model to convert plain-language instructions
into dense vectors for conceptor generation.
"""

import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5EncoderModel
from typing import List, Union, Optional
import logging

logger = logging.getLogger(__name__)


class InstructionEncoder(nn.Module):
    """
    Encodes plain-language instructions into dense vectors using a frozen T5 encoder.
    
    The encoder takes instructions like "answer in Shakespearean English" and converts
    them into dense vectors z that can be used by the hyper-network to generate
    layer-specific conceptors.
    """
    
    def __init__(
        self,
        model_name: str = "t5-small",
        max_length: int = 128,
        pooling: str = "mean",
        device: Optional[str] = None,
        dtype: torch.dtype = torch.float16
    ):
        """
        Initialize the instruction encoder.
        
        Args:
            model_name: T5 model name/path
            max_length: Maximum sequence length for tokenization
            pooling: Pooling strategy ("mean", "max", "cls", "last")
            device: Device to run the model on
        """
        super().__init__()
        
        self.model_name = model_name
        self.max_length = max_length
        self.pooling = pooling
        self.dtype = dtype
        from .device_utils import get_device
        raw_device = get_device(device)
        
        # Convert to torch.device for consistent handling
        if isinstance(raw_device, str):
            self.device = torch.device(raw_device)
        else:
            self.device = raw_device
        
        # Load tokenizer and model with low precision
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        try:
            # Try loading with specified dtype for memory savings
            self.model = T5EncoderModel.from_pretrained(
                model_name, 
                torch_dtype=dtype
            )
        except Exception:
            # Fallback to default if dtype not supported
            self.model = T5EncoderModel.from_pretrained(model_name)
            self.model = self.model.to(dtype=dtype)
        
        # Freeze the model
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.eval()
        self.model.to(self.device)
        
        # Get embedding dimension
        self.embedding_dim = self.model.config.d_model
        
        logger.info(f"Initialized InstructionEncoder with {model_name}")
        logger.info(f"Embedding dimension: {self.embedding_dim}")
    
    def tokenize(self, instructions: Union[str, List[str]]) -> dict:
        """
        Tokenize instructions for T5 encoder.
        
        Args:
            instructions: Single instruction or list of instructions
            
        Returns:
            Dictionary with tokenized inputs
        """
        if isinstance(instructions, str):
            instructions = [instructions]
            
        # Add task prefix for T5
        prefixed_instructions = [f"encode: {inst}" for inst in instructions]
        
        tokens = self.tokenizer(
            prefixed_instructions,
            max_length=self.max_length,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        
        return {k: v.to(self.device) for k, v in tokens.items()}
    
    def pool_embeddings(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Pool hidden states to get single vector per instruction.
        
        Args:
            hidden_states: [batch_size, seq_len, hidden_dim]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Pooled embeddings [batch_size, hidden_dim]
        """
        if self.pooling == "mean":
            # Mean pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
            sum_mask = mask_expanded.sum(dim=1)
            sum_mask = torch.clamp(sum_mask, min=1e-9)
            return sum_embeddings / sum_mask
            
        elif self.pooling == "max":
            # Max pooling with attention mask
            mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
            hidden_states = hidden_states.masked_fill(mask_expanded == 0, -1e9)
            return torch.max(hidden_states, dim=1)[0]
            
        elif self.pooling == "cls":
            # Use first token (CLS-like)
            return hidden_states[:, 0]
            
        elif self.pooling == "last":
            # Use last valid token
            batch_size = hidden_states.size(0)
            last_indices = attention_mask.sum(dim=1) - 1
            return hidden_states[torch.arange(batch_size), last_indices]
            
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
    
    def forward(self, instructions: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode instructions into dense vectors.
        
        Args:
            instructions: Single instruction or list of instructions
            
        Returns:
            Dense vectors [batch_size, embedding_dim]
        """
        with torch.no_grad():
            # Tokenize
            tokens = self.tokenize(instructions)
            
            # Encode
            outputs = self.model(**tokens)
            hidden_states = outputs.last_hidden_state
            
            # Pool
            pooled = self.pool_embeddings(hidden_states, tokens["attention_mask"])
            
            return pooled
    
    def encode_batch(self, instructions: List[str], batch_size: int = 32) -> torch.Tensor:
        """
        Encode a large batch of instructions efficiently.
        
        Args:
            instructions: List of instructions
            batch_size: Batch size for processing
            
        Returns:
            Dense vectors [len(instructions), embedding_dim]
        """
        all_embeddings = []
        
        for i in range(0, len(instructions), batch_size):
            batch = instructions[i:i + batch_size]
            embeddings = self.forward(batch)
            all_embeddings.append(embeddings)
            
        return torch.cat(all_embeddings, dim=0)
    
    def get_embedding_dim(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dim 