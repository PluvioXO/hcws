"""
Hyper-Conceptor Weighted Steering (HCWS)

A lightweight method for steering large language models using conceptor-based 
activation modification during inference.
"""

from .model import HCWSModel
from .encoder import InstructionEncoder
from .simple_encoder import SimpleInstructionEncoder
from .hyper_network import HyperNetwork
from .conceptors import Conceptor, ConceptorBank
from .controller import SteeringController
from .actadd import ActAddModel
from .model_registry import (
    ModelConfig, 
    get_model_config, 
    list_available_models, 
    print_available_models,
    detect_model_config
)

__version__ = "0.1.0"
__all__ = [
    "HCWSModel",
    "InstructionEncoder",
    "SimpleInstructionEncoder",
    "HyperNetwork",
    "Conceptor",
    "ConceptorBank",
    "SteeringController",
    "ActAddModel",
    "ModelConfig",
    "get_model_config",
    "list_available_models", 
    "print_available_models",
    "detect_model_config"
] 