"""
Hyper-Conceptor Weighted Steering (HCWS)

A lightweight method for steering large language models using conceptor-based 
activation modification during inference.
"""

from .model import HCWSModel
from .encoder import InstructionEncoder
from .hyper_network import HyperNetwork
from .conceptors import Conceptor, ConceptorBank
from .controller import SteeringController
from .actadd import ActAddModel

__version__ = "0.1.0"
__all__ = [
    "HCWSModel",
    "InstructionEncoder", 
    "HyperNetwork",
    "Conceptor",
    "ConceptorBank",
    "SteeringController",
    "ActAddModel"
] 