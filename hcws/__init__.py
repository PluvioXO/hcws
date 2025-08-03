"""
HCWS - Hyper-Conceptor Weighted Steering

A library for steering language model behavior using hyper-networks and conceptors.
"""

from .model import HCWSModel
from .actadd import ActAddModel
from .device_utils import get_best_device, print_device_info, get_device
from .model_registry import get_model_config, print_available_models
from .training import model_train, train_hcws_model, train_hcws_model_with_instruction_check, evaluate_hcws_model, save_training_data_template

__version__ = "0.1.0"

# CLI is available but not exported by default
# Use: python -m hcws train --model gpt2

__all__ = [
    "HCWSModel",
    "ActAddModel", 
    "get_best_device",
    "print_device_info",
    "get_device",
    "get_model_config",
    "print_available_models",
    "model_train",
    "train_hcws_model", 
    "train_hcws_model_with_instruction_check",
    "evaluate_hcws_model",
    "save_training_data_template"
] 