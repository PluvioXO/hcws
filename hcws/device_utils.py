"""
Device utilities for HCWS - Apple Silicon and CUDA support

Provides consistent device detection across the entire codebase.
"""

import torch
import os

def get_best_device():
    """
    Get the best available device for computation.
    
    Priority:
    1. MPS (Apple Silicon GPU) - for MacBook Pro M1/M2/M3
    2. CUDA (NVIDIA GPU) - for systems with NVIDIA GPUs
    3. CPU (fallback) - for systems without GPU acceleration
    
    Returns:
        str: Device string ("mps", "cuda", or "cpu")
    """
    if torch.backends.mps.is_available():
        # Enable MPS fallback for unsupported operations
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"

def get_device_info():
    """
    Get detailed information about the current device.
    
    Returns:
        dict: Device information including type, name, and capabilities
    """
    device = get_best_device()
    
    info = {
        "device": device,
        "device_name": "Unknown",
        "acceleration": False,
        "memory_gb": 0
    }
    
    if device == "mps":
        info.update({
            "device_name": "Apple Silicon GPU (Metal Performance Shaders)",
            "acceleration": True,
            "framework": "MPS",
            "fallback_enabled": os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1'
        })
    elif device == "cuda":
        if torch.cuda.is_available():
            info.update({
                "device_name": torch.cuda.get_device_name(0),
                "acceleration": True,
                "framework": "CUDA",
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9
            })
    else:
        info.update({
            "device_name": "CPU",
            "acceleration": False,
            "framework": "CPU"
        })
    
    return info

def print_device_info():
    """Print formatted device information."""
    info = get_device_info()
    
    print(f"Device: {info['device'].upper()}")
    print(f"Name: {info['device_name']}")
    
    if info['acceleration']:
        print(f"GPU Acceleration: ENABLED ({info['framework']})")
        if info['device'] == 'mps':
            print("Expected Performance: 3-5x faster than CPU")
            if info.get('fallback_enabled'):
                print("CPU Fallback: ENABLED for unsupported operations")
        elif info['device'] == 'cuda' and info['memory_gb'] > 0:
            print(f"GPU Memory: {info['memory_gb']:.1f} GB")
    else:
        print("GPU Acceleration: DISABLED")

# Legacy compatibility function
def get_device(device=None):
    """
    Legacy compatibility function for existing code.
    
    Args:
        device (str, optional): Override device selection
        
    Returns:
        str: Device string
    """
    if device is not None:
        return device
    return get_best_device() 