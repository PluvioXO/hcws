"""
Device utilities for HCWS - Google TPU, Apple Silicon and CUDA support

Provides consistent device detection across the entire codebase.
"""

import torch
import os
import logging

logger = logging.getLogger(__name__)


def _check_torch_xla():
    """Check if torch_xla is available for TPU support."""
    try:
        import torch_xla
        import torch_xla.core.xla_model as xm
        return True
    except ImportError:
        return False


def get_best_device():
    """
    Get the best available device for computation.
    
    Priority:
    1. TPU (Google Cloud TPU) - for Cloud TPU VMs and Colab TPU
    2. MPS (Apple Silicon GPU) - for MacBook Pro M1/M2/M3
    3. CUDA (NVIDIA GPU) - for systems with NVIDIA GPUs
    4. CPU (fallback) - for systems without GPU acceleration
    
    Returns:
        str: Device string ("tpu", "mps", "cuda", or "cpu")
    """
    # Check for TPU first
    if _check_torch_xla():
        try:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            if device.type == 'xla':
                return "tpu"
        except Exception as e:
            logger.debug(f"TPU check failed: {e}")
    
    # Check for MPS (Apple Silicon)
    if torch.backends.mps.is_available():
        # Enable MPS fallback for unsupported operations
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
        return "mps"
    
    # Check for CUDA
    elif torch.cuda.is_available():
        return "cuda"
    
    # Fallback to CPU
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
    
    if device == "tpu":
        try:
            import torch_xla.core.xla_model as xm
            tpu_device = xm.xla_device()
            
            info.update({
                "device_name": f"Google Cloud TPU ({tpu_device})",
                "acceleration": True,
                "framework": "XLA",
                "device_type": tpu_device.type,
                "ordinal": xm.get_ordinal(),
                "world_size": xm.xrt_world_size(),
                "local_rank": xm.get_local_ordinal(),
                "master_ip": os.environ.get('XRT_TPU_CONFIG', 'N/A')
            })
            
            # Try to get TPU topology information
            try:
                from torch_xla._internal import tpu
                topology = tpu.get_tpu_env()
                if topology:
                    info.update({
                        "tpu_topology": topology.get('TPU_TOPOLOGY', 'Unknown'),
                        "tpu_chips": topology.get('TPU_NUM_DEVICES', 'Unknown')
                    })
            except Exception:
                logger.debug("Could not get TPU topology information")
                
        except Exception as e:
            logger.warning(f"Error getting TPU info: {e}")
            info.update({
                "device_name": "Google Cloud TPU (details unavailable)",
                "acceleration": True,
                "framework": "XLA"
            })
            
    elif device == "mps":
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
                "memory_gb": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "device_count": torch.cuda.device_count(),
                "current_device": torch.cuda.current_device()
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
        print(f"GPU/TPU Acceleration: ENABLED ({info['framework']})")
        
        if info['device'] == 'tpu':
            print("Expected Performance: 10-100x faster than CPU for large models")
            if 'world_size' in info:
                print(f"TPU Cores: {info['world_size']}")
                print(f"Current Ordinal: {info['ordinal']}")
                print(f"Local Rank: {info['local_rank']}")
            if 'tpu_topology' in info:
                print(f"TPU Topology: {info['tpu_topology']}")
                print(f"TPU Chips: {info['tpu_chips']}")
                
        elif info['device'] == 'mps':
            print("Expected Performance: 3-5x faster than CPU")
            if info.get('fallback_enabled'):
                print("CPU Fallback: ENABLED for unsupported operations")
                
        elif info['device'] == 'cuda' and info['memory_gb'] > 0:
            print(f"GPU Memory: {info['memory_gb']:.1f} GB")
            if 'device_count' in info:
                print(f"GPU Count: {info['device_count']}")
    else:
        print("GPU/TPU Acceleration: DISABLED")


def get_tpu_device():
    """
    Get the TPU device using torch_xla.
    
    Returns:
        torch.device: TPU device object
        
    Raises:
        RuntimeError: If TPU is not available
    """
    if not _check_torch_xla():
        raise RuntimeError("torch_xla is not installed. Install with: pip install torch_xla")
    
    try:
        import torch_xla.core.xla_model as xm
        return xm.xla_device()
    except Exception as e:
        raise RuntimeError(f"Failed to get TPU device: {e}")


def initialize_tpu():
    """
    Initialize TPU environment and synchronize devices.
    
    This function should be called once at the beginning of training
    when using TPUs for multi-core synchronization.
    """
    if not _check_torch_xla():
        logger.warning("torch_xla not available, skipping TPU initialization")
        return
    
    try:
        import torch_xla.core.xla_model as xm
        
        # Mark step to initialize XLA compilation
        xm.mark_step()
        
        # Wait for all devices to be ready
        xm.wait_device_ops()
        
        logger.info(f"TPU initialized successfully with {xm.xrt_world_size()} cores")
        
    except Exception as e:
        logger.error(f"Failed to initialize TPU: {e}")
        raise


def is_tpu_available():
    """Check if TPU is available."""
    return get_best_device() == "tpu"


def get_tpu_cores():
    """
    Get the number of available TPU cores.
    
    Returns:
        int: Number of TPU cores, or 0 if TPU not available
    """
    if not is_tpu_available():
        return 0
    
    try:
        import torch_xla.core.xla_model as xm
        return xm.xrt_world_size()
    except Exception:
        return 0


# Legacy compatibility function
def get_device(device=None):
    """
    Legacy compatibility function for existing code.
    
    Args:
        device (str, optional): Override device selection
        
    Returns:
        str or torch.device: Device string or TPU device object
    """
    if device is not None:
        if device == "tpu" and is_tpu_available():
            return get_tpu_device()
        return device
    
    best_device = get_best_device()
    if best_device == "tpu":
        return get_tpu_device()
    return best_device 