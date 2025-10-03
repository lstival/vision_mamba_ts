"""
Vision Mamba: A hybrid architecture combining State Space Models and Attention
for image processing and time series analysis.

This package provides:
- Vision Mamba models for image classification and masked autoencoding
- Training utilities and scripts
- Evaluation tools and metrics
- Configuration management
- Fast attention implementations
"""

__version__ = "0.1.0"
__author__ = "Vision Mamba Team"

# Import main components for easy access
from .models import (
    VisionMamba,
    VisionMambaMAE,
    create_vision_mamba_tiny,
    create_vision_mamba_small,
    create_vision_mamba_base,
    create_vision_mamba_large,
    create_vision_mamba_mae
)

from .config import Config, load_config, get_config
from .utils import set_seed, get_device

__all__ = [
    'VisionMamba',
    'VisionMambaMAE',
    'create_vision_mamba_tiny',
    'create_vision_mamba_small',
    'create_vision_mamba_base',
    'create_vision_mamba_large',
    'create_vision_mamba_mae',
    'Config',
    'load_config',
    'get_config',
    'set_seed',
    'get_device'
]