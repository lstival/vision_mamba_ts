"""
Configuration Management Package

Contains configuration utilities:
- Config class for YAML-based configuration
- Legacy dataclass-based configuration
- Configuration loading and validation
"""

from .yaml_config import Config, load_config, get_config, list_available_configs, create_config_template
from .config import (
    ModelConfig,
    DataConfig,
    TrainingConfig,
    LoggingConfig,
    ExperimentConfig,
    get_cifar10_config,
    get_cifar100_config,
    get_fashion_mnist_config,
    get_imagenet_config,
    CONFIGS
)

__all__ = [
    'Config',
    'load_config',
    'get_config',
    'list_available_configs',
    'create_config_template',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'LoggingConfig',
    'ExperimentConfig',
    'get_cifar10_config',
    'get_cifar100_config',
    'get_fashion_mnist_config',
    'get_imagenet_config',
    'CONFIGS'
]