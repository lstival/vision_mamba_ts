"""
YAML-based configuration system for Vision Mamba training
Replaces the dataclass-based config.py with flexible YAML configurations
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


class Config:
    """Configuration class that loads from YAML files"""
    _global_config = {}  # Class variable to store accumulated config
    
    def __init__(self, config_dict: Dict[str, Any]):
        """Initialize config from dictionary"""
        # Merge with existing global config instead of overriding
        self._merge_config(config_dict, self._global_config)
        
        # Set all top-level keys as attributes from the merged config
        for key, value in self._global_config.items():
            if isinstance(value, dict):
                # Convert nested dictionaries to Config objects for dot notation access
                self.__dict__[key] = self._create_nested_config(value)
            else:
                self.__dict__[key] = value
        
        # Post-process configuration
        self._post_process()
    
    def _merge_config(self, source: Dict[str, Any], target: Dict[str, Any]):
        """Recursively merge source config into target config"""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                # If both are dicts, merge recursively
                self._merge_config(value, target[key])
            else:
                # Otherwise, update the value
                target[key] = value
    
    def _create_nested_config(self, config_dict: Dict[str, Any]):
        """Create a nested config object without affecting global state"""
        nested_config = Config.__new__(Config)  # Create instance without calling __init__
        for key, value in config_dict.items():
            if isinstance(value, dict):
                nested_config.__dict__[key] = self._create_nested_config(value)
            else:
                nested_config.__dict__[key] = value
        return nested_config
    
    def _post_process(self):
        """Post-process configuration to set up paths and validate settings"""

        # Ensure experiment_name exists
        if hasattr(self, 'data') and hasattr(self.data, 'dataset_name'):
            self.experiment_name = f"vision_mamba_{self.data.dataset_name.lower()}"
        else:
            raise AttributeError("The configuration is missing the 'data.dataset_name' field.")

        # Update number of classes based on dataset
        if hasattr(self, 'data'):
            if self.data.dataset_name == "CIFAR10":
                self.data.num_classes = 10
            elif self.data.dataset_name == "CIFAR100":
                self.data.num_classes = 100
            elif self.data.dataset_name == "FashionMNIST":
                self.data.num_classes = 10
            elif self.data.dataset_name == "imagenet":
                self.data.num_classes = 1000

        # Create directories if logging config exists
        if hasattr(self, 'logging'):
            # Create base directories
            os.makedirs(self.logging.log_dir, exist_ok=True)
            os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
            os.makedirs(self.logging.tensorboard_dir, exist_ok=True)

            if hasattr(self, 'data'):
                os.makedirs(self.data.data_dir, exist_ok=True)

            # Update experiment-specific paths
            exp_log_path = os.path.join(self.logging.log_dir, self.experiment_name)
            exp_checkpoint_path = os.path.join(self.logging.checkpoint_dir, self.experiment_name)
            exp_tensorboard_path = os.path.join(self.logging.tensorboard_dir, self.experiment_name)

            self.logging.log_dir = exp_log_path
            self.logging.checkpoint_dir = exp_checkpoint_path
            self.logging.tensorboard_dir = exp_tensorboard_path

            # Create experiment-specific directories
            os.makedirs(self.logging.log_dir, exist_ok=True)
            os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
            os.makedirs(self.logging.tensorboard_dir, exist_ok=True)
    
    def to_dict(self):
        """Convert config back to dictionary"""
        result = {}
        for key, value in self.__dict__.items():
            if isinstance(value, Config):
                result[key] = value.to_dict()
            else:
                result[key] = value
        return result
    
    def save(self, path: str):
        """Save configuration to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
    
    def __repr__(self):
        return f"Config({self.experiment_name})"


def load_config(config_path: str) -> Config:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Config object with loaded configuration
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return Config(config_dict)


def get_config(config_name: str, config_dir: str = "configs") -> Config:
    """
    Get configuration by name from the configs directory
    
    Args:
        config_name: Name of configuration (without .yaml extension)
        config_dir: Directory containing configuration files
        
    Returns:
        Config object
    """
    # Handle both with and without .yaml extension
    if not config_name.endswith('.yaml'):
        config_name += '.yaml'
    
    config_path = os.path.join(config_dir, config_name)
    return load_config(config_path)


def list_available_configs(config_dir: str = "configs") -> list:
    """
    List all available configuration files
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        List of available configuration names (without .yaml extension)
    """
    if not os.path.exists(config_dir):
        return []
    
    configs = []
    for file in os.listdir(config_dir):
        if file.endswith('.yaml'):
            configs.append(file[:-5])  # Remove .yaml extension
    
    return sorted(configs)


def create_config_template(output_path: str):
    """
    Create a template configuration file
    
    Args:
        output_path: Path where to save the template
    """
    template = {
        'experiment_name': 'vision_mamba_experiment',
        'seed': 42,
        'device': 'auto',
        'model': {
            'name': 'vision_mamba_tiny',
            'img_size': 32,
            'patch_size': 4,
            'in_channels': 3,
            'embed_dim': 192,
            'depth': 12,
            'd_state': 16,
            'd_conv': 4,
            'num_heads': 3,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'use_cls_token': True
        },
        'data': {
            'dataset_name': 'CIFAR10',
            'data_dir': './data',
            'num_classes': 10,
            'batch_size': 128,
            'val_batch_size': 256,
            'num_workers': 4,
            'pin_memory': True,
            'use_augmentation': True,
            'random_crop': True,
            'random_flip': True,
            'normalize': True,
            'train_ratio': 0.8,
            'val_ratio': 0.2
        },
        'training': {
            'epochs': 100,
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'optimizer': 'adamw',
            'scheduler': 'cosine',
            'min_lr': 0.000001,
            'warmup_epochs': 10,
            'step_size': 30,
            'gamma': 0.1,
            'patience': 15,
            'min_delta': 0.0001,
            'max_grad_norm': 1.0,
            'label_smoothing': 0.1,
            'use_amp': True
        },
        'logging': {
            'log_dir': './logs',
            'checkpoint_dir': './checkpoints',
            'tensorboard_dir': './tensorboard',
            'log_interval': 10,
            'val_interval': 1,
            'save_interval': 5,
            'save_best_only': True,
            'save_last': True,
            'monitor_metric': 'val_acc',
            'plot_training_curves': True,
            'plot_confusion_matrix': True,
            'save_predictions': True
        }
    }
    
    with open(output_path, 'w') as f:
        yaml.dump(template, f, default_flow_style=False, indent=2)


if __name__ == "__main__":
    # Example usage
    print("Available configurations:")
    configs = list_available_configs()
    for config in configs:
        print(f"  - {config}")
    
    if configs:
        # Load first available config as example
        config = get_config(configs[1])
        print(f"\nLoaded config: {config}")
        print(f"Experiment: {config.experiment_name}")
        print(f"Dataset: {config.data.dataset_name}")
        print(f"Model: {config.model.name}")
        print(f"Batch size: {config.data.batch_size}")
        print(f"Learning rate: {config.training.learning_rate}")
        print(f"Epochs: {config.training.epochs}")
        print(f"Log directory: {config.logging.log_dir}")
    else:
        print("No configuration files found in configs/ directory")
        print("Creating template configuration...")
        create_config_template("configs/template.yaml")
        print("Template created at configs/template.yaml")