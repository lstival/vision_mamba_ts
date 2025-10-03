"""
Configuration file for Vision Mamba training
Contains all hyperparameters, paths, and training settings
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model architecture configuration"""
    model_name: str = "vision_mamba_tiny"
    img_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 192
    depth: int = 12
    d_state: int = 16
    d_conv: int = 4
    num_heads: int = 3
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    use_cls_token: bool = True


@dataclass
class DataConfig:
    """Dataset configuration"""
    dataset_name: str = "CIFAR10"  # Options: CIFAR10, CIFAR100, imagenet, FashionMNIST
    data_dir: str = "./data"
    num_classes: int = 10
    batch_size: int = 128
    val_batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    
    # Data augmentation
    use_augmentation: bool = True
    random_crop: bool = True
    random_flip: bool = True
    normalize: bool = True
    
    # Dataset split
    train_ratio: float = 0.8
    val_ratio: float = 0.2


@dataclass
class TrainingConfig:
    """Training configuration"""
    epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"  # Options: adam, adamw, sgd
    scheduler: str = "cosine"  # Options: cosine, step, plateau
    
    # Learning rate scheduler parameters
    min_lr: float = 1e-6
    warmup_epochs: int = 10
    step_size: int = 30
    gamma: float = 0.1
    
    # Early stopping
    patience: int = 15
    min_delta: float = 1e-4
    
    # Gradient clipping
    max_grad_norm: float = 1.0
    
    # Loss function
    label_smoothing: float = 0.1
    
    # Mixed precision training
    use_amp: bool = True


@dataclass
class LoggingConfig:
    """Logging and checkpointing configuration"""
    log_dir: str = "./logs"
    checkpoint_dir: str = "./checkpoints"
    tensorboard_dir: str = "./tensorboard"
    
    # Logging frequency
    log_interval: int = 10  # Log every N batches
    val_interval: int = 1   # Validate every N epochs
    save_interval: int = 5  # Save checkpoint every N epochs
    
    # What to save
    save_best_only: bool = True
    save_last: bool = True
    monitor_metric: str = "val_acc"  # Metric to monitor for best model
    
    # Visualization
    plot_training_curves: bool = True
    plot_confusion_matrix: bool = True
    save_predictions: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    seed: int = 42
    device: str = "auto"  # auto, cpu, cuda
    
    # Configurations
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    def __post_init__(self):
        """Post-initialization to set up paths and validate config"""
        self.experiment_name: str = f"vision_mamba_{self.data.dataset_name}"

        # Update model num_classes based on dataset
        if self.data.dataset_name == "CIFAR10":
            self.data.num_classes = 10
        elif self.data.dataset_name == "CIFAR100":
            self.data.num_classes = 100
        elif self.data.dataset_name == "FashionMNIST":
            self.data.num_classes = 10
        elif self.data.dataset_name == "imagenet":
            self.data.num_classes = 1000
            
        # Create directories
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logging.tensorboard_dir, exist_ok=True)
        os.makedirs(self.data.data_dir, exist_ok=True)
        
        # Update experiment-specific paths
        exp_path = os.path.join(self.logging.log_dir, self.experiment_name)
        self.logging.log_dir = exp_path
        
        exp_checkpoint_path = os.path.join(self.logging.checkpoint_dir, self.experiment_name)
        self.logging.checkpoint_dir = exp_checkpoint_path
        
        exp_tensorboard_path = os.path.join(self.logging.tensorboard_dir, self.experiment_name)
        self.logging.tensorboard_dir = exp_tensorboard_path
        
        # Create experiment-specific directories
        os.makedirs(self.logging.log_dir, exist_ok=True)
        os.makedirs(self.logging.checkpoint_dir, exist_ok=True)
        os.makedirs(self.logging.tensorboard_dir, exist_ok=True)


# Predefined configurations for different experiments
def get_cifar10_config():
    """CIFAR-10 experiment configuration"""
    config = ExperimentConfig()
    config.experiment_name = "vision_mamba_cifar10"
    config.data.dataset_name = "CIFAR10"
    config.data.num_classes = 10
    config.data.batch_size = 128
    config.training.epochs = 200
    config.training.learning_rate = 1e-3
    return config


def get_cifar100_config():
    """CIFAR-100 experiment configuration"""
    config = ExperimentConfig()
    config.experiment_name = "vision_mamba_cifar100"
    config.data.dataset_name = "CIFAR100"
    config.data.num_classes = 100
    config.data.batch_size = 128
    config.training.epochs = 300
    config.training.learning_rate = 1e-3
    config.training.label_smoothing = 0.2
    return config


def get_fashion_mnist_config():
    """Fashion-MNIST experiment configuration"""
    config = ExperimentConfig()
    config.experiment_name = "vision_mamba_fashion_mnist"
    config.data.dataset_name = "FashionMNIST"
    config.data.num_classes = 10
    config.data.batch_size = 256
    config.model.in_channels = 1  # Grayscale images
    config.training.epochs = 100
    config.training.learning_rate = 1e-3
    return config


def get_imagenet_config():
    """ImageNet experiment configuration (requires more resources)"""
    config = ExperimentConfig()
    config.experiment_name = "vision_mamba_imagenet"
    config.data.dataset_name = "imagenet"
    config.data.num_classes = 1000
    config.data.batch_size = 64  # Smaller batch size for memory
    config.model.model_name = "vision_mamba_base"  # Larger model
    config.model.embed_dim = 768
    config.model.num_heads = 12
    config.training.epochs = 300
    config.training.learning_rate = 1e-4
    config.training.warmup_epochs = 20
    return config


# Available configurations
CONFIGS = {
    "cifar10": get_cifar10_config,
    "cifar100": get_cifar100_config,
    "fashion_mnist": get_fashion_mnist_config,
    "imagenet": get_imagenet_config,
}


def get_config(config_name: str = "cifar10") -> ExperimentConfig:
    """Get a predefined configuration by name"""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown configuration: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name]()


if __name__ == "__main__":
    # Example usage
    # config = get_cifar10_config()
    config = get_imagenet_config()
    print(f"Experiment: {config.experiment_name}")
    print(f"Dataset: {config.data.dataset_name}")
    print(f"Model: {config.model.model_name}")
    print(f"Batch size: {config.data.batch_size}")
    print(f"Learning rate: {config.training.learning_rate}")
    print(f"Epochs: {config.training.epochs}")
    print(f"Log directory: {config.logging.log_dir}")