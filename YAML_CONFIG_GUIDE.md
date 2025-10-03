# YAML Configuration System Guide

This guide explains how to use the new YAML-based configuration system for Vision Mamba training, which replaces the previous `config.py` dataclass system.

## Overview

The new system uses YAML files to define experiment configurations, making it easier to:
- Create and manage different experiment setups
- Version control configurations
- Share reproducible experiment settings
- Quickly switch between different model architectures and datasets

## Configuration Files

All configuration files are stored in the `configs/` directory with `.yaml` extension.

### Available Configurations

- `cifar10_tiny.yaml` - Tiny Vision Mamba for CIFAR-10
- `cifar100_small.yaml` - Small Vision Mamba for CIFAR-100  
- `fashion_mnist.yaml` - Tiny Vision Mamba for Fashion-MNIST
- `imagenet_base.yaml` - Base Vision Mamba for ImageNet

### Configuration Structure

Each YAML file contains the following sections:

```yaml
experiment_name: "vision_mamba_cifar10_tiny"
seed: 42
device: "auto"

model:
  name: "vision_mamba_tiny"
  img_size: 32
  patch_size: 4
  # ... other model parameters

data:
  dataset_name: "CIFAR10"
  batch_size: 128
  # ... other data parameters

training:
  epochs: 200
  learning_rate: 0.001
  # ... other training parameters

logging:
  log_dir: "./logs"
  # ... other logging parameters
```

## Usage

### Training with YAML Configs

#### Basic Usage
```bash
# Train with default CIFAR-10 tiny configuration
python train_vision_mamba.py

# Train with specific configuration
python train_vision_mamba.py --config cifar100_small

# Train with custom configuration file
python train_vision_mamba.py --config-path /path/to/my_config.yaml
```

#### List Available Configurations
```bash
python train_vision_mamba.py --list-configs
```

### Resume Training
```bash
# Resume with same dataset
python resume_training.py --checkpoint ./checkpoints/vision_mamba_cifar10/best_model.pth

# Resume with different dataset
python resume_training.py --checkpoint ./checkpoints/vision_mamba_cifar10/best_model.pth --config cifar100_tiny

# List available configs for resume training
python resume_training.py --list-configs
```

### Evaluation
```bash
# Evaluate with auto-detected checkpoint
python enhanced_evaluate.py

# Evaluate with specific config
python enhanced_evaluate.py --config cifar10_tiny

# List available configs for evaluation
python enhanced_evaluate.py --list-configs
```

## Creating Custom Configurations

### Method 1: Copy and Modify Existing Config
```bash
cp configs/cifar10_tiny.yaml configs/my_experiment.yaml
# Edit my_experiment.yaml with your parameters
python train_vision_mamba.py --config my_experiment
```

### Method 2: Create from Template
```python
from yaml_config import create_config_template
create_config_template("configs/my_template.yaml")
```

### Method 3: Programmatically Create Config
```python
from yaml_config import Config
import yaml

# Create configuration dictionary
config_dict = {
    'experiment_name': 'my_custom_experiment',
    'model': {
        'name': 'vision_mamba_small',
        'embed_dim': 384,
        # ... other parameters
    },
    # ... other sections
}

# Save to YAML file
with open('configs/my_config.yaml', 'w') as f:
    yaml.dump(config_dict, f, default_flow_style=False, indent=2)
```

## Configuration Parameters

### Model Section
```yaml
model:
  name: "vision_mamba_tiny"  # vision_mamba_tiny, vision_mamba_small, vision_mamba_base
  img_size: 32              # Input image size
  patch_size: 4             # Patch size for tokenization
  in_channels: 3            # Number of input channels (3 for RGB, 1 for grayscale)
  embed_dim: 192            # Embedding dimension
  depth: 12                 # Number of transformer layers
  d_state: 16               # State space model dimension
  d_conv: 4                 # Convolution dimension
  num_heads: 3              # Number of attention heads
  mlp_ratio: 4.0            # MLP expansion ratio
  dropout: 0.1              # Dropout rate
  use_cls_token: true       # Whether to use classification token
```

### Data Section
```yaml
data:
  dataset_name: "CIFAR10"   # CIFAR10, CIFAR100, FashionMNIST, imagenet
  data_dir: "./data"        # Data directory
  num_classes: 10           # Number of classes (auto-set based on dataset)
  batch_size: 128           # Training batch size
  val_batch_size: 256       # Validation batch size
  num_workers: 4            # Number of data loading workers
  pin_memory: true          # Whether to pin memory for faster GPU transfer
  
  # Data augmentation
  use_augmentation: true    # Enable data augmentation
  random_crop: true         # Random crop augmentation
  random_flip: true         # Random horizontal flip
  normalize: true           # Normalize inputs
```

### Training Section
```yaml
training:
  epochs: 200               # Number of training epochs
  learning_rate: 0.001      # Initial learning rate
  weight_decay: 0.0001      # Weight decay for regularization
  optimizer: "adamw"        # Optimizer (adam, adamw, sgd)
  scheduler: "cosine"       # LR scheduler (cosine, step, plateau)
  
  # Scheduler parameters
  min_lr: 0.000001         # Minimum learning rate
  warmup_epochs: 10        # Number of warmup epochs
  step_size: 30            # Step size for step scheduler
  gamma: 0.1               # Decay factor
  
  # Training control
  patience: 15             # Early stopping patience
  min_delta: 0.0001        # Minimum improvement for early stopping
  max_grad_norm: 1.0       # Gradient clipping norm
  label_smoothing: 0.1     # Label smoothing factor
  use_amp: true            # Use automatic mixed precision
```

### Logging Section
```yaml
logging:
  log_dir: "./logs"             # Log directory
  checkpoint_dir: "./checkpoints" # Checkpoint directory
  tensorboard_dir: "./tensorboard" # TensorBoard log directory
  
  # Logging frequency
  log_interval: 10          # Log every N batches
  val_interval: 1           # Validate every N epochs
  save_interval: 5          # Save checkpoint every N epochs
  
  # Save options
  save_best_only: true      # Save only best model
  save_last: true           # Save final model
  monitor_metric: "val_acc" # Metric to monitor for best model
  
  # Visualization
  plot_training_curves: true    # Plot training curves
  plot_confusion_matrix: true   # Plot confusion matrix
  save_predictions: true        # Save model predictions
```

## Best Practices

### 1. Naming Convention
- Use descriptive names: `{dataset}_{model_size}.yaml`
- Include key parameters in filename: `cifar10_tiny_deep.yaml`

### 2. Parameter Tuning
- Start with provided configurations
- Modify one section at a time
- Keep notes in YAML comments

### 3. Experiment Tracking
- Use meaningful experiment names
- Include date/time in experiment names for uniqueness
- Save successful configurations for reuse

### 4. Version Control
- Commit configuration files with code
- Use separate configs for different experiments
- Document configuration changes in commit messages

## Troubleshooting

### Common Issues

1. **"Configuration file not found"**
   ```bash
   # Check available configs
   python train_vision_mamba.py --list-configs
   
   # Use correct config name (without .yaml)
   python train_vision_mamba.py --config cifar10_tiny
   ```

2. **YAML syntax errors**
   - Check indentation (use spaces, not tabs)
   - Ensure proper key-value format
   - Validate YAML syntax online

3. **Parameter validation errors**
   - Check parameter types (int, float, bool, string)
   - Ensure required parameters are present
   - Verify parameter ranges are valid

### Migration from config.py

If you have old code using `config.py`:

1. **Replace imports:**
   ```python
   # Old
   from config import get_config, ExperimentConfig
   
   # New
   from yaml_config import get_config, Config
   ```

2. **Update type annotations:**
   ```python
   # Old
   def my_function(config: ExperimentConfig):
   
   # New  
   def my_function(config: Config):
   ```

3. **Update config access:**
   ```python
   # Both old and new work the same way
   config.model.embed_dim
   config.data.batch_size
   config.training.learning_rate
   ```

## Examples

### Quick Start Examples

```bash
# Train tiny model on CIFAR-10
python train_vision_mamba.py --config cifar10_tiny

# Train small model on CIFAR-100 with 500 epochs
# (modify epochs in config file first)
python train_vision_mamba.py --config cifar100_small

# Resume training with different dataset
python resume_training.py \
  --checkpoint ./checkpoints/vision_mamba_cifar10/best_model.pth \
  --config fashion_mnist

# Evaluate model
python enhanced_evaluate.py --config cifar10_tiny
```

This new system provides much more flexibility and maintainability compared to the previous dataclass-based approach, while maintaining the same functionality and ease of use.