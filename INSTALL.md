# Installation Guide

## Installation Options

### Option 1: Development Installation (Recommended)

Install the package in development mode so you can edit the code:

```bash
# Clone or navigate to the project directory
cd vision_mamba_ts

# Install in development mode
pip install -e .
```

This allows you to:
- Import the package from anywhere: `from vision_mamba import VisionMamba`
- Edit the source code and have changes reflected immediately
- Use console scripts: `vision-mamba-train`, `vision-mamba-evaluate`

### Option 2: Direct Use

If you prefer not to install the package, you can run scripts directly:

```bash
# Make sure you're in the project root directory
cd vision_mamba_ts

# Run scripts directly
python scripts/train_vision_mamba.py --config cifar10_tiny
python demos/demo_training.py
```

### Option 3: Package Installation

Build and install as a standard package:

```bash
# Build the package
python setup.py sdist bdist_wheel

# Install from built package
pip install dist/vision_mamba-0.1.0-py3-none-any.whl
```

## Testing Installation

Run the test script to verify everything works:

```bash
python test_package.py
```

## Usage After Installation

### Development Installation Usage

```python
# Import the package
from vision_mamba import VisionMamba, Config, get_config

# Load configuration
config = get_config('cifar10_tiny')

# Create model
from vision_mamba.models import create_vision_mamba_tiny
model = create_vision_mamba_tiny(num_classes=10)

# Use utilities
from vision_mamba.utils import set_seed, get_device
set_seed(42)
device = get_device('auto')
```

### Console Scripts (after development installation)

```bash
# Train a model
vision-mamba-train --config cifar10_tiny

# Train MAE
vision-mamba-mae-train --config cifar10_tiny_mae

# Evaluate model
vision-mamba-evaluate --experiment vision_mamba_cifar10

# Evaluate MAE
vision-mamba-mae-evaluate --checkpoint path/to/model.pth
```

## Troubleshooting

### Import Errors
If you get import errors, make sure:
1. You're in the project root directory
2. The `vision_mamba` folder exists with `__init__.py` files
3. Required dependencies are installed: `pip install -r requirements.txt`

### Path Issues
If scripts can't find modules, try:
1. Installing in development mode: `pip install -e .`
2. Adding project root to Python path manually in scripts

### Dependencies
Make sure all required packages are installed:
```bash
pip install -r requirements.txt
```

For additional features:
```bash
# For advanced MAE metrics
pip install scikit-image

# For experiment tracking
pip install wandb

# For development
pip install -e .[dev]
```