# Vision Mamba Training Framework

A comprehensive training framework for Vision Mamba models on image classification tasks using PyTorch and torchvision datasets.

## Features

- **Vision Mamba Architecture**: Hybrid model combining State Space Models (SSM) with attention mechanisms
- **Multiple Datasets**: Support for CIFAR-10, CIFAR-100, Fashion-MNIST, and more
- **Comprehensive Training**: Full training pipeline with validation, early stopping, and learning rate scheduling
- **Detailed Evaluation**: Confusion matrix, classification reports, and per-class performance metrics
- **Experiment Tracking**: TensorBoard integration and detailed logging
- **Configurable**: Easy-to-modify configuration system for different experiments

## Files Structure

```
├── vision_mamba.py           # Vision Mamba model implementation
├── config.py                 # Configuration system for experiments
├── train_vision_mamba.py     # Main training script
├── evaluate_model.py         # Model evaluation script
├── demo_training.py          # Quick demo/testing script
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Make sure you have a compatible PyTorch installation with CUDA support (if using GPU):
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

## Quick Start

### 1. Test the Setup

First, run the demo to make sure everything works:
```bash
python demo_training.py
```

### 2. Train a Model

Train on CIFAR-10 (recommended for quick testing):
```bash
python train_vision_mamba.py --config cifar10
```

Train on CIFAR-100:
```bash
python train_vision_mamba.py --config cifar100
```

Train on Fashion-MNIST:
```bash
python train_vision_mamba.py --config fashion_mnist
```

### 3. Monitor Training

Open TensorBoard to monitor training progress:
```bash
tensorboard --logdir ./tensorboard
```

### 4. Evaluate the Model

After training, evaluate the best model:
```bash
python evaluate_model.py --checkpoint ./checkpoints/vision_mamba_cifar10/best_model.pth --output_dir ./evaluation_results
```

## Configuration

The `config.py` file contains all experiment configurations. You can modify the following aspects:

### Model Configuration
- Model size (tiny, small, base, large)
- Embedding dimensions
- Number of layers and attention heads
- Dropout rates

### Training Configuration
- Learning rate and optimization settings
- Batch size and number of epochs
- Data augmentation parameters
- Loss function settings

### Logging Configuration
- Output directories
- Checkpoint saving frequency
- Visualization options

## Model Architecture

The Vision Mamba model combines:

1. **Patch Embedding**: Converts 2D images into 1D token sequences
2. **SSM Layers**: State Space Models for efficient sequence modeling
3. **Attention Layers**: Multi-head attention for global context
4. **Hybrid Blocks**: Learnable combination of SSM and attention
5. **Classification Head**: Final layer for class prediction

## Training Features

- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Cosine annealing, step, or plateau schedulers
- **Gradient Clipping**: Prevents gradient explosion
- **Data Augmentation**: Random crops, flips, and normalization
- **Checkpointing**: Automatic saving of best and intermediate models

## Evaluation Features

- **Accuracy Metrics**: Overall and per-class accuracy
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Precision, recall, and F1-scores
- **Sample Visualizations**: Visual inspection of model predictions

## Output Structure

After training, the following structure is created:

```
├── logs/
│   └── vision_mamba_cifar10/
│       ├── training_log.json
│       ├── confusion_matrix.png
│       └── training_curves.png
├── checkpoints/
│   └── vision_mamba_cifar10/
│       ├── best_model.pth
│       ├── final_model.pth
│       └── checkpoint_epoch_*.pth
└── tensorboard/
    └── vision_mamba_cifar10/
        └── [tensorboard logs]
```

## Example Results

The framework provides comprehensive evaluation including:

1. **Training Curves**: Loss and accuracy plots over epochs
2. **Confusion Matrix**: Both raw counts and normalized percentages
3. **Per-Class Performance**: Precision, recall, and F1-score for each class
4. **Sample Predictions**: Visual examples of model predictions

## Customization

### Adding New Datasets

To add support for new datasets, modify the `create_data_loaders` function in `train_vision_mamba.py`:

```python
elif config.data.dataset_name == "YOUR_DATASET":
    train_dataset = torchvision.datasets.YourDataset(...)
    class_names = ["class1", "class2", ...]
```

### Modifying Model Architecture

The Vision Mamba architecture can be customized in `vision_mamba.py`:

- Change embedding dimensions
- Modify SSM parameters (d_state, d_conv)
- Adjust attention heads and MLP ratios
- Add new model variants

### Custom Training Loops

The training loop in `train_vision_mamba.py` can be extended with:

- Custom loss functions
- Additional metrics
- Different optimization strategies
- Custom callbacks

## Performance Tips

1. **GPU Memory**: Reduce batch size if you encounter out-of-memory errors
2. **Training Speed**: Use mixed precision training and multiple workers
3. **Model Size**: Start with smaller models (tiny/small) for quick experimentation
4. **Data Loading**: Increase `num_workers` for faster data loading

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or model size
2. **Slow training**: Enable mixed precision and increase num_workers
3. **Poor convergence**: Try different learning rates or schedulers
4. **Dataset download issues**: Check internet connection and storage space

### Debug Mode

Run the demo script to test your setup:
```bash
python demo_training.py
```

## Contributing

Feel free to extend this framework with:
- Additional datasets
- New model variants
- Advanced training techniques
- Better evaluation metrics

## License

This project is provided as-is for educational and research purposes.