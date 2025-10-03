# Vision Mamba Training Framework

A comprehensive training framework for Vision Mamba models supporting both **image classification** and **masked autoencoder (MAE)** tasks using PyTorch and torchvision datasets.

## Features

- **Vision Mamba Architecture**: Hybrid model combining State Space Models (SSM) with attention mechanisms
- **Fast Attention Support**: PyTorch 2.0+, Flash Attention, and memory-efficient implementations
- **Dual Training Modes**: 
  - **Classification**: Traditional supervised learning for image classification
  - **MAE (Masked Autoencoder)**: Self-supervised pre-training with patch reconstruction
- **Multiple Datasets**: Support for CIFAR-10, CIFAR-100, Fashion-MNIST, and more
- **Comprehensive Training**: Full training pipeline with validation, early stopping, and learning rate scheduling
- **Detailed Evaluation**: Confusion matrix, classification reports, and reconstruction visualizations
- **Experiment Tracking**: TensorBoard integration and detailed logging
- **Configurable**: Easy-to-modify configuration system for different experiments

## Files Structure

```
├── vision_mamba.py           # Vision Mamba model implementation (encoder)
├── vision_mamba_mae.py       # Vision Mamba MAE model implementation
├── config.py                 # Configuration system for experiments
├── yaml_config.py            # YAML-based configuration system
├── train_vision_mamba.py     # Main training script for classification
├── train_vision_mamba_mae.py # MAE training script for self-supervised learning
├── evaluate_model.py         # Model evaluation script
├── demo_training.py          # Quick demo/testing script for classification
├── demo_mae.py               # Quick demo/testing script for MAE
├── evaluate_mae.py           # Comprehensive MAE evaluation script
├── demo_mae_evaluation.py    # Demo script for MAE evaluation
├── fast_attention.py         # Fast attention implementations
├── fast_vision_mamba.py      # Vision Mamba with fast attention
├── demo_fast_attention.py    # Demo script for fast attention
├── FAST_ATTENTION_GUIDE.md   # Comprehensive fast attention guide
├── requirements.txt          # Python dependencies
├── configs/                  # Configuration files
│   ├── cifar10_tiny.yaml     # CIFAR-10 classification config
│   ├── cifar10_tiny_mae.yaml # CIFAR-10 MAE config
│   └── ...                   # Other dataset configs
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

**For Classification:**
```bash
python demo_training.py
```

**For MAE (Masked Autoencoder):**
```bash
python demo_mae.py
```

### 2. Train Models

#### Classification Training

Train on CIFAR-10 (recommended for quick testing):
```bash
python train_vision_mamba.py --config cifar10_tiny
```

Train on CIFAR-100:
```bash
python train_vision_mamba.py --config cifar100_tiny
```

Train on Fashion-MNIST:
```bash
python train_vision_mamba.py --config fashion_mnist
```

#### MAE (Self-Supervised) Training

Train MAE on CIFAR-10 for self-supervised pre-training:
```bash
python train_vision_mamba_mae.py --config cifar10_tiny_mae
```

Train MAE on CIFAR-100:
```bash
python train_vision_mamba_mae.py --config cifar100_tiny_mae
```

**Note**: MAE training learns to reconstruct masked image patches, which can be useful for:
- Pre-training before fine-tuning on classification tasks
- Learning robust visual representations
- Data augmentation and understanding image structure

### 3. Monitor Training

Open TensorBoard to monitor training progress:
```bash
tensorboard --logdir ./tensorboard
```

### 4. Evaluate Models

**Classification Model:**
```bash
python evaluate_model.py --checkpoint ./checkpoints/vision_mamba_cifar10_tiny/best_model.pth --output_dir ./evaluation_results
```

**MAE Model:**
```bash
python evaluate_mae.py --checkpoint ./checkpoints/vision_mamba_cifar10_tiny_mae/best_model.pth --config cifar10_tiny_mae --output-dir ./mae_evaluation_results
```

**MAE Evaluation Demo:**
```bash
python demo_mae_evaluation.py
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

## Model Architectures

### Vision Mamba (Classification)

The Vision Mamba model combines:

1. **Patch Embedding**: Converts 2D images into 1D token sequences
2. **SSM Layers**: State Space Models for efficient sequence modeling
3. **Attention Layers**: Multi-head attention for global context
4. **Hybrid Blocks**: Learnable combination of SSM and attention
5. **Classification Head**: Final layer for class prediction

### Vision Mamba MAE (Masked Autoencoder)

The MAE architecture consists of:

1. **Encoder**: Vision Mamba model (without classification head)
   - Processes only visible (unmasked) patches (~25% of image)
   - Same hybrid SSM + attention architecture as classification model
   
2. **Lightweight Decoder**: 
   - Smaller transformer decoder (8 layers, 512 dim by default)
   - Reconstructs all patches from encoded visible patches
   - Uses learnable mask tokens for missing patches
   
3. **Reconstruction Head**: Predicts pixel values for each patch

**Key MAE Features:**
- **High Masking Ratio**: 75% of patches are masked by default
- **Asymmetric Design**: Small decoder reduces computational cost
- **Self-Supervised Learning**: No labels required, learns from image structure
- **Patch-Level Reconstruction**: Learns fine-grained visual features

## Training Features

### Common Features (Both Classification & MAE)
- **Mixed Precision Training**: Automatic mixed precision for faster training
- **Early Stopping**: Prevents overfitting with patience-based stopping
- **Learning Rate Scheduling**: Cosine annealing, step, or plateau schedulers
- **Gradient Clipping**: Prevents gradient explosion
- **Checkpointing**: Automatic saving of best and intermediate models
- **TensorBoard Logging**: Real-time training monitoring

### Classification-Specific Features
- **Data Augmentation**: Random crops, flips, and normalization
- **Label Smoothing**: Regularization technique for better generalization
- **Class-wise Metrics**: Per-class precision, recall, and F1-scores

### MAE-Specific Features
- **Random Patch Masking**: Configurable masking ratio (default: 75%)
- **Reconstruction Loss**: MSE loss on masked patches only
- **Minimal Augmentation**: Light augmentation to preserve reconstruction targets
- **Visualization**: Automatic generation of original/masked/reconstructed comparisons

## Evaluation Features

### Classification Evaluation
- **Accuracy Metrics**: Overall and per-class accuracy
- **Confusion Matrix**: Visual representation of classification performance
- **Classification Report**: Precision, recall, and F1-scores
- **Sample Visualizations**: Visual inspection of model predictions

### MAE Evaluation
- **Reconstruction Metrics**: MSE, MAE, PSNR, SSIM (comprehensive quality assessment)
- **Visual Reconstruction**: Side-by-side comparison of original, masked, and reconstructed images
- **Mask Ratio Analysis**: Testing different masking ratios (25%, 50%, 75%, 90%)
- **Reconstruction Gallery**: Visual examples across multiple samples and mask ratios
- **Detailed Reports**: JSON export of all metrics and analysis results

## Output Structure

### Classification Training Output
```
├── logs/
│   └── vision_mamba_cifar10_tiny/
│       ├── training_log.json
│       ├── confusion_matrix.png
│       └── training_curves.png
├── checkpoints/
│   └── vision_mamba_cifar10_tiny/
│       ├── best_model.pth
│       ├── final_model.pth
│       └── checkpoint_epoch_*.pth
└── tensorboard/
    └── vision_mamba_cifar10_tiny/
        └── [tensorboard logs]
```

### MAE Training Output
```
├── logs/
│   └── vision_mamba_cifar10_tiny_mae/
│       ├── training_log.json
│       ├── mae_reconstruction.png
│       └── training_curves.png
├── checkpoints/
│   └── vision_mamba_cifar10_tiny_mae/
│       ├── best_model.pth
│       ├── final_model.pth
│       └── checkpoint_epoch_*.pth
└── tensorboard/
    └── vision_mamba_cifar10_tiny_mae/
        └── [tensorboard logs]
```

## Example Results

### Classification Results
The framework provides comprehensive evaluation including:

1. **Training Curves**: Loss and accuracy plots over epochs
2. **Confusion Matrix**: Both raw counts and normalized percentages
3. **Per-Class Performance**: Precision, recall, and F1-score for each class
4. **Sample Predictions**: Visual examples of model predictions

### MAE Results
The MAE training provides:

1. **Reconstruction Visualizations**: Original, masked, and reconstructed images
2. **Training Curves**: Reconstruction loss over epochs
3. **Mask Ratio Comparisons**: Results with different masking ratios (25%, 50%, 75%, 90%)
4. **Patch Quality Assessment**: Visual evaluation of reconstruction fidelity

**Typical MAE Results:**
- **75% masking**: Good balance between difficulty and reconstruction quality
- **90% masking**: More challenging, learns very robust features
- **50% masking**: Easier task, may lead to less robust representations

## MAE Training Guide

### Why Use MAE?

Masked Autoencoder training offers several advantages:

1. **Self-Supervised Learning**: No labeled data required
2. **Robust Feature Learning**: Learns to understand image structure and context
3. **Transfer Learning**: Pre-trained MAE models can be fine-tuned for classification
4. **Data Efficiency**: Can leverage unlabeled data for better representations

### MAE Configuration

Key MAE-specific parameters in the config file:

```yaml
model:
  mask_ratio: 0.75              # Fraction of patches to mask
  decoder_embed_dim: 512        # Decoder hidden dimension
  decoder_depth: 8              # Number of decoder layers
  decoder_num_heads: 8          # Decoder attention heads
  use_cls_token: false          # MAE doesn't use CLS token

training:
  learning_rate: 0.0015         # Slightly higher LR for MAE
  weight_decay: 0.05            # Higher weight decay
  normalize: false              # Better for visualization
```

### MAE Training Tips

1. **Masking Ratio**: 
   - Start with 75% (default)
   - Higher ratios (90%) for more challenging tasks
   - Lower ratios (50%) for easier datasets

2. **Learning Rate**: 
   - MAE typically needs higher learning rates than classification
   - Start with 1.5x the classification learning rate

3. **Batch Size**: 
   - Smaller batch sizes work well for MAE
   - Reduce if memory is limited

4. **Epochs**: 
   - MAE often converges faster than classification
   - 100-200 epochs usually sufficient

### Transfer Learning with MAE

After MAE pre-training, you can fine-tune for classification:

1. **Extract Encoder**: Use the trained encoder from MAE
2. **Add Classification Head**: Replace decoder with classification layer
3. **Fine-tune**: Train on labeled data with lower learning rate

## Customization

### Adding New Datasets

To add support for new datasets, modify the `create_data_loaders` function in both `train_vision_mamba.py` and `train_vision_mamba_mae.py`:

```python
elif config.data.dataset_name == "YOUR_DATASET":
    train_dataset = torchvision.datasets.YourDataset(...)
    # For classification only:
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

**Test Classification Setup:**
```bash
python demo_training.py
```

**Test MAE Setup:**
```bash
python demo_mae.py
```

### Available Configurations

List all available configuration files:
```bash
python train_vision_mamba.py --list-configs
python train_vision_mamba_mae.py --list-configs
```

## Command Reference

### Classification Training Commands

```bash
# Basic training
python train_vision_mamba.py --config cifar10_tiny

# Resume training from checkpoint
python train_vision_mamba.py --config cifar10_tiny --resume ./checkpoints/vision_mamba_cifar10_tiny/checkpoint_epoch_50.pth

# Use custom config file
python train_vision_mamba.py --config-path ./my_custom_config.yaml

# List available configurations
python train_vision_mamba.py --list-configs
```

### MAE Training Commands

```bash
# Basic MAE training
python train_vision_mamba_mae.py --config cifar10_tiny_mae

# Resume MAE training
python train_vision_mamba_mae.py --config cifar10_tiny_mae --resume ./checkpoints/vision_mamba_cifar10_tiny_mae/checkpoint_epoch_25.pth

# Custom MAE config
python train_vision_mamba_mae.py --config-path ./my_mae_config.yaml
```

### Demo Commands

```bash
# Test classification demo
python demo_training.py

# Test MAE demo with reconstruction visualization
python demo_mae.py

# Test MAE evaluation functionality
python demo_mae_evaluation.py

# Test fast attention performance
python demo_fast_attention.py

# Evaluate trained classification model
python evaluate_model.py --checkpoint ./checkpoints/vision_mamba_cifar10_tiny/best_model.pth --output_dir ./results

# Evaluate trained MAE model
python evaluate_mae.py --checkpoint ./checkpoints/vision_mamba_cifar10_tiny_mae/best_model.pth --config cifar10_tiny_mae
```

## Model Variants

The framework supports different model sizes:

| Model | Embed Dim | Depth | Heads | Parameters | Use Case |
|-------|-----------|-------|-------|------------|----------|
| Tiny  | 96        | 6     | 2     | ~2M        | Quick experiments, limited resources |
| Small | 384       | 12    | 6     | ~22M       | Balanced performance and efficiency |
| Base  | 768       | 12    | 12    | ~86M       | High performance applications |

## Contributing

Feel free to extend this framework with:
- Additional datasets
- New model variants
- Advanced training techniques
- Better evaluation metrics
- Transfer learning utilities
- MAE fine-tuning scripts

## License

This project is provided as-is for educational and research purposes.