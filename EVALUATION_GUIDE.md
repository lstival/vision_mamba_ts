# Vision Mamba Model Evaluation Guide

This guide explains how to evaluate your trained Vision Mamba models using the provided evaluation scripts.

## Available Evaluation Scripts

### 1. Enhanced Evaluation Script (Recommended)
**File**: `enhanced_evaluate.py`

This is the most comprehensive evaluation script that provides:
- ✅ **Auto-detection** of the best model checkpoint
- 📊 **Comprehensive metrics**: Accuracy, Precision, Recall, F1-Score (macro & weighted)
- 📈 **Rich visualizations**: Confusion matrices, per-class metrics, class distributions
- 💾 **Detailed JSON results** with model and dataset information
- ⚡ **Performance metrics**: Inference time and throughput analysis

#### Usage:
```bash
# Basic usage (auto-detects best CIFAR-10 model)
python enhanced_evaluate.py

# Specify different experiment
python enhanced_evaluate.py --experiment vision_mamba_cifar100 --config cifar100

# Custom output directory
python enhanced_evaluate.py --output_dir ./my_evaluation_results

# Use specific checkpoint
python enhanced_evaluate.py --checkpoint ./checkpoints/vision_mamba_cifar10/best_model.pth
```

#### Arguments:
- `--experiment`: Experiment name (default: `vision_mamba_cifar10`)
- `--config`: Configuration name (default: `cifar10`)
- `--checkpoint`: Specific checkpoint path (optional, auto-detects if not provided)
- `--output_dir`: Output directory (default: `./evaluation_results`)
- `--device`: Device to use (`auto`, `cpu`, `cuda`)

### 2. Original Evaluation Script
**File**: `evaluate_model.py`

Basic evaluation script that provides:
- 📊 Basic metrics and classification report
- 📈 Confusion matrix and per-class performance plots
- 🔍 Sample prediction visualizations

#### Usage:
```bash
python evaluate_model.py --checkpoint ./checkpoints/vision_mamba_cifar10/best_model.pth
```

### 3. Demo Script
**File**: `demo_evaluation.py`

Simple demo script that shows how to run evaluations and provides usage examples.

#### Usage:
```bash
python demo_evaluation.py
```

## Output Files

### Enhanced Evaluation Output
When you run `enhanced_evaluate.py`, it creates a timestamped folder with:

```
evaluation_results/
└── vision_mamba_cifar10_20241003_143022/
    ├── detailed_evaluation_results.json    # Comprehensive metrics in JSON
    ├── classification_report.txt           # Text-based classification report
    ├── confusion_matrices.png             # Raw and normalized confusion matrices
    ├── per_class_metrics.png              # Bar chart of precision, recall, F1
    └── class_distribution.png             # Test set class distribution
```

### JSON Results Structure
The `detailed_evaluation_results.json` contains:
```json
{
  "evaluation_info": {
    "timestamp": "2024-10-03T14:30:22",
    "inference_time_seconds": 12.45,
    "total_test_samples": 10000
  },
  "model_info": {
    "name": "vision_mamba_tiny",
    "total_parameters": 1234567,
    "architecture": { ... }
  },
  "test_metrics": {
    "overall": {
      "accuracy": 0.8542,
      "precision_macro": 0.8501,
      "recall_macro": 0.8495,
      "f1_macro": 0.8498
    },
    "per_class": { ... }
  }
}
```

## Supported Datasets

The evaluation scripts support the same datasets as the training script:
- ✅ **CIFAR-10**: 10 classes, 32x32 color images
- ✅ **CIFAR-100**: 100 classes, 32x32 color images  
- ✅ **Fashion-MNIST**: 10 classes, 28x28 grayscale images

## Model Architectures

Supports all Vision Mamba variants:
- 🔥 **vision_mamba_tiny**: Lightweight version
- 🚀 **vision_mamba_small**: Balanced performance
- 💪 **vision_mamba_base**: Full-featured version

## Prerequisites

Make sure you have:
1. A trained model checkpoint in `./checkpoints/[experiment_name]/`
2. Required Python packages installed (see `requirements.txt`)
3. Test dataset downloaded (handled automatically)

## Quick Start Example

```bash
# 1. Train a model (if you haven't already)
python train_vision_mamba.py --config cifar10

# 2. Run enhanced evaluation
python enhanced_evaluate.py

# 3. Check results
ls evaluation_results/
```

## Performance Interpretation

### Key Metrics:
- **Accuracy**: Overall correctness (higher is better)
- **Macro-avg**: Unweighted average across classes (good for balanced evaluation)
- **Weighted-avg**: Weighted by class frequency (accounts for class imbalance)
- **Per-class metrics**: Individual class performance

### Good Performance Indicators:
- ✅ Accuracy > 0.85 for CIFAR-10
- ✅ Balanced precision and recall across classes
- ✅ Small difference between macro and weighted averages
- ✅ Low inference time per sample

## Troubleshooting

### Common Issues:

1. **"No checkpoint found"**
   ```bash
   # Make sure you have trained a model first
   python train_vision_mamba.py --config cifar10
   ```

2. **"CUDA out of memory"**
   ```bash
   # Use CPU or reduce batch size
   python enhanced_evaluate.py --device cpu
   ```

3. **"Import error"**
   ```bash
   # Install required packages
   pip install -r requirements.txt
   ```

### Getting Help:

If you encounter issues:
1. Check that your model checkpoint exists
2. Verify your Python environment has all dependencies
3. Try running with `--device cpu` if you have GPU memory issues
4. Check the console output for specific error messages

## Advanced Usage

### Custom Configuration
You can modify the evaluation by:
1. Editing the configuration files in `config.py`
2. Using different model checkpoints
3. Adjusting output formats and visualizations

### Batch Evaluation
To evaluate multiple models:
```bash
# Evaluate different experiments
python enhanced_evaluate.py --experiment vision_mamba_cifar10 --config cifar10
python enhanced_evaluate.py --experiment vision_mamba_cifar100 --config cifar100
python enhanced_evaluate.py --experiment vision_mamba_fashion_mnist --config fashion_mnist
```

---

This evaluation system provides comprehensive analysis of your Vision Mamba models, helping you understand their performance and make informed decisions about model deployment and further improvements.