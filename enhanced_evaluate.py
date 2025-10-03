"""
Enhanced evaluation script for Vision Mamba model
Automatically loads the best model checkpoint and provides comprehensive evaluation
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import (
    confusion_matrix, 
    classification_report, 
    accuracy_score,
    precision_recall_fscore_support,
    top_k_accuracy_score,
    roc_auc_score
)

# Import our modules
from vision_mamba import create_vision_mamba_small, create_vision_mamba_base, create_vision_mamba_tiny
from yaml_config import Config, get_config, list_available_configs


def find_best_checkpoint(experiment_name="vision_mamba_cifar10"):
    """Automatically find the best model checkpoint"""
    checkpoint_dir = os.path.join("checkpoints", experiment_name)
    best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
    
    if os.path.exists(best_model_path):
        return best_model_path
    
    # If best_model.pth doesn't exist, look for other checkpoints
    if os.path.exists(checkpoint_dir):
        checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
        if checkpoint_files:
            # Return the most recent checkpoint
            checkpoint_files.sort()
            return os.path.join(checkpoint_dir, checkpoint_files[-1])
    
    raise FileNotFoundError(f"No checkpoint found in {checkpoint_dir}")


def get_device(device_config):
    """Get the appropriate device"""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_config)


def create_test_loader(config: Config):
    """Create test data loader"""
    
    # Define transforms (same as validation in training)
    if config.data.normalize:
        if config.data.dataset_name in ["CIFAR10", "CIFAR100"]:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif config.data.dataset_name == "FashionMNIST":
            normalize = transforms.Normalize(
                mean=[0.286],
                std=[0.353]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
    else:
        normalize = transforms.Lambda(lambda x: x)
    
    # Test transforms (no augmentation)
    test_transforms = [
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
        normalize
    ]
    
    test_transform = transforms.Compose(test_transforms)
    
    # Load test dataset
    if config.data.dataset_name == "CIFAR10":
        test_dataset = torchvision.datasets.CIFAR10(
            root=config.data.data_dir, train=False, download=True, transform=test_transform
        )
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
    elif config.data.dataset_name == "CIFAR100":
        test_dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=False, download=True, transform=test_transform
        )
        class_names = [f"class_{i}" for i in range(100)]
        
    elif config.data.dataset_name == "FashionMNIST":
        test_dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=False, download=True, transform=test_transform
        )
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset_name}")
    
    # Create test loader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return test_loader, class_names


def create_model(config: Config):
    """Create the Vision Mamba model"""
    if config.model.model_name == "vision_mamba_small":
        model = create_vision_mamba_small(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            in_channels=config.model.in_channels,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            d_state=config.model.d_state,
            d_conv=config.model.d_conv,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            num_classes=config.data.num_classes,
            use_cls_token=config.model.use_cls_token
        )
    elif config.model.model_name == "vision_mamba_tiny":
        model = create_vision_mamba_tiny(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            in_channels=config.model.in_channels,
            embed_dim=config.model.embed_dim,
            depth=config.model.depth,
            d_state=config.model.d_state,
            d_conv=config.model.d_conv,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
            num_classes=config.data.num_classes,
            use_cls_token=config.model.use_cls_token
        )
    elif config.model.model_name == "vision_mamba_base":
        model = create_vision_mamba_base(
            img_size=config.model.img_size,
            patch_size=config.model.patch_size,
            in_channels=config.model.in_channels,
            num_classes=config.data.num_classes,
            use_cls_token=config.model.use_cls_token
        )
    else:
        raise ValueError(f"Unknown model: {config.model.model_name}")
    
    return model


def load_checkpoint(checkpoint_path, model, device):
    """Load model from checkpoint"""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Get training info
    epoch = checkpoint.get('epoch', 'unknown')
    val_acc = checkpoint.get('val_acc', 'unknown')
    
    print(f"Loaded checkpoint from epoch {epoch}")
    if val_acc != 'unknown':
        print(f"Validation accuracy during training: {val_acc:.4f}")
    
    return model, checkpoint


def evaluate_model(model, test_loader, device):
    """Evaluate model on test set with detailed metrics"""
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_probabilities = []
    batch_losses = []
    
    criterion = nn.CrossEntropyLoss()
    
    print("Evaluating model on test set...")
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Testing")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Get predictions and probabilities
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            # Store results
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            batch_losses.append(loss.item())
            
            # Update progress bar
            batch_acc = (predictions == targets).float().mean().item()
            pbar.set_postfix({'Batch Acc': f"{batch_acc:.4f}"})
    
    avg_loss = np.mean(batch_losses)
    
    return all_predictions, all_targets, all_probabilities, avg_loss


def compute_comprehensive_metrics(y_true, y_pred, y_prob, class_names):
    """Compute comprehensive evaluation metrics"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names)), zero_division=0
    )
    
    # Macro and weighted averages
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average='macro', zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted', zero_division=0
    )
    
    metrics['precision_macro'] = precision_macro
    metrics['recall_macro'] = recall_macro
    metrics['f1_macro'] = f1_macro
    metrics['precision_weighted'] = precision_weighted
    metrics['recall_weighted'] = recall_weighted
    metrics['f1_weighted'] = f1_weighted
    
    # Top-k accuracy (if more than 5 classes)
    if len(class_names) >= 5:
        y_prob_array = np.array(y_prob)
        try:
            metrics['top_5_accuracy'] = top_k_accuracy_score(y_true, y_prob_array, k=min(5, len(class_names)))
        except:
            metrics['top_5_accuracy'] = None
    
    # Per-class details
    metrics['per_class'] = {}
    for i, class_name in enumerate(class_names):
        metrics['per_class'][class_name] = {
            'precision': precision[i] if not np.isnan(precision[i]) else 0.0,
            'recall': recall[i] if not np.isnan(recall[i]) else 0.0,
            'f1_score': f1[i] if not np.isnan(f1[i]) else 0.0,
            'support': int(support[i])
        }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()
    
    return metrics


def create_comprehensive_plots(y_true, y_pred, y_prob, class_names, metrics, output_dir):
    """Create comprehensive evaluation plots"""
    
    # 1. Confusion Matrix (both raw and normalized)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)', fontsize=14)
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('Actual', fontsize=12)
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)', fontsize=14)
    ax2.set_xlabel('Predicted', fontsize=12)
    ax2.set_ylabel('Actual', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-class metrics bar chart
    per_class = metrics['per_class']
    precision_scores = [per_class[name]['precision'] for name in class_names]
    recall_scores = [per_class[name]['recall'] for name in class_names]
    f1_scores = [per_class[name]['f1_score'] for name in class_names]
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(16, 8))
    
    bars1 = ax.bar(x - width, precision_scores, width, label='Precision', alpha=0.8, color='skyblue')
    bars2 = ax.bar(x, recall_scores, width, label='Recall', alpha=0.8, color='lightcoral')
    bars3 = ax.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8, color='lightgreen')
    
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Scores', fontsize=12)
    ax.set_title('Per-Class Performance Metrics', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if height > 0
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
    
    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_class_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Class distribution and support
    support_counts = [per_class[name]['support'] for name in class_names]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(class_names, support_counts, alpha=0.7, color='orange')
    ax.set_title('Test Set Class Distribution', fontsize=14)
    ax.set_xlabel('Classes', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
               f'{int(height)}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'class_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()


def save_detailed_results(metrics, config, checkpoint_info, inference_time, output_dir):
    """Save detailed evaluation results to JSON"""
    
    # Calculate additional stats
    total_params = sum(p.numel() for p in create_model(config).parameters())
    
    results = {
        'evaluation_info': {
            'timestamp': datetime.now().isoformat(),
            'inference_time_seconds': inference_time,
            'total_test_samples': sum([metrics['per_class'][cls]['support'] for cls in metrics['per_class']])
        },
        'model_info': {
            'name': config.model.model_name,
            'total_parameters': int(total_params),
            'architecture': {
                'embed_dim': config.model.embed_dim,
                'depth': config.model.depth,
                'num_heads': config.model.num_heads,
                'img_size': config.model.img_size,
                'patch_size': config.model.patch_size,
                'd_state': config.model.d_state,
                'd_conv': config.model.d_conv
            }
        },
        'dataset_info': {
            'name': config.data.dataset_name,
            'num_classes': config.data.num_classes,
            'test_batch_size': config.data.val_batch_size
        },
        'checkpoint_info': {
            'epoch': checkpoint_info.get('epoch', 'unknown'),
            'training_val_acc': float(checkpoint_info.get('val_acc', 0.0)) if checkpoint_info.get('val_acc', 'unknown') != 'unknown' else None,
        },
        'test_metrics': {
            'overall': {
                'accuracy': float(metrics['accuracy']),
                'precision_macro': float(metrics['precision_macro']),
                'recall_macro': float(metrics['recall_macro']),
                'f1_macro': float(metrics['f1_macro']),
                'precision_weighted': float(metrics['precision_weighted']),
                'recall_weighted': float(metrics['recall_weighted']),
                'f1_weighted': float(metrics['f1_weighted'])
            },
            'per_class': {}
        }
    }
    
    # Add top-k accuracy if available
    if 'top_5_accuracy' in metrics and metrics['top_5_accuracy'] is not None:
        results['test_metrics']['overall']['top_5_accuracy'] = float(metrics['top_5_accuracy'])
    
    # Add per-class results
    for class_name, class_metrics in metrics['per_class'].items():
        results['test_metrics']['per_class'][class_name] = {
            'precision': float(class_metrics['precision']),
            'recall': float(class_metrics['recall']),
            'f1_score': float(class_metrics['f1_score']),
            'support': int(class_metrics['support'])
        }
    
    # Save to file
    results_path = os.path.join(output_dir, 'detailed_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return results_path


def print_evaluation_summary(metrics, class_names, inference_time, total_samples):
    """Print comprehensive evaluation summary"""
    print("\n" + "="*80)
    print("VISION MAMBA MODEL EVALUATION SUMMARY")
    print("="*80)
    
    print(f"\nOverall Performance:")
    print(f"  • Test Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  • Macro-avg Precision: {metrics['precision_macro']:.4f}")
    print(f"  • Macro-avg Recall: {metrics['recall_macro']:.4f}")
    print(f"  • Macro-avg F1-Score: {metrics['f1_macro']:.4f}")
    print(f"  • Weighted-avg Precision: {metrics['precision_weighted']:.4f}")
    print(f"  • Weighted-avg Recall: {metrics['recall_weighted']:.4f}")
    print(f"  • Weighted-avg F1-Score: {metrics['f1_weighted']:.4f}")
    
    if 'top_5_accuracy' in metrics and metrics['top_5_accuracy'] is not None:
        print(f"  • Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    
    print(f"\nInference Performance:")
    print(f"  • Total inference time: {inference_time:.2f} seconds")
    print(f"  • Average time per sample: {(inference_time/total_samples)*1000:.2f} ms")
    print(f"  • Throughput: {total_samples/inference_time:.1f} samples/second")
    
    # Find best and worst performing classes
    per_class = metrics['per_class']
    f1_scores = [(name, data['f1_score']) for name, data in per_class.items()]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\nBest Performing Classes (by F1-Score):")
    for i, (class_name, f1_score) in enumerate(f1_scores[:3]):
        print(f"  {i+1}. {class_name}: {f1_score:.4f}")
    
    print(f"\nWorst Performing Classes (by F1-Score):")
    for i, (class_name, f1_score) in enumerate(f1_scores[-3:]):
        print(f"  {len(f1_scores)-2+i}. {class_name}: {f1_score:.4f}")
    
    print("\n" + "="*80)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Enhanced Vision Mamba Model Evaluation')
    parser.add_argument('--experiment', type=str, default='vision_mamba_cifar10',
                       help='Experiment name (corresponds to checkpoint folder)')
    parser.add_argument('--config', type=str, default='cifar10_tiny',
                       help='Configuration name (without .yaml extension)')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Specific checkpoint path (if not provided, will auto-detect best model)')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--list-configs', action='store_true',
                       help='List available configurations and exit')
    args = parser.parse_args()
    
    # Handle list configs
    if args.list_configs:
        print("Available configurations:")
        configs = list_available_configs()
        for config_name in configs:
            print(f"  - {config_name}")
        return
    
    print("Starting Enhanced Vision Mamba Model Evaluation...")
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"{args.experiment}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find checkpoint
    if args.checkpoint is None:
        try:
            checkpoint_path = find_best_checkpoint(args.experiment)
            print(f"Auto-detected checkpoint: {checkpoint_path}")
        except FileNotFoundError as e:
            print(f"Error: {e}")
            return
    else:
        checkpoint_path = args.checkpoint
    
    # Get configuration
    config = get_config(args.config)
    config.device = args.device
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create test data loader
    print("Creating test data loader...")
    test_loader, class_names = create_test_loader(config)
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Number of classes: {len(class_names)}")
    print(f"Class names: {class_names}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load checkpoint
    model, checkpoint_info = load_checkpoint(checkpoint_path, model, device)
    
    # Evaluate model
    start_time = time.time()
    predictions, targets, probabilities, avg_loss = evaluate_model(model, test_loader, device)
    inference_time = time.time() - start_time
    
    print(f"\nEvaluation completed!")
    print(f"Average test loss: {avg_loss:.4f}")
    
    # Compute comprehensive metrics
    print("Computing comprehensive metrics...")
    metrics = compute_comprehensive_metrics(targets, predictions, probabilities, class_names)
    
    # Print summary
    print_evaluation_summary(metrics, class_names, inference_time, len(targets))
    
    # Create comprehensive plots
    print("Generating comprehensive visualizations...")
    create_comprehensive_plots(targets, predictions, probabilities, class_names, metrics, output_dir)
    
    # Save detailed results
    print("Saving detailed results...")
    results_path = save_detailed_results(metrics, config, checkpoint_info, inference_time, output_dir)
    
    # Save classification report
    report_path = os.path.join(output_dir, 'classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Vision Mamba Model - Classification Report\n")
        f.write("="*50 + "\n\n")
        f.write(f"Overall Accuracy: {metrics['accuracy']:.4f}\n")
        f.write(f"Test Loss: {avg_loss:.4f}\n")
        f.write(f"Inference Time: {inference_time:.2f} seconds\n\n")
        f.write(classification_report(targets, predictions, target_names=class_names))
    
    print(f"\nEvaluation completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated files:")
    print(f"  • {results_path}")
    print(f"  • {report_path}")
    print(f"  • confusion_matrices.png")
    print(f"  • per_class_metrics.png")
    print(f"  • class_distribution.png")


if __name__ == "__main__":
    main()