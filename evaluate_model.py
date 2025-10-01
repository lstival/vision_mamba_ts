"""
Evaluation script for trained Vision Mamba model
Load a trained model and evaluate on test data with detailed metrics
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import argparse

from vision_mamba import create_vision_mamba_small, create_vision_mamba_base, create_vision_mamba_tiny
from config import get_config


def load_model(checkpoint_path, device):
    """Load trained model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
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
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def create_test_loader(config):
    """Create test data loader"""
    # Define transforms (same as validation)
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
    
    test_transform = transforms.Compose([
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return test_loader, class_names


def evaluate_model(model, test_loader, device, class_names):
    """Evaluate model on test data"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc="Evaluating")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_targets, all_predictions)
    
    # Classification report
    report = classification_report(
        all_targets, all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    return {
        'predictions': all_predictions,
        'targets': all_targets,
        'probabilities': all_probabilities,
        'accuracy': accuracy,
        'report': report
    }


def plot_detailed_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot detailed confusion matrix with percentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Raw counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix (Counts)')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    
    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Confusion Matrix (Normalized)')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_class_performance(report, class_names, save_path):
    """Plot per-class performance metrics"""
    metrics = ['precision', 'recall', 'f1-score']
    classes = class_names
    
    # Extract metrics for each class
    precision = [report[cls]['precision'] for cls in classes]
    recall = [report[cls]['recall'] for cls in classes]
    f1_score = [report[cls]['f1-score'] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 8))
    
    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1_score, width, label='F1-Score', alpha=0.8)
    
    ax.set_xlabel('Classes')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def visualize_predictions(model, test_loader, class_names, device, save_path, num_samples=16):
    """Visualize sample predictions"""
    model.eval()
    
    # Get a batch of test data
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    images, labels = images[:num_samples], labels[:num_samples]
    images = images.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
    
    # Move back to CPU for visualization
    images = images.cpu()
    predictions = predictions.cpu()
    probabilities = probabilities.cpu()
    
    # Plot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            # Denormalize image for display
            img = images[i].permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img = torch.clamp(img, 0, 1)
            
            ax.imshow(img)
            true_label = class_names[labels[i]]
            pred_label = class_names[predictions[i]]
            confidence = probabilities[i][predictions[i]] * 100
            
            color = 'green' if labels[i] == predictions[i] else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.1f}%',
                        color=color, fontsize=10)
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Vision Mamba model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results',
                       help='Directory to save evaluation results')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model, config = load_model(args.checkpoint, device)
    
    # Create test loader
    print("Creating test data loader...")
    test_loader, class_names = create_test_loader(config)
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Evaluate model
    print("Evaluating model...")
    results = evaluate_model(model, test_loader, device, class_names)
    
    # Print results
    print(f"\nTest Accuracy: {results['accuracy']:.4f}")
    print("\nDetailed Classification Report:")
    report_str = classification_report(
        results['targets'], results['predictions'],
        target_names=class_names
    )
    print(report_str)
    
    # Save classification report to file
    with open(os.path.join(args.output_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Test Accuracy: {results['accuracy']:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report_str)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix_detailed.png')
    plot_detailed_confusion_matrix(
        results['targets'], results['predictions'], class_names, cm_path
    )
    print(f"Detailed confusion matrix saved to: {cm_path}")
    
    # Plot per-class performance
    perf_path = os.path.join(args.output_dir, 'class_performance.png')
    plot_class_performance(results['report'], class_names, perf_path)
    print(f"Class performance plot saved to: {perf_path}")
    
    # Visualize sample predictions
    pred_path = os.path.join(args.output_dir, 'sample_predictions.png')
    visualize_predictions(model, test_loader, class_names, device, pred_path)
    print(f"Sample predictions saved to: {pred_path}")
    
    print(f"\nEvaluation completed! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()