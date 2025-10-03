"""
Training script for Vision Mamba model
Supports various torchvision datasets with comprehensive logging and evaluation
"""
import os
import sys
import json
import time
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# Import our modules
from vision_mamba import create_vision_mamba_small, create_vision_mamba_base, create_vision_mamba_tiny, VisionMamba
from yaml_config import Config, get_config, list_available_configs


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=15, min_delta=1e-4, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        score = -val_loss
        
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


class MetricsTracker:
    """Track and compute training metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = defaultdict(list)
        self.running_loss = 0.0
        self.running_correct = 0
        self.total_samples = 0
        
    def update(self, loss, predictions, targets):
        batch_size = targets.size(0)
        self.running_loss += loss * batch_size
        self.running_correct += (predictions == targets).sum().item()
        self.total_samples += batch_size
        
    def get_metrics(self):
        avg_loss = self.running_loss / self.total_samples
        accuracy = self.running_correct / self.total_samples
        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }
        
    def log_metrics(self, metrics, prefix=''):
        for key, value in metrics.items():
            self.metrics[f"{prefix}{key}"].append(value)


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config):
    """Get the appropriate device"""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_config)


def create_data_loaders(config: Config):
    """Create data loaders for training and validation"""
    
    # Define transforms
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
    
    # Training transforms
    train_transforms = []
    if config.data.random_crop and config.data.dataset_name not in ["FashionMNIST"]:
        train_transforms.append(transforms.RandomCrop(32, padding=4))
    if config.data.random_flip:
        train_transforms.append(transforms.RandomHorizontalFlip())
    
    train_transforms.extend([
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    # Validation transforms
    val_transforms = [
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
        normalize
    ]
    
    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)
    
    # Load dataset
    if config.data.dataset_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                      'dog', 'frog', 'horse', 'ship', 'truck']
        
    elif config.data.dataset_name == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        class_names = [f"class_{i}" for i in range(100)]
        
    elif config.data.dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                      'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    elif config.data.dataset_name == "imagenet":
        train_dataset = torchvision.datasets.ImageNet(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.ImageNet(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        class_names = []
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset_name}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.data.val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory
    )
    
    return train_loader, val_loader, class_names


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


def create_optimizer(model, config: Config):
    """Create optimizer"""
    if config.training.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.training.learning_rate,
            weight_decay=config.training.weight_decay
        )
    elif config.training.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.training.learning_rate,
            momentum=0.9,
            weight_decay=config.training.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {config.training.optimizer}")
    
    return optimizer


def create_scheduler(optimizer, config: Config, steps_per_epoch):
    """Create learning rate scheduler"""
    if config.training.scheduler.lower() == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.epochs,
            eta_min=config.training.min_lr
        )
    elif config.training.scheduler.lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.training.step_size,
            gamma=config.training.gamma
        )
    elif config.training.scheduler.lower() == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.training.gamma,
            patience=config.training.patience // 2,
            min_lr=config.training.min_lr
        )
    else:
        scheduler = None
    
    return scheduler


def train_epoch(model, train_loader, criterion, optimizer, device, config, scaler=None):
    """Train for one epoch"""
    model.train()
    metrics_tracker = MetricsTracker()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, targets) in enumerate(pbar):
        data, targets = data.to(device), targets.to(device)
        
        optimizer.zero_grad()
        
        if config.training.use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(data)
                loss = criterion(outputs, targets)
            
            scaler.scale(loss).backward()
            if config.training.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            
            if config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
        
        # Update metrics
        predictions = outputs.argmax(dim=1)
        metrics_tracker.update(loss.item(), predictions, targets)
        
        # Update progress bar
        if batch_idx % config.logging.log_interval == 0:
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'Loss': f"{current_metrics['loss']:.4f}",
                'Acc': f"{current_metrics['accuracy']:.4f}"
            })
    
    return metrics_tracker.get_metrics()


def validate_epoch(model, val_loader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    metrics_tracker = MetricsTracker()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for data, targets in pbar:
            data, targets = data.to(device), targets.to(device)
            
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            predictions = outputs.argmax(dim=1)
            metrics_tracker.update(loss.item(), predictions, targets)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            current_metrics = metrics_tracker.get_metrics()
            pbar.set_postfix({
                'Loss': f"{current_metrics['loss']:.4f}",
                'Acc': f"{current_metrics['accuracy']:.4f}"
            })
    
    metrics = metrics_tracker.get_metrics()
    return metrics, all_predictions, all_targets


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_metrics, val_metrics, save_path):
    """Plot training curves"""
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(epochs, train_metrics['loss'], 'b-', label='Training Loss')
    ax1.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(epochs, train_metrics['accuracy'], 'b-', label='Training Accuracy')
    ax2.plot(epochs, val_metrics['accuracy'], 'r-', label='Validation Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_log(config, train_metrics, val_metrics, best_epoch, save_path):
    """Save training log as JSON"""
    log_data = {
        'experiment_name': config.experiment_name,
        'config': {
            'model': {
                'name': config.model.model_name,
                'embed_dim': config.model.embed_dim,
                'depth': config.model.depth,
                'num_heads': config.model.num_heads
            },
            'data': {
                'dataset': config.data.dataset_name,
                'batch_size': config.data.batch_size,
                'num_classes': config.data.num_classes
            },
            'training': {
                'epochs': config.training.epochs,
                'learning_rate': config.training.learning_rate,
                'optimizer': config.training.optimizer,
                'scheduler': config.training.scheduler
            }
        },
        'results': {
            'best_epoch': best_epoch,
            'best_val_acc': max(val_metrics['accuracy']),
            'best_val_loss': min(val_metrics['loss']),
            'final_train_acc': train_metrics['accuracy'][-1],
            'final_train_loss': train_metrics['loss'][-1]
        },
        'training_history': {
            'train_loss': train_metrics['loss'],
            'train_accuracy': train_metrics['accuracy'],
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy']
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def main():
    """Main training function"""
    # Parse arguments or use default config
    import argparse
    parser = argparse.ArgumentParser(description='Train Vision Mamba')
    parser.add_argument('--config', type=str, default='cifar10_tiny',
                       help='Configuration name (without .yaml extension)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Full path to configuration file')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
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
    
    # Get configuration
    if args.config_path:
        from yaml_config import load_config
        config = load_config(args.config_path)
    else:
        config = get_config(args.config)
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader, class_names = create_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
    # Create loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)
    
    # Create AMP scaler if using mixed precision
    scaler = torch.cuda.amp.GradScaler() if config.training.use_amp and device.type == 'cuda' else None
    
    # Create early stopping
    early_stopping = EarlyStopping(
        patience=config.training.patience,
        min_delta=config.training.min_delta
    )
    
    # Create tensorboard writer
    writer = SummaryWriter(config.logging.tensorboard_dir)
    
    # Training loop
    print(f"Starting training for {config.training.epochs} epochs...")
    start_time = time.time()
    
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch+1}/{config.training.epochs}")
        
        # Train
        epoch_train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, config, scaler
        )
        
        # Validate
        if (epoch + 1) % config.logging.val_interval == 0:
            epoch_val_metrics, val_predictions, val_targets = validate_epoch(
                model, val_loader, criterion, device
            )
        else:
            epoch_val_metrics = {'loss': float('inf'), 'accuracy': 0.0}
            val_predictions, val_targets = [], []
        
        # Update learning rate
        if scheduler is not None:
            if config.training.scheduler.lower() == "plateau":
                scheduler.step(epoch_val_metrics['loss'])
            else:
                scheduler.step()
        
        # Log metrics
        for key, value in epoch_train_metrics.items():
            train_metrics[key].append(value)
        for key, value in epoch_val_metrics.items():
            val_metrics[key].append(value)
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', epoch_train_metrics['loss'], epoch)
        writer.add_scalar('Loss/Val', epoch_val_metrics['loss'], epoch)
        writer.add_scalar('Accuracy/Train', epoch_train_metrics['accuracy'], epoch)
        writer.add_scalar('Accuracy/Val', epoch_val_metrics['accuracy'], epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        print(f"Train Loss: {epoch_train_metrics['loss']:.4f}, "
              f"Train Acc: {epoch_train_metrics['accuracy']:.4f}")
        print(f"Val Loss: {epoch_val_metrics['loss']:.4f}, "
              f"Val Acc: {epoch_val_metrics['accuracy']:.4f}")
        
        # Save best model
        if epoch_val_metrics['accuracy'] > best_val_acc:
            best_val_acc = epoch_val_metrics['accuracy']
            best_epoch = epoch
            
            if config.logging.save_best_only:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_acc': best_val_acc,
                    'config': config
                }, os.path.join(config.logging.checkpoint_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % config.logging.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_acc': epoch_val_metrics['accuracy'],
                'config': config
            }, os.path.join(config.logging.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if early_stopping(epoch_val_metrics['loss'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch+1}")
    
    # Final evaluation
    print("\nFinal evaluation...")
    final_val_metrics, final_predictions, final_targets = validate_epoch(
        model, val_loader, criterion, device
    )
    
    # Generate classification report
    report = classification_report(final_targets, final_predictions, 
                                 target_names=class_names, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(final_targets, final_predictions, target_names=class_names))
    
    # Plot confusion matrix
    if config.logging.plot_confusion_matrix:
        cm_path = os.path.join(config.logging.log_dir, 'confusion_matrix.png')
        plot_confusion_matrix(final_targets, final_predictions, class_names, cm_path)
        print(f"Confusion matrix saved to: {cm_path}")
    
    # Plot training curves
    if config.logging.plot_training_curves:
        curves_path = os.path.join(config.logging.log_dir, 'training_curves.png')
        plot_training_curves(train_metrics, val_metrics, curves_path)
        print(f"Training curves saved to: {curves_path}")
    
    # Save training log
    log_path = os.path.join(config.logging.log_dir, 'training_log.json')
    save_training_log(config, train_metrics, val_metrics, best_epoch, log_path)
    print(f"Training log saved to: {log_path}")
    
    # Save final model
    if config.logging.save_last:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': final_val_metrics['accuracy'],
            'config': config,
            'classification_report': report
        }, os.path.join(config.logging.checkpoint_dir, 'final_model.pth'))
    
    writer.close()
    print("Training completed successfully!")


if __name__ == "__main__":
    main()