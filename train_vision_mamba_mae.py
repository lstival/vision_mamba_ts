"""
Training script for Vision Mamba Masked Autoencoder (MAE)
Supports various torchvision datasets with comprehensive logging and visualization
"""

import os
import sys
import argparse
import json
import time
import random
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error

# Import our modules
from vision_mamba_mae import create_vision_mamba_mae, VisionMambaMAE
from yaml_config import Config, get_config, list_available_configs


class EarlyStopping:
    """Early stopping utility for MAE training"""
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


class MAEMetricsTracker:
    """Track and compute MAE training metrics"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.metrics = defaultdict(list)
        self.running_loss = 0.0
        self.total_samples = 0
        
    def update(self, loss, batch_size):
        self.running_loss += loss * batch_size
        self.total_samples += batch_size
        
    def get_metrics(self):
        avg_loss = self.running_loss / self.total_samples
        return {
            'loss': avg_loss
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
    """Create data loaders for MAE training"""
    
    # Define transforms - simpler for MAE since we're reconstructing original images
    # We don't want heavy augmentations that make reconstruction too difficult
    base_transforms = [
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
    ]
    
    # Optional normalization
    if config.data.normalize:
        if config.data.dataset_name in ["CIFAR10", "CIFAR100"]:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        elif config.data.dataset_name == "FashionMNIST":
            normalize = transforms.Normalize(
                mean=[0.5], std=[0.5]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        base_transforms.append(normalize)
    
    # Light augmentation for training (optional)
    train_transforms = []
    if config.data.use_augmentation:
        if config.data.random_flip:
            train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        # Add small random crop for slight variation
        if config.data.random_crop and config.data.dataset_name not in ["FashionMNIST"]:
            train_transforms.append(transforms.RandomCrop(
                # (config.model.img_size, config.model.img_size), 
                (40, 40), 
                padding=4
            ))
    
    train_transforms.extend(base_transforms)
    val_transforms = base_transforms.copy()
    
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
        
    elif config.data.dataset_name == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        
    elif config.data.dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        
    elif config.data.dataset_name == "imagenet":
        train_dataset = torchvision.datasets.ImageNet(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.ImageNet(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
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
    
    return train_loader, val_loader


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


def train_epoch(model, train_loader, optimizer, device, config, scaler=None):
    """Train for one epoch"""
    model.train()
    metrics_tracker = MAEMetricsTracker()
    
    pbar = tqdm(train_loader, desc="Training")
    for batch_idx, (data, _) in enumerate(pbar):  # Ignore labels for MAE
        data = data.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        if config.training.use_amp and scaler is not None:
            with torch.cuda.amp.autocast():
                loss, pred, mask = model(data)
            
            scaler.scale(loss).backward()
            if config.training.max_grad_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss, pred, mask = model(data)
            loss.backward()
            
            if config.training.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
            optimizer.step()
        
        # Update metrics
        metrics_tracker.update(loss.item(), data.size(0))
        
        # Update progress bar
        if batch_idx % config.logging.log_interval == 0:
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Mask': f'{mask.mean().item():.3f}'
            })
    
    return metrics_tracker.get_metrics()


def validate_epoch(model, val_loader, device):
    """Validate for one epoch"""
    model.eval()
    metrics_tracker = MAEMetricsTracker()
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for data, _ in pbar:  # Ignore labels for MAE
            data = data.to(device, non_blocking=True)
            
            loss, pred, mask = model(data)
            
            # Update metrics
            metrics_tracker.update(loss.item(), data.size(0))
            
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Mask': f'{mask.mean().item():.3f}'
            })
    
    return metrics_tracker.get_metrics()


def visualize_mae_results(model, data_loader, device, save_path, num_samples=8):
    """Visualize MAE reconstruction results"""
    model.eval()
    
    # Get a batch of images
    data_iter = iter(data_loader)
    images, _ = next(data_iter)
    images = images[:num_samples].to(device)
    
    with torch.no_grad():
        results = model.visualize_reconstruction(images)
    
    # Convert to numpy and denormalize if needed
    original = results['original'].cpu()
    masked = results['masked'].cpu()
    reconstructed = results['reconstructed'].cpu()
    
    # Create visualization
    fig, axes = plt.subplots(3, num_samples, figsize=(num_samples * 2, 6))
    
    for i in range(num_samples):
        # Original image
        img_orig = original[i].permute(1, 2, 0)
        if img_orig.shape[2] == 1:  # Grayscale
            axes[0, i].imshow(img_orig.squeeze(), cmap='gray')
        else:
            # Clamp values to [0, 1] for display
            img_orig = torch.clamp(img_orig, 0, 1)
            axes[0, i].imshow(img_orig)
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Masked image
        img_masked = masked[i].permute(1, 2, 0)
        if img_masked.shape[2] == 1:  # Grayscale
            axes[1, i].imshow(img_masked.squeeze(), cmap='gray')
        else:
            img_masked = torch.clamp(img_masked, 0, 1)
            axes[1, i].imshow(img_masked)
        axes[1, i].set_title('Masked')
        axes[1, i].axis('off')
        
        # Reconstructed image
        img_recon = reconstructed[i].permute(1, 2, 0)
        if img_recon.shape[2] == 1:  # Grayscale
            axes[2, i].imshow(img_recon.squeeze(), cmap='gray')
        else:
            img_recon = torch.clamp(img_recon, 0, 1)
            axes[2, i].imshow(img_recon)
        axes[2, i].set_title('Reconstructed')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_training_curves(train_metrics, val_metrics, save_path):
    """Plot training curves for MAE"""
    epochs = range(1, len(train_metrics['loss']) + 1)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Loss curves
    ax.plot(epochs, train_metrics['loss'], 'b-', label='Training Loss')
    ax.plot(epochs, val_metrics['loss'], 'r-', label='Validation Loss')
    ax.set_title('MAE Training and Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Reconstruction Loss')
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_training_log(config, train_metrics, val_metrics, best_epoch, save_path):
    """Save training log as JSON"""
    log_data = {
        'experiment_name': config.experiment_name,
        'model_type': 'MAE',
        'config': {
            'model': {
                'name': config.model.model_name,
                'embed_dim': config.model.embed_dim,
                'depth': config.model.depth,
                'num_heads': config.model.num_heads,
                'mask_ratio': getattr(config.model, 'mask_ratio', 0.75),
                'decoder_embed_dim': getattr(config.model, 'decoder_embed_dim', 512),
                'decoder_depth': getattr(config.model, 'decoder_depth', 8)
            },
            'data': {
                'dataset': config.data.dataset_name,
                'batch_size': config.data.batch_size,
                'img_size': config.model.img_size
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
            'best_val_loss': min(val_metrics['loss']),
            'final_train_loss': train_metrics['loss'][-1],
            'final_val_loss': val_metrics['loss'][-1]
        },
        'training_history': {
            'train_loss': train_metrics['loss'],
            'val_loss': val_metrics['loss']
        }
    }
    
    with open(save_path, 'w') as f:
        json.dump(log_data, f, indent=2)


def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train Vision Mamba MAE')
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
    
    # Generate timestamp for unique model naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    
    # Update experiment name for MAE with timestamp
    config.experiment_name = f"{config.experiment_name}_mae_{timestamp}"
    
    print(f"Experiment name: {config.experiment_name}")
    print(f"Timestamp: {timestamp}")
    
    # Add MAE-specific model parameters if not present
    if not hasattr(config.model, 'mask_ratio'):
        config.model.mask_ratio = 0.75
    if not hasattr(config.model, 'decoder_embed_dim'):
        config.model.decoder_embed_dim = 192
    if not hasattr(config.model, 'decoder_depth'):
        config.model.decoder_depth = 16
    if not hasattr(config.model, 'decoder_num_heads'):
        config.model.decoder_num_heads = 2
    
    # Update logging directories
    config.logging.log_dir = os.path.join(config.logging.log_dir, config.experiment_name)
    config.logging.checkpoint_dir = os.path.join(config.logging.checkpoint_dir, config.experiment_name)
    config.logging.tensorboard_dir = os.path.join(config.logging.tensorboard_dir, config.experiment_name)
    
    # Create directories
    os.makedirs(config.logging.log_dir, exist_ok=True)
    os.makedirs(config.logging.checkpoint_dir, exist_ok=True)
    os.makedirs(config.logging.tensorboard_dir, exist_ok=True)
    
    print(f"Output directories created:")
    print(f"  - Logs: {config.logging.log_dir}")
    print(f"  - Checkpoints: {config.logging.checkpoint_dir}")
    print(f"  - Tensorboard: {config.logging.tensorboard_dir}")
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    # Create data loaders
    print("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    
    # Create model
    print("Creating MAE model...")
    model = create_vision_mamba_mae(config)
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Decoder/Encoder ratio: {decoder_params/encoder_params:.3f}")
    print(f"Mask ratio: {config.model.mask_ratio}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))
    
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
    print(f"Starting MAE training for {config.training.epochs} epochs...")
    start_time = time.time()
    
    train_metrics = defaultdict(list)
    val_metrics = defaultdict(list)
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(config.training.epochs):
        print(f"\nEpoch {epoch+1}/{config.training.epochs}")
        
        # Train
        epoch_train_metrics = train_epoch(
            model, train_loader, optimizer, device, config, scaler
        )
        
        # Validate
        if (epoch + 1) % config.logging.val_interval == 0:
            epoch_val_metrics = validate_epoch(model, val_loader, device)
        else:
            # Use previous validation metrics if not validating this epoch
            epoch_val_metrics = {'loss': val_metrics['loss'][-1]} if val_metrics['loss'] else {'loss': 0.0}
        
        # Update learning rate
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
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
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        print(f"Train Loss: {epoch_train_metrics['loss']:.6f}")
        print(f"Val Loss: {epoch_val_metrics['loss']:.6f}")
        
        # Save best model
        if epoch_val_metrics['loss'] < best_val_loss:
            best_val_loss = epoch_val_metrics['loss']
            best_epoch = epoch
            
            if config.logging.save_best_only:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                    'val_loss': best_val_loss,
                    'config': config
                }, os.path.join(config.logging.checkpoint_dir, 'best_model.pth'))
        
        # Save checkpoint
        if (epoch + 1) % config.logging.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_loss': epoch_val_metrics['loss'],
                'config': config
            }, os.path.join(config.logging.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        
        # Early stopping
        if early_stopping(epoch_val_metrics['loss'], model):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Training completed
    training_time = time.time() - start_time
    print(f"\nTraining completed in {training_time/3600:.2f} hours")
    print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch+1}")
    
    # Final evaluation and visualization
    print("\nGenerating visualizations...")
    
    # Visualize reconstruction results
    vis_path = os.path.join(config.logging.log_dir, 'mae_reconstruction.png')
    visualize_mae_results(model, val_loader, device, vis_path)
    print(f"Reconstruction visualization saved to: {vis_path}")
    
    # Plot training curves
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
            'val_loss': val_metrics['loss'][-1],
            'config': config
        }, os.path.join(config.logging.checkpoint_dir, 'final_model.pth'))
    
    writer.close()
    print("MAE training completed successfully!")


if __name__ == "__main__":
    main()