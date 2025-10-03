"""
Resume training script for Vision Mamba model
Loads a checkpoint (model, optimizer, scheduler, gradients if available) and continues training
Allows switching to a different dataset/configuration
"""
import os
import sys
import time
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import classification_report

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_mamba.models import create_vision_mamba_small, create_vision_mamba_base, create_vision_mamba_tiny
from vision_mamba.config import get_config, Config, list_available_configs


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_config):
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_config)


def create_data_loaders(config: Config):
    # ...existing code from train_vision_mamba.py...
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
    val_transforms = [
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
        normalize
    ]
    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)
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
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset_name}")
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


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Resume Vision Mamba Training')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to checkpoint to resume from')
    parser.add_argument('--config', type=str, default='cifar10_tiny', help='New configuration name (without .yaml extension)')
    parser.add_argument('--list-configs', action='store_true', help='List available configurations and exit')
    args = parser.parse_args()

    # Handle list configs
    if args.list_configs:
        print("Available configurations:")
        configs = list_available_configs()
        for config_name in configs:
            print(f"  - {config_name}")
        return

    # Load new config
    config = get_config(args.config)
    set_seed(config.seed)
    device = get_device(config.device)
    print(f"Using device: {device}")

    # Create data loaders for new dataset
    print("Creating data loaders for new dataset...")
    train_loader, val_loader, class_names = create_data_loaders(config)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Create model for new config
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)

    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, len(train_loader))

    # Create loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=config.training.label_smoothing)

    # AMP scaler
    scaler = torch.cuda.amp.GradScaler() if config.training.use_amp and device.type == 'cuda' else None

    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint} ...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if 'scheduler_state_dict' in checkpoint and scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    if scaler is not None and 'scaler' in checkpoint:
        scaler.load_state_dict(checkpoint['scaler'])
    start_epoch = checkpoint.get('epoch', 0) + 1
    print(f"Checkpoint loaded. Last epoch: {start_epoch-1}")
    print(f"Checkpoint info: {list(checkpoint.keys())}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy at checkpoint: {checkpoint['val_acc']}")
    if 'classification_report' in checkpoint:
        print(f"Classification report at checkpoint:\n{checkpoint['classification_report']}")

    # Training loop
    writer = SummaryWriter(config.logging.tensorboard_dir)
    print(f"Resuming training for {config.training.epochs - start_epoch} more epochs...")
    for epoch in range(start_epoch, config.training.epochs):
        print(f"\nEpoch {epoch+1}/{config.training.epochs}")
        model.train()
        running_loss = 0.0
        running_correct = 0
        total_samples = 0
        pbar = tqdm(train_loader, desc="Training")
        for data, targets in pbar:
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
            predictions = outputs.argmax(dim=1)
            running_loss += loss.item() * data.size(0)
            running_correct += (predictions == targets).sum().item()
            total_samples += data.size(0)
        train_loss = running_loss / total_samples
        train_acc = running_correct / total_samples
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_val_preds = []
        all_val_targets = []
        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
                loss = criterion(outputs, targets)
                preds = outputs.argmax(dim=1)
                val_loss += loss.item() * data.size(0)
                val_correct += (preds == targets).sum().item()
                val_total += data.size(0)
                all_val_preds.extend(preds.cpu().numpy())
                all_val_targets.extend(targets.cpu().numpy())
        val_loss /= val_total
        val_acc = val_correct / val_total
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'val_acc': val_acc,
            'config': config,
            'scaler': scaler.state_dict() if scaler is not None else None,
            'classification_report': classification_report(all_val_targets, all_val_preds, output_dict=True)
        }, os.path.join(config.logging.checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
        if scheduler is not None:
            if config.training.scheduler.lower() == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()
    writer.close()
    print("Training resumed and completed!")


if __name__ == "__main__":
    main()
