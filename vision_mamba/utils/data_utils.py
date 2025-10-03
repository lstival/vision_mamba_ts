"""
Data loading utilities for training and evaluation
"""
import torch
import torchvision
from torch.utils.data import DataLoader, random_split
from torchvision import transforms


def create_data_loaders(config):
    """Create data loaders for training and validation"""
    
    # Define transforms
    if config.data.normalize:
        if config.data.dataset_name == "CIFAR10":
            normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        elif config.data.dataset_name == "CIFAR100":
            normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        elif config.data.dataset_name == "FashionMNIST":
            normalize = transforms.Normalize(
                mean=[0.2860],
                std=[0.3530]
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
    
    # Load datasets
    if config.data.dataset_name == "CIFAR10":
        train_dataset = torchvision.datasets.CIFAR10(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR10(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        
    elif config.data.dataset_name == "CIFAR100":
        train_dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        class_names = [f'class_{i}' for i in range(100)]
        
    elif config.data.dataset_name == "FashionMNIST":
        train_dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=True, download=True, transform=train_transform
        )
        val_dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=False, download=True, transform=val_transform
        )
        class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
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


def create_mae_data_loaders(config):
    """Create data loaders for MAE training"""
    
    # Define transforms - simpler for MAE since we're reconstructing original images
    base_transforms = [
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
    ]
    
    # Optional normalization
    if config.data.normalize:
        if config.data.dataset_name == "CIFAR10":
            normalize = transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        elif config.data.dataset_name == "CIFAR100":
            normalize = transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761]
            )
        elif config.data.dataset_name == "FashionMNIST":
            normalize = transforms.Normalize(
                mean=[0.2860],
                std=[0.3530]
            )
        else:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        
        base_transforms.append(normalize)
    
    # Light augmentation for training (optional)
    train_transforms = []
    if hasattr(config.data, 'use_augmentation') and config.data.use_augmentation:
        if config.data.dataset_name not in ["FashionMNIST"]:
            train_transforms.extend([
                transforms.RandomCrop(32, padding=2),
                transforms.RandomHorizontalFlip(p=0.3)
            ])
    
    train_transforms.extend(base_transforms)
    val_transforms = base_transforms.copy()
    
    train_transform = transforms.Compose(train_transforms)
    val_transform = transforms.Compose(val_transforms)
    
    # Load datasets
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