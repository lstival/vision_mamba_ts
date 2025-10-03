"""
Demo script to test Vision Mamba training setup
Run a quick training test with reduced epochs to verify everything works
"""
import os
import sys
import torch
from config import get_config
from train_vision_mamba import (
    create_data_loaders, create_model, create_optimizer, 
    train_epoch, validate_epoch, set_seed, get_device
)
import torch.nn as nn
from tqdm import tqdm


def quick_test():
    """Run a quick test of the training setup"""
    print("Running Vision Mamba training demo...")
    
    # Get a minimal config for testing
    config = get_config("imagenet")
    config.training.epochs = 2  # Just 2 epochs for demo
    config.data.batch_size = 32  # Smaller batch size
    config.data.num_workers = 0  # No multiprocessing for demo
    config.logging.log_interval = 5
    
    # Set seed
    set_seed(config.seed)
    
    # Get device
    device = get_device(config.device)
    print(f"Using device: {device}")
    
    try:
        # Create data loaders
        print("Creating data loaders...")
        train_loader, val_loader, class_names = create_data_loaders(config)
        print(f"Train samples: {len(train_loader.dataset)}")
        print(f"Val samples: {len(val_loader.dataset)}")
        print(f"Classes: {class_names}")
        
        # Create model
        print("Creating model...")
        model = create_model(config)
        model = model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total parameters: {total_params:,}")
        
        # Test forward pass
        print("Testing forward pass...")
        sample_batch = next(iter(train_loader))
        x, y = sample_batch
        x, y = x.to(device), y.to(device)
        
        with torch.no_grad():
            output = model(x)
            print(f"Input shape: {x.shape}")
            print(f"Output shape: {output.shape}")
            print(f"Expected classes: {config.data.num_classes}")
        
        # Create optimizer and loss
        optimizer = create_optimizer(model, config)
        criterion = nn.CrossEntropyLoss()
        
        # Test training loop
        print("Testing training loop...")
        for epoch in range(config.training.epochs):
            print(f"\nEpoch {epoch+1}/{config.training.epochs}")
            
            # Train for one epoch
            train_metrics = train_epoch(
                model, train_loader, criterion, optimizer, device, config
            )
            
            # Validate
            val_metrics, _, _ = validate_epoch(
                model, val_loader, criterion, device
            )
            
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
            print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        
        print("\n✅ Demo completed successfully!")
        print("The training setup is working correctly.")
        print("You can now run the full training with: python train_vision_mamba.py")
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_model_variants():
    """Test different model configurations"""
    print("\nTesting different model variants...")
    
    configs = {
        "CIFAR-10": get_config("cifar10"),
        "CIFAR-100": get_config("cifar100"),
        "Fashion-MNIST": get_config("fashion_mnist")
    }
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for name, config in configs.items():
        try:
            print(f"\nTesting {name}...")
            model = create_model(config)
            model = model.to(device)
            
            # Create dummy input
            if config.model.in_channels == 1:
                dummy_input = torch.randn(2, 1, 32, 32).to(device)
            else:
                dummy_input = torch.randn(2, 3, 32, 32).to(device)
            
            with torch.no_grad():
                output = model(dummy_input)
                print(f"  Input: {dummy_input.shape}")
                print(f"  Output: {output.shape}")
                print(f"  Expected classes: {config.data.num_classes}")
                print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            print(f"  ✅ {name} model works correctly")
            
        except Exception as e:
            print(f"  ❌ {name} model failed: {str(e)}")


if __name__ == "__main__":
    print("Vision Mamba Training Setup Demo")
    print("=" * 50)
    
    # Test basic training setup
    success = quick_test()
    
    if success:
        # Test different model variants
        test_model_variants()
        
        print("\n" + "=" * 50)
        print("All tests completed!")
        print("\nTo start full training, run:")
        print("  python train_vision_mamba.py --config cifar10")
        print("  python train_vision_mamba.py --config cifar100")
        print("  python train_vision_mamba.py --config fashion_mnist")
        print("\nTo evaluate a trained model, run:")
        print("  python evaluate_model.py --checkpoint path/to/best_model.pth")
    else:
        print("\nPlease fix the issues above before running full training.")