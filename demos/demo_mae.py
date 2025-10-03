"""
Demo script for Vision Mamba MAE
Tests the MAE model and visualizes reconstruction results
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_mamba.models import create_vision_mamba_mae
from vision_mamba.config import get_config


def demo_mae_reconstruction():
    """Demo MAE reconstruction on CIFAR-10"""
    
    # Load configuration
    config = get_config('cifar10_tiny_mae')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    print("Creating Vision Mamba MAE model...")
    model = create_vision_mamba_mae(config)
    model = model.to(device)
    model.eval()
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"Model Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Encoder parameters: {encoder_params:,}")
    print(f"  Decoder parameters: {decoder_params:,}")
    print(f"  Decoder/Total ratio: {decoder_params/total_params:.3f}")
    print(f"  Mask ratio: {model.mask_ratio}")
    
    # Load CIFAR-10 data
    print("Loading CIFAR-10 data...")
    transform = transforms.Compose([
        transforms.Resize((config.model.img_size, config.model.img_size)),
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    
    # Get a few sample images
    num_samples = 8
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    images = torch.stack([dataset[i][0] for i in indices])
    images = images.to(device)
    
    print(f"Input shape: {images.shape}")
    
    # Test MAE reconstruction
    print("Testing MAE reconstruction...")
    with torch.no_grad():
        # Forward pass
        loss, pred, mask = model(images)
        print(f"Reconstruction loss: {loss.item():.6f}")
        print(f"Mask ratio: {mask.mean().item():.3f}")
        
        # Get visualization results
        vis_results = model.visualize_reconstruction(images)
        
    # Visualize results
    print("Creating visualization...")
    original = vis_results['original'].cpu()
    masked = vis_results['masked'].cpu()
    reconstructed = vis_results['reconstructed'].cpu()
    mask_vis = vis_results['mask'].cpu()
    
    # Create figure
    fig, axes = plt.subplots(4, num_samples, figsize=(num_samples * 2, 8))
    
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                   'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i in range(num_samples):
        # Get class name
        _, label = dataset[indices[i]]
        class_name = class_names[label]
        
        # Original image
        img_orig = original[i].permute(1, 2, 0)
        img_orig = torch.clamp(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f'Original\n({class_name})', fontsize=10)
        axes[0, i].axis('off')
        
        # Masked image
        img_masked = masked[i].permute(1, 2, 0)
        img_masked = torch.clamp(img_masked, 0, 1)
        axes[1, i].imshow(img_masked)
        axes[1, i].set_title('Masked', fontsize=10)
        axes[1, i].axis('off')
        
        # Mask visualization
        patch_size = model.patch_size
        img_size = config.model.img_size
        num_patches_per_side = img_size // patch_size
        
        mask_2d = mask_vis[i].reshape(num_patches_per_side, num_patches_per_side)
        mask_img = mask_2d.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
        
        axes[2, i].imshow(mask_img, cmap='gray')
        axes[2, i].set_title('Mask', fontsize=10)
        axes[2, i].axis('off')
        
        # Reconstructed image
        img_recon = reconstructed[i].permute(1, 2, 0)
        img_recon = torch.clamp(img_recon, 0, 1)
        axes[3, i].imshow(img_recon)
        axes[3, i].set_title('Reconstructed', fontsize=10)
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mae_demo_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Demo completed! Results saved to 'mae_demo_results.png'")
    
    # Test different mask ratios
    print("\nTesting different mask ratios...")
    mask_ratios = [0.25, 0.5, 0.75, 0.9]
    
    fig, axes = plt.subplots(len(mask_ratios), 3, figsize=(6, len(mask_ratios) * 2))
    
    # Use first image
    test_img = images[0:1]  # Keep batch dimension
    
    with torch.no_grad():
        for i, mask_ratio in enumerate(mask_ratios):
            vis_results = model.visualize_reconstruction(test_img, mask_ratio=mask_ratio)
            
            original = vis_results['original'][0].cpu().permute(1, 2, 0)
            masked = vis_results['masked'][0].cpu().permute(1, 2, 0)
            reconstructed = vis_results['reconstructed'][0].cpu().permute(1, 2, 0)
            
            # Clamp values
            original = torch.clamp(original, 0, 1)
            masked = torch.clamp(masked, 0, 1)
            reconstructed = torch.clamp(reconstructed, 0, 1)
            
            if len(mask_ratios) == 1:
                axes = [axes]  # Make it iterable for single row
            
            axes[i][0].imshow(original)
            axes[i][0].set_title(f'Original (mask={mask_ratio})')
            axes[i][0].axis('off')
            
            axes[i][1].imshow(masked)
            axes[i][1].set_title('Masked')
            axes[i][1].axis('off')
            
            axes[i][2].imshow(reconstructed)
            axes[i][2].set_title('Reconstructed')
            axes[i][2].axis('off')
    
    plt.tight_layout()
    plt.savefig('mae_mask_ratio_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Mask ratio comparison saved to 'mae_mask_ratio_comparison.png'")


def test_mae_training_step():
    """Test a single training step of MAE"""
    
    # Load configuration
    config = get_config('cifar10_tiny_mae')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = create_vision_mamba_mae(config)
    model = model.to(device)
    model.train()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    # Create dummy batch
    batch_size = 4
    images = torch.randn(batch_size, 3, config.model.img_size, config.model.img_size).to(device)
    
    print(f"Testing training step with batch size {batch_size}...")
    print(f"Input shape: {images.shape}")
    
    # Forward pass
    optimizer.zero_grad()
    loss, pred, mask = model(images)
    
    print(f"Loss: {loss.item():.6f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Actual mask ratio: {mask.mean().item():.3f}")
    
    # Backward pass
    loss.backward()
    optimizer.step()
    
    print("Training step completed successfully!")


if __name__ == "__main__":
    print("=== Vision Mamba MAE Demo ===\n")
    
    # Test training step first
    test_mae_training_step()
    print("\n" + "="*50 + "\n")
    
    # Demo reconstruction
    demo_mae_reconstruction()