"""
Demo script for MAE evaluation
Tests the evaluation functionality without requiring a trained model
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vision_mamba.models import create_vision_mamba_mae
from vision_mamba.config import get_config


def demo_mae_evaluation():
    """Demo MAE evaluation with a randomly initialized model"""
    
    print("=== Vision Mamba MAE Evaluation Demo ===\n")
    
    # Load configuration
    config = get_config('cifar10_tiny_mae')
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model (randomly initialized)
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
    
    # Create dummy data
    batch_size = 8
    img_size = config.model.img_size
    images = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    print(f"\nTesting with dummy data:")
    print(f"  Batch size: {batch_size}")
    print(f"  Image size: {img_size}x{img_size}")
    
    # Test forward pass
    print("\n1. Testing forward pass...")
    with torch.no_grad():
        loss, pred, mask = model(images)
        print(f"   Reconstruction loss: {loss.item():.6f}")
        print(f"   Prediction shape: {pred.shape}")
        print(f"   Mask shape: {mask.shape}")
        print(f"   Actual mask ratio: {mask.mean().item():.3f}")
    
    # Test visualization
    print("\n2. Testing reconstruction visualization...")
    with torch.no_grad():
        vis_results = model.visualize_reconstruction(images)
        
        original = vis_results['original']
        masked = vis_results['masked']
        reconstructed = vis_results['reconstructed']
        mask_vis = vis_results['mask']
        
        print(f"   Original shape: {original.shape}")
        print(f"   Masked shape: {masked.shape}")
        print(f"   Reconstructed shape: {reconstructed.shape}")
        print(f"   Mask shape: {mask_vis.shape}")
    
    # Test different mask ratios
    print("\n3. Testing different mask ratios...")
    mask_ratios = [0.25, 0.5, 0.75, 0.9]
    
    for mask_ratio in mask_ratios:
        with torch.no_grad():
            loss, pred, mask = model(images, mask_ratio=mask_ratio)
            actual_ratio = mask.mean().item()
            print(f"   Mask ratio {mask_ratio:.0%}: Loss = {loss.item():.6f}, Actual = {actual_ratio:.3f}")
    
    # Create a simple visualization
    print("\n4. Creating visualization...")
    
    # Use a smaller subset for visualization
    vis_images = images[:4]
    
    with torch.no_grad():
        vis_results = model.visualize_reconstruction(vis_images, mask_ratio=0.75)
    
    # Convert to CPU and numpy
    original = vis_results['original'].cpu().numpy()
    masked = vis_results['masked'].cpu().numpy()
    reconstructed = vis_results['reconstructed'].cpu().numpy()
    mask_tensor = vis_results['mask'].cpu()
    
    # Create figure
    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    
    patch_size = model.patch_size
    num_patches_per_side = img_size // patch_size
    
    for i in range(4):
        # Original
        img_orig = np.transpose(original[i], (1, 2, 0))
        img_orig = np.clip(img_orig, 0, 1)
        axes[0, i].imshow(img_orig)
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Masked
        img_masked = np.transpose(masked[i], (1, 2, 0))
        img_masked = np.clip(img_masked, 0, 1)
        axes[1, i].imshow(img_masked)
        axes[1, i].set_title('Masked')
        axes[1, i].axis('off')
        
        # Reconstructed
        img_recon = np.transpose(reconstructed[i], (1, 2, 0))
        img_recon = np.clip(img_recon, 0, 1)
        axes[2, i].imshow(img_recon)
        axes[2, i].set_title('Reconstructed')
        axes[2, i].axis('off')
        
        # Mask
        mask_2d = mask_tensor[i].reshape(num_patches_per_side, num_patches_per_side)
        mask_img = mask_2d.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
        axes[3, i].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
        axes[3, i].set_title('Mask')
        axes[3, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('mae_evaluation_demo.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   Visualization saved to 'mae_evaluation_demo.png'")
    
    # Test metric computation (basic)
    print("\n5. Testing basic metric computation...")
    
    def compute_basic_metrics(original, reconstructed):
        """Compute basic reconstruction metrics"""
        original_np = original.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        
        # MSE
        mse = np.mean((original_np - reconstructed_np) ** 2)
        
        # MAE
        mae = np.mean(np.abs(original_np - reconstructed_np))
        
        # Simple PSNR (without scikit-image)
        if mse > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse))
        else:
            psnr = float('inf')
        
        return {'mse': mse, 'mae': mae, 'psnr': psnr}
    
    with torch.no_grad():
        vis_results = model.visualize_reconstruction(vis_images)
        metrics = compute_basic_metrics(vis_results['original'], vis_results['reconstructed'])
        
        print(f"   MSE: {metrics['mse']:.6f}")
        print(f"   MAE: {metrics['mae']:.6f}")
        print(f"   PSNR: {metrics['psnr']:.2f} dB")
    
    print("\n=== Demo completed successfully! ===")
    print("\nTo evaluate a trained model, use:")
    print("python evaluate_mae.py --checkpoint /path/to/model.pth --config cifar10_tiny_mae")


if __name__ == "__main__":
    demo_mae_evaluation()