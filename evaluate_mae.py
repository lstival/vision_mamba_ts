"""
Evaluation script for Vision Mamba Masked Autoencoder (MAE)
Provides comprehensive evaluation including reconstruction metrics and visualizations
"""

import os
import sys
import argparse
import json
import time
import numpy as np
from datetime import datetime
from tqdm import tqdm
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Try to import scikit-image metrics, provide fallback if not available
try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not found. PSNR and SSIM metrics will not be available.")
    print("Install with: pip install scikit-image")

# Import our modules
from vision_mamba_mae import create_vision_mamba_mae, VisionMambaMAE
from yaml_config import Config, get_config, load_config


class MAEEvaluator:
    """Comprehensive evaluator for MAE models"""
    
    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.results = {}
        
    def compute_reconstruction_metrics(self, original, reconstructed):
        """
        Compute various reconstruction metrics
        Args:
            original: Original images (B, C, H, W)
            reconstructed: Reconstructed images (B, C, H, W)
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Convert to numpy for metric computation
        original_np = original.cpu().numpy()
        reconstructed_np = reconstructed.cpu().numpy()
        
        # MSE Loss (per image, then averaged)
        mse_per_image = np.mean((original_np - reconstructed_np) ** 2, axis=(1, 2, 3))
        metrics['mse'] = np.mean(mse_per_image)
        metrics['mse_std'] = np.std(mse_per_image)
        
        # MAE Loss (per image, then averaged)
        mae_per_image = np.mean(np.abs(original_np - reconstructed_np), axis=(1, 2, 3))
        metrics['mae'] = np.mean(mae_per_image)
        metrics['mae_std'] = np.std(mae_per_image)
        
        # PSNR and SSIM (if scikit-image is available)
        if HAS_SKIMAGE:
            psnr_values = []
            ssim_values = []
            
            batch_size = original_np.shape[0]
            for i in range(batch_size):
                # Convert to proper format for PSNR/SSIM (H, W, C)
                if original_np.shape[1] == 3:  # RGB
                    orig_img = np.transpose(original_np[i], (1, 2, 0))
                    recon_img = np.transpose(reconstructed_np[i], (1, 2, 0))
                else:  # Grayscale
                    orig_img = original_np[i, 0]
                    recon_img = reconstructed_np[i, 0]
                
                # Ensure values are in [0, 1] range
                orig_img = np.clip(orig_img, 0, 1)
                recon_img = np.clip(recon_img, 0, 1)
                
                # Compute PSNR
                psnr_val = psnr(orig_img, recon_img, data_range=1.0)
                psnr_values.append(psnr_val)
                
                # Compute SSIM
                if len(orig_img.shape) == 3:  # RGB
                    ssim_val = ssim(orig_img, recon_img, data_range=1.0, channel_axis=2)
                else:  # Grayscale
                    ssim_val = ssim(orig_img, recon_img, data_range=1.0)
                ssim_values.append(ssim_val)
            
            metrics['psnr'] = np.mean(psnr_values)
            metrics['psnr_std'] = np.std(psnr_values)
            metrics['ssim'] = np.mean(ssim_values)
            metrics['ssim_std'] = np.std(ssim_values)
        else:
            # Fallback: compute simple PSNR without scikit-image
            mse_for_psnr = np.mean((original_np - reconstructed_np) ** 2)
            if mse_for_psnr > 0:
                metrics['psnr'] = 20 * np.log10(1.0 / np.sqrt(mse_for_psnr))
                metrics['psnr_std'] = 0.0
            else:
                metrics['psnr'] = float('inf')
                metrics['psnr_std'] = 0.0
            
            # Simple correlation-based similarity (approximation of SSIM)
            def simple_ssim_approx(x, y):
                # Flatten arrays
                x_flat = x.flatten()
                y_flat = y.flatten()
                # Compute correlation coefficient
                correlation = np.corrcoef(x_flat, y_flat)[0, 1]
                return correlation if not np.isnan(correlation) else 0.0
            
            ssim_approx_values = []
            batch_size = original_np.shape[0]
            for i in range(batch_size):
                ssim_approx = simple_ssim_approx(original_np[i], reconstructed_np[i])
                ssim_approx_values.append(ssim_approx)
            
            metrics['ssim'] = np.mean(ssim_approx_values)
            metrics['ssim_std'] = np.std(ssim_approx_values)
        
        return metrics
    
    def evaluate_dataset(self, data_loader, num_batches=None):
        """
        Evaluate model on entire dataset
        Args:
            data_loader: DataLoader for evaluation
            num_batches: Limit evaluation to N batches (None for full dataset)
        """
        self.model.eval()
        
        all_metrics = defaultdict(list)
        reconstruction_losses = []
        
        print("Evaluating reconstruction quality...")
        
        with torch.no_grad():
            pbar = tqdm(data_loader, desc="Evaluation")
            for batch_idx, (data, _) in enumerate(pbar):
                if num_batches is not None and batch_idx >= num_batches:
                    break
                    
                data = data.to(self.device, non_blocking=True)
                
                # Forward pass
                loss, pred, mask = self.model(data)
                reconstruction_losses.append(loss.item())
                
                # Get reconstruction visualization
                vis_results = self.model.visualize_reconstruction(data)
                original = vis_results['original']
                reconstructed = vis_results['reconstructed']
                
                # Compute metrics
                batch_metrics = self.compute_reconstruction_metrics(original, reconstructed)
                
                for key, value in batch_metrics.items():
                    all_metrics[key].append(value)
                
                # Update progress bar
                progress_info = {
                    'Loss': f'{loss.item():.6f}',
                    'MSE': f'{batch_metrics["mse"]:.6f}'
                }
                if 'psnr' in batch_metrics:
                    progress_info['PSNR'] = f'{batch_metrics["psnr"]:.2f}'
                if 'ssim' in batch_metrics:
                    progress_info['SSIM'] = f'{batch_metrics["ssim"]:.4f}'
                
                pbar.set_postfix(progress_info)
        
        # Aggregate metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            final_metrics[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        final_metrics['reconstruction_loss'] = {
            'mean': np.mean(reconstruction_losses),
            'std': np.std(reconstruction_losses)
        }
        
        self.results['dataset_metrics'] = final_metrics
        return final_metrics
    
    def create_reconstruction_gallery(self, data_loader, save_path, num_samples=16, mask_ratios=None):
        """
        Create a gallery of reconstruction examples
        Args:
            data_loader: DataLoader for getting images
            save_path: Path to save the gallery
            num_samples: Number of samples to show
            mask_ratios: List of mask ratios to test (default: [0.5, 0.75, 0.9])
        """
        if mask_ratios is None:
            mask_ratios = [0.5, 0.75, 0.9]
        
        self.model.eval()
        
        # Get sample images
        data_iter = iter(data_loader)
        images, labels = next(data_iter)
        images = images[:num_samples].to(self.device)
        
        # Get class names if available
        class_names = None
        if hasattr(data_loader.dataset, 'classes'):
            class_names = data_loader.dataset.classes
        elif self.config.data.dataset_name == "CIFAR10":
            class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                          'dog', 'frog', 'horse', 'ship', 'truck']
        elif self.config.data.dataset_name == "FashionMNIST":
            class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                          'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        
        with torch.no_grad():
            # Create figure
            n_rows = len(mask_ratios) * 3 + 1  # Original + (Masked, Reconstructed, Mask) for each ratio
            fig, axes = plt.subplots(n_rows, num_samples, figsize=(num_samples * 1.5, n_rows * 1.5))
            
            # Plot original images
            for i in range(num_samples):
                img = images[i].cpu().permute(1, 2, 0)
                if img.shape[2] == 1:  # Grayscale
                    axes[0, i].imshow(img.squeeze(), cmap='gray')
                else:
                    img = torch.clamp(img, 0, 1)
                    axes[0, i].imshow(img)
                
                # Add title with class name if available
                if class_names and i < len(labels):
                    title = f'Original\n{class_names[labels[i]]}'
                else:
                    title = 'Original'
                axes[0, i].set_title(title, fontsize=8)
                axes[0, i].axis('off')
            
            # Plot reconstructions for different mask ratios
            row_idx = 1
            for mask_ratio in mask_ratios:
                vis_results = self.model.visualize_reconstruction(images, mask_ratio=mask_ratio)
                
                original = vis_results['original'].cpu()
                masked = vis_results['masked'].cpu()
                reconstructed = vis_results['reconstructed'].cpu()
                mask = vis_results['mask'].cpu()
                
                # Masked images
                for i in range(num_samples):
                    img_masked = masked[i].permute(1, 2, 0)
                    if img_masked.shape[2] == 1:
                        axes[row_idx, i].imshow(img_masked.squeeze(), cmap='gray')
                    else:
                        img_masked = torch.clamp(img_masked, 0, 1)
                        axes[row_idx, i].imshow(img_masked)
                    
                    if i == 0:
                        axes[row_idx, i].set_ylabel(f'Masked\n({mask_ratio:.0%})', fontsize=8)
                    axes[row_idx, i].axis('off')
                
                row_idx += 1
                
                # Reconstructed images
                for i in range(num_samples):
                    img_recon = reconstructed[i].permute(1, 2, 0)
                    if img_recon.shape[2] == 1:
                        axes[row_idx, i].imshow(img_recon.squeeze(), cmap='gray')
                    else:
                        img_recon = torch.clamp(img_recon, 0, 1)
                        axes[row_idx, i].imshow(img_recon)
                    
                    if i == 0:
                        axes[row_idx, i].set_ylabel('Reconstructed', fontsize=8)
                    axes[row_idx, i].axis('off')
                
                row_idx += 1
                
                # Mask visualization
                patch_size = self.model.patch_size
                img_size = self.config.model.img_size
                num_patches_per_side = img_size // patch_size
                
                for i in range(num_samples):
                    mask_2d = mask[i].reshape(num_patches_per_side, num_patches_per_side)
                    mask_img = mask_2d.repeat_interleave(patch_size, dim=0).repeat_interleave(patch_size, dim=1)
                    
                    axes[row_idx, i].imshow(mask_img, cmap='gray', vmin=0, vmax=1)
                    if i == 0:
                        axes[row_idx, i].set_ylabel('Mask', fontsize=8)
                    axes[row_idx, i].axis('off')
                
                row_idx += 1
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"Reconstruction gallery saved to: {save_path}")
    
    def analyze_mask_ratio_performance(self, data_loader, mask_ratios, num_batches=5):
        """
        Analyze performance across different mask ratios
        Args:
            data_loader: DataLoader for evaluation
            mask_ratios: List of mask ratios to test
            num_batches: Number of batches to evaluate
        """
        self.model.eval()
        
        mask_ratio_results = {}
        
        print("Analyzing performance across mask ratios...")
        
        with torch.no_grad():
            # Get sample data
            data_batches = []
            data_iter = iter(data_loader)
            for _ in range(num_batches):
                try:
                    data, _ = next(data_iter)
                    data_batches.append(data.to(self.device))
                except StopIteration:
                    break
            
            for mask_ratio in tqdm(mask_ratios, desc="Mask ratios"):
                ratio_metrics = defaultdict(list)
                
                for data in data_batches:
                    # Forward pass with specific mask ratio
                    loss, pred, mask = self.model(data, mask_ratio=mask_ratio)
                    
                    # Get reconstructions
                    vis_results = self.model.visualize_reconstruction(data, mask_ratio=mask_ratio)
                    original = vis_results['original']
                    reconstructed = vis_results['reconstructed']
                    
                    # Compute metrics
                    batch_metrics = self.compute_reconstruction_metrics(original, reconstructed)
                    batch_metrics['reconstruction_loss'] = loss.item()
                    batch_metrics['actual_mask_ratio'] = mask.mean().item()
                    
                    for key, value in batch_metrics.items():
                        ratio_metrics[key].append(value)
                
                # Aggregate results for this mask ratio
                mask_ratio_results[mask_ratio] = {}
                for key, values in ratio_metrics.items():
                    mask_ratio_results[mask_ratio][key] = {
                        'mean': np.mean(values),
                        'std': np.std(values)
                    }
        
        self.results['mask_ratio_analysis'] = mask_ratio_results
        return mask_ratio_results
    
    def plot_mask_ratio_analysis(self, mask_ratio_results, save_path):
        """Plot mask ratio analysis results"""
        mask_ratios = list(mask_ratio_results.keys())
        
        # Metrics to plot (check availability)
        available_metrics = []
        metric_names = []
        
        # Always available
        available_metrics.extend(['mse', 'reconstruction_loss'])
        metric_names.extend(['MSE', 'Reconstruction Loss'])
        
        # Check if PSNR and SSIM are available
        first_result = next(iter(mask_ratio_results.values()))
        if 'psnr' in first_result:
            available_metrics.append('psnr')
            metric_names.append('PSNR (dB)')
        if 'ssim' in first_result:
            available_metrics.append('ssim')
            metric_names.append('SSIM')
        
        metrics_to_plot = available_metrics
        
        # Determine subplot layout based on available metrics
        n_metrics = len(metrics_to_plot)
        if n_metrics <= 2:
            fig, axes = plt.subplots(1, n_metrics, figsize=(6*n_metrics, 5))
            if n_metrics == 1:
                axes = [axes]
        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            axes = axes.flatten()
        
        for i, (metric, name) in enumerate(zip(metrics_to_plot, metric_names)):
            if i >= len(axes):
                break
                
            means = [mask_ratio_results[mr][metric]['mean'] for mr in mask_ratios]
            stds = [mask_ratio_results[mr][metric]['std'] for mr in mask_ratios]
            
            axes[i].errorbar(mask_ratios, means, yerr=stds, marker='o', capsize=5)
            axes[i].set_xlabel('Mask Ratio')
            axes[i].set_ylabel(name)
            axes[i].set_title(f'{name} vs Mask Ratio')
            axes[i].grid(True, alpha=0.3)
            axes[i].set_xlim(min(mask_ratios) - 0.05, max(mask_ratios) + 0.05)
        
        # Hide unused subplots
        if n_metrics < 4:
            for i in range(n_metrics, len(axes)):
                axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Mask ratio analysis plot saved to: {save_path}")
    
    def generate_detailed_report(self, save_path, training_timestamp=None):
        """Generate a detailed evaluation report"""
        # Use training timestamp if provided, otherwise use current time
        timestamp_iso = datetime.now().isoformat()
        if training_timestamp:
            # Convert training timestamp format to ISO format
            try:
                dt = datetime.strptime(training_timestamp, "%Y%m%d_%H%M")
                timestamp_iso = dt.isoformat()
            except ValueError:
                pass  # Use current time if parsing fails
        
        def convert_to_json_serializable(obj):
            """Convert numpy types and other non-serializable objects to JSON-safe types"""
            if isinstance(obj, dict):
                return {key: convert_to_json_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif hasattr(obj, 'item'):  # For single-element numpy arrays
                return obj.item()
            else:
                return obj
                
        report = {
            'evaluation_info': {
                'timestamp': timestamp_iso,
                'training_timestamp': training_timestamp,
                'model_config': {
                    'model_name': self.config.model.model_name,
                    'embed_dim': int(self.config.model.embed_dim),
                    'depth': int(self.config.model.depth),
                    'num_heads': int(self.config.model.num_heads),
                    'mask_ratio': float(getattr(self.config.model, 'mask_ratio', 0.75)),
                    'decoder_embed_dim': int(getattr(self.config.model, 'decoder_embed_dim', 192)),
                    'decoder_depth': int(getattr(self.config.model, 'decoder_depth', 16))
                },
                'dataset': self.config.data.dataset_name,
                'img_size': int(self.config.model.img_size)
            },
            'results': convert_to_json_serializable(self.results)
        }
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Detailed evaluation report saved to: {save_path}")


def create_data_loader(config, split='test'):
    """Create data loader for evaluation"""
    # Define transforms - same as training validation transforms
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
    
    transform = transforms.Compose(base_transforms)
    
    # Load dataset
    if config.data.dataset_name == "CIFAR10":
        dataset = torchvision.datasets.CIFAR10(
            root=config.data.data_dir, train=False, download=True, transform=transform
        )
    elif config.data.dataset_name == "CIFAR100":
        dataset = torchvision.datasets.CIFAR100(
            root=config.data.data_dir, train=False, download=True, transform=transform
        )
    elif config.data.dataset_name == "FashionMNIST":
        dataset = torchvision.datasets.FashionMNIST(
            root=config.data.data_dir, train=False, download=True, transform=transform
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.data.dataset_name}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=64,  # Use smaller batch size for evaluation
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return data_loader


def extract_timestamp_from_checkpoint_path(checkpoint_path):
    """Extract timestamp from checkpoint path"""
    import re
    
    # Look for timestamp pattern YYYYMMDD_HHMM in the path
    timestamp_pattern = r'(\d{8}_\d{4})'
    match = re.search(timestamp_pattern, checkpoint_path)
    
    if match:
        return match.group(1)
    else:
        # If no timestamp found in path, generate a new one
        print("Warning: Could not extract timestamp from checkpoint path. Generating new timestamp.")
        return datetime.now().strftime("%Y%m%d_%H%M")


def load_model_from_checkpoint(checkpoint_path, config, device):
    """Load MAE model from checkpoint"""
    print(f"Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Create model
    model = create_vision_mamba_mae(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from epoch {checkpoint['epoch'] + 1}")
    print(f"Checkpoint validation loss: {checkpoint.get('val_loss', 'N/A')}")
    
    return model


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate Vision Mamba MAE')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to the model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration name (will try to infer from checkpoint if not provided)')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Full path to configuration file')
    parser.add_argument('--output-dir', type=str, default='./mae_evaluation_results',
                       help='Base directory to save evaluation results (timestamp will be automatically appended)')
    parser.add_argument('--num-samples', type=int, default=16,
                       help='Number of samples for reconstruction gallery')
    parser.add_argument('--num-batches', type=int, default=None,
                       help='Limit evaluation to N batches (None for full dataset)')
    parser.add_argument('--mask-ratios', type=float, nargs='+', default=[0.25, 0.5, 0.75, 0.9],
                       help='Mask ratios to analyze')
    args = parser.parse_args()
    
    # Extract timestamp from checkpoint path to maintain consistency
    timestamp = extract_timestamp_from_checkpoint_path(args.checkpoint)
    output_dir = f"evaluation_results/{args.output_dir}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Using timestamp from checkpoint: {timestamp}")
    print(f"Evaluation results will be saved to: {output_dir}")
    
    # Get device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load configuration
    if args.config_path:
        config = load_config(args.config_path)
    elif args.config:
        config = get_config(args.config)
    else:
        # Try to get config from checkpoint
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if 'config' in checkpoint:
            config = checkpoint['config']
            print("Using configuration from checkpoint")
        else:
            raise ValueError("No configuration provided and none found in checkpoint")
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, config, device)
    
    # Create data loader
    print("Creating data loader...")
    data_loader = create_data_loader(config)
    print(f"Test dataset size: {len(data_loader.dataset)}")
    
    # Create evaluator
    evaluator = MAEEvaluator(model, device, config)
    
    # Run evaluation
    print("\n" + "="*50)
    print("VISION MAMBA MAE EVALUATION")
    print("="*50)
    
    # 1. Dataset-wide evaluation
    print("\n1. Evaluating reconstruction quality on test dataset...")
    dataset_metrics = evaluator.evaluate_dataset(data_loader, num_batches=args.num_batches)
    
    # Print summary metrics
    print("\nDataset Evaluation Results:")
    print("-" * 30)
    for metric, values in dataset_metrics.items():
        if isinstance(values, dict):
            print(f"{metric.upper()}: {values['mean']:.6f} ± {values['std']:.6f}")
        else:
            print(f"{metric.upper()}: {values:.6f}")
    
    # 2. Create reconstruction gallery
    print("\n2. Creating reconstruction gallery...")
    gallery_path = os.path.join(output_dir, 'reconstruction_gallery.png')
    evaluator.create_reconstruction_gallery(
        data_loader, gallery_path, 
        num_samples=args.num_samples, 
        mask_ratios=args.mask_ratios
    )
    
    # 3. Analyze mask ratio performance
    print("\n3. Analyzing performance across mask ratios...")
    mask_ratio_results = evaluator.analyze_mask_ratio_performance(
        data_loader, args.mask_ratios, num_batches=10
    )
    
    # Plot mask ratio analysis
    mask_analysis_path = os.path.join(output_dir, 'mask_ratio_analysis.png')
    evaluator.plot_mask_ratio_analysis(mask_ratio_results, mask_analysis_path)
    
    # Print mask ratio results
    print("\nMask Ratio Analysis Results:")
    print("-" * 40)
    for mask_ratio, metrics in mask_ratio_results.items():
        print(f"\nMask Ratio: {mask_ratio:.0%}")
        for metric, values in metrics.items():
            if metric in ['mse', 'mae', 'psnr', 'ssim', 'reconstruction_loss']:
                if metric == 'psnr' and values['mean'] == float('inf'):
                    print(f"  {metric.upper()}: inf (perfect reconstruction)")
                else:
                    print(f"  {metric.upper()}: {values['mean']:.6f} ± {values['std']:.6f}")
    
    # 4. Generate detailed report
    print("\n4. Generating detailed evaluation report...")
    report_path = os.path.join(output_dir, 'evaluation_report.json')
    evaluator.generate_detailed_report(report_path, training_timestamp=timestamp)
    
    # 5. Create summary visualization
    print("\n5. Creating summary visualization...")
    
    # Summary plot with key metrics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Reconstruction quality metrics (only available ones)
    available_metrics = ['mse', 'mae']
    available_metric_names = ['MSE', 'MAE']
    
    if 'psnr' in dataset_metrics:
        available_metrics.append('psnr')
        available_metric_names.append('PSNR (dB)')
    if 'ssim' in dataset_metrics:
        available_metrics.append('ssim')
        available_metric_names.append('SSIM')
    
    values = [dataset_metrics[m]['mean'] for m in available_metrics]
    errors = [dataset_metrics[m]['std'] for m in available_metrics]
    
    axes[0, 0].bar(available_metric_names, values, yerr=errors, capsize=5, alpha=0.7)
    axes[0, 0].set_title('Reconstruction Quality Metrics')
    axes[0, 0].set_ylabel('Value')
    
    # Plot 2: MSE vs Mask Ratio
    mask_ratios = list(mask_ratio_results.keys())
    mse_values = [mask_ratio_results[mr]['mse']['mean'] for mr in mask_ratios]
    mse_errors = [mask_ratio_results[mr]['mse']['std'] for mr in mask_ratios]
    
    axes[0, 1].errorbar(mask_ratios, mse_values, yerr=mse_errors, marker='o', capsize=5)
    axes[0, 1].set_title('MSE vs Mask Ratio')
    axes[0, 1].set_xlabel('Mask Ratio')
    axes[0, 1].set_ylabel('MSE')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: PSNR vs Mask Ratio (if available)
    if 'psnr' in next(iter(mask_ratio_results.values())):
        psnr_values = [mask_ratio_results[mr]['psnr']['mean'] for mr in mask_ratios]
        psnr_errors = [mask_ratio_results[mr]['psnr']['std'] for mr in mask_ratios]
        
        axes[1, 0].errorbar(mask_ratios, psnr_values, yerr=psnr_errors, marker='s', capsize=5, color='orange')
        axes[1, 0].set_title('PSNR vs Mask Ratio')
        axes[1, 0].set_xlabel('Mask Ratio')
        axes[1, 0].set_ylabel('PSNR (dB)')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'PSNR not available\n(install scikit-image)', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('PSNR vs Mask Ratio')
    
    # Plot 4: SSIM vs Mask Ratio (if available)
    if 'ssim' in next(iter(mask_ratio_results.values())):
        ssim_values = [mask_ratio_results[mr]['ssim']['mean'] for mr in mask_ratios]
        ssim_errors = [mask_ratio_results[mr]['ssim']['std'] for mr in mask_ratios]
        
        axes[1, 1].errorbar(mask_ratios, ssim_values, yerr=ssim_errors, marker='^', capsize=5, color='green')
        axes[1, 1].set_title('SSIM vs Mask Ratio')
        axes[1, 1].set_xlabel('Mask Ratio')
        axes[1, 1].set_ylabel('SSIM')
        axes[1, 1].grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, 'SSIM not available\n(install scikit-image)', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('SSIM vs Mask Ratio')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, 'evaluation_summary.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Summary visualization saved to: {summary_path}")
    
    # Final summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETED")
    print("="*50)
    print(f"Results saved to: {output_dir}")
    print("\nGenerated files:")
    print(f"  - reconstruction_gallery.png: Visual examples of reconstructions")
    print(f"  - mask_ratio_analysis.png: Performance across mask ratios")
    print(f"  - evaluation_summary.png: Summary of key metrics")
    print(f"  - evaluation_report.json: Detailed numerical results")
    
    # Best performing mask ratio
    best_mask_ratio = min(mask_ratio_results.keys(), 
                         key=lambda x: mask_ratio_results[x]['mse']['mean'])
    worst_mask_ratio = max(mask_ratio_results.keys(), 
                          key=lambda x: mask_ratio_results[x]['mse']['mean'])
    
    print(f"\nKey Findings:")
    print(f"  - Best performing mask ratio: {best_mask_ratio:.0%} (lowest MSE)")
    print(f"  - Most challenging mask ratio: {worst_mask_ratio:.0%} (highest MSE)")
    
    if 'psnr' in dataset_metrics:
        print(f"  - Overall PSNR: {dataset_metrics['psnr']['mean']:.2f} ± {dataset_metrics['psnr']['std']:.2f} dB")
    else:
        print(f"  - PSNR: Not available (install scikit-image for advanced metrics)")
        
    if 'ssim' in dataset_metrics:
        print(f"  - Overall SSIM: {dataset_metrics['ssim']['mean']:.4f} ± {dataset_metrics['ssim']['std']:.4f}")
    else:
        print(f"  - SSIM: Not available (install scikit-image for advanced metrics)")
        
    print(f"  - Overall MSE: {dataset_metrics['mse']['mean']:.6f} ± {dataset_metrics['mse']['std']:.6f}")
    print(f"  - Overall MAE: {dataset_metrics['mae']['mean']:.6f} ± {dataset_metrics['mae']['std']:.6f}")


if __name__ == "__main__":
    main()