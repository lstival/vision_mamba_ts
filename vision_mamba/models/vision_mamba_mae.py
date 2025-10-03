"""
Vision Mamba Masked Autoencoder (MAE)
Implements a masked autoencoder using Vision Mamba as the encoder
with a lightweight decoder for patch reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import random
from .vision_mamba import VisionMamba, PatchEmbedding, create_vision_mamba_tiny, create_vision_mamba_small, create_vision_mamba_base


class MAEDecoder(nn.Module):
    """
    Lightweight decoder for MAE reconstruction with skip connections
    """
    def __init__(
        self,
        embed_dim=768,
        decoder_embed_dim=192,
        decoder_depth=16,
        decoder_num_heads=28,
        mlp_ratio=4.0,
        patch_size=16,
        in_channels=3,
        dropout=0.1,
        encoder_depth=16
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.decoder_embed_dim = decoder_embed_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.encoder_depth = encoder_depth
        self.decoder_depth = decoder_depth
        
        # Projection to decoder dimension
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        
        # Skip connection projections from encoder features to decoder dimension
        # We'll use features from multiple encoder layers
        self.skip_projections = nn.ModuleList()
        skip_layers = self._get_skip_layer_indices()
        for _ in skip_layers:
            self.skip_projections.append(
                nn.Linear(embed_dim, decoder_embed_dim, bias=True)
            )
        self.skip_layers = skip_layers
        
        # Mask token - learnable parameter for masked patches
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Decoder blocks (simple transformer blocks)
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
                dim=decoder_embed_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(decoder_depth)
        ])
        
        # Final layer norm
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        # Decoder head - reconstruct patches
        # Each patch is patch_size x patch_size x in_channels
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, 
            patch_size ** 2 * in_channels, 
            bias=True
        )
        
        self._init_weights()
    
    def _get_skip_layer_indices(self):
        """
        Determine which encoder layers to use for skip connections
        """
        # Use encoder layers at 1/4, 1/2, and 3/4 depth
        indices = []
        if self.encoder_depth >= 4:
            indices.append(self.encoder_depth // 4)
        if self.encoder_depth >= 8:
            indices.append(self.encoder_depth // 2)
        if self.encoder_depth >= 12:
            indices.append(3 * self.encoder_depth // 4)
        return indices
        
    def _init_weights(self):
        # Initialize mask token
        torch.nn.init.normal_(self.mask_token, std=0.02)
        
        # Initialize linear layers
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
    def forward(self, x, ids_restore, skip_features=None):
        """
        Forward pass of decoder
        Args:
            x: encoded features of visible patches (B, L_visible, embed_dim)
            ids_restore: indices to restore the original order (B, L_total)
        Returns:
            reconstructed patches (B, L_total, patch_size**2 * in_channels)
        """
        # Project to decoder dimension
        x = self.decoder_embed(x)  # (B, L_visible, decoder_embed_dim)
        
        # Append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )  # (B, L_masked, decoder_embed_dim)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, L_total, decoder_embed_dim)
        
        # Unshuffle to restore original order
        x_full = torch.gather(
            x_full, dim=1, 
            index=ids_restore.unsqueeze(-1).repeat(1, 1, x_full.shape[2])
        )  # (B, L_total, decoder_embed_dim)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            x_full = block(x_full)
            
        x_full = self.decoder_norm(x_full)
        
        # Predict pixel values for each patch
        x_rec = self.decoder_pred(x_full)  # (B, L_total, patch_size**2 * in_channels)
        
        return x_rec


class DecoderBlock(nn.Module):
    """
    Simple transformer decoder block
    """
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        
        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        
        return x


class VisionMambaMAE(nn.Module):
    """
    Vision Mamba Masked Autoencoder
    Uses Vision Mamba as encoder and a lightweight transformer as decoder
    """
    def __init__(
        self,
        encoder_config,
        mask_ratio=0.75,
        decoder_embed_dim=96,
        decoder_depth=6,
        decoder_num_heads=2,
        mlp_ratio=4.0,
        dropout=0.1
    ):
        super().__init__()
        self.mask_ratio = mask_ratio
        
        # Create encoder (Vision Mamba without classification head)
        if encoder_config['model_name'] == 'vision_mamba_tiny':
            self.encoder = create_vision_mamba_tiny(
                img_size=encoder_config['img_size'],
                patch_size=encoder_config['patch_size'],
                in_channels=encoder_config['in_channels'],
                embed_dim=encoder_config['embed_dim'],
                depth=encoder_config['depth'],
                d_state=encoder_config['d_state'],
                d_conv=encoder_config['d_conv'],
                num_heads=encoder_config['num_heads'],
                mlp_ratio=encoder_config['mlp_ratio'],
                dropout=encoder_config['dropout'],
                num_classes=0,  # No classification head
                use_cls_token=False  # Don't use CLS token for MAE
            )
        elif encoder_config['model_name'] == 'vision_mamba_small':
            self.encoder = create_vision_mamba_small(
                img_size=encoder_config['img_size'],
                patch_size=encoder_config['patch_size'],
                in_channels=encoder_config['in_channels'],
                num_classes=0,
                use_cls_token=False
            )
        elif encoder_config['model_name'] == 'vision_mamba_base':
            self.encoder = create_vision_mamba_base(
                img_size=encoder_config['img_size'],
                patch_size=encoder_config['patch_size'],
                in_channels=encoder_config['in_channels'],
                num_classes=0,
                use_cls_token=False
            )
        else:
            raise ValueError(f"Unknown encoder model: {encoder_config['model_name']}")
        
        # Get encoder embed_dim for decoder
        encoder_embed_dim = self.encoder.embed_dim
        
        # Create decoder
        self.decoder = MAEDecoder(
            embed_dim=encoder_embed_dim,
            decoder_embed_dim=decoder_embed_dim,
            decoder_depth=decoder_depth,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=mlp_ratio,
            patch_size=encoder_config['patch_size'],
            in_channels=encoder_config['in_channels'],
            dropout=dropout,
            encoder_depth=encoder_config['depth']
        )
        
        self.patch_size = encoder_config['patch_size']
        self.in_channels = encoder_config['in_channels']
        
    def patchify(self, imgs):
        """
        Convert images to patches
        Args:
            imgs: (B, C, H, W)
        Returns:
            patches: (B, L, patch_size**2 * C)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
        
        h = w = imgs.shape[2] // p
        x = imgs.reshape(imgs.shape[0], self.in_channels, h, p, w, p)
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(imgs.shape[0], h * w, p**2 * self.in_channels)
        return x
    
    def unpatchify(self, x):
        """
        Convert patches back to images
        Args:
            x: (B, L, patch_size**2 * C)
        Returns:
            imgs: (B, C, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(x.shape[0], h, w, p, p, self.in_channels)
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(x.shape[0], self.in_channels, h * p, h * p)
        return imgs
    
    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling
        Args:
            x: sequence of patches (B, L, D)
            mask_ratio: ratio of patches to mask
        Returns:
            x_masked: visible patches (B, L_visible, D)
            mask: binary mask (B, L), 0 is keep, 1 is remove
            ids_restore: indices to restore original order (B, L)
        """
        B, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(B, L, device=x.device)  # noise in [0, 1]
        
        # Sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L], device=x.device)
        mask[:, :len_keep] = 0
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_masked, mask, ids_restore
    
    def forward_encoder(self, x, mask_ratio):
        """
        Forward pass through encoder with masking and skip feature collection
        Args:
            x: input images (B, C, H, W)
            mask_ratio: ratio of patches to mask
        Returns:
            x_encoded: encoded visible patches (B, L_visible, embed_dim)
            mask: binary mask (B, L)
            ids_restore: indices to restore order (B, L)
            skip_features: list of intermediate encoder features for skip connections
        """
        # Patch embedding (without position embedding added yet)
        x = self.encoder.patch_embed.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.encoder.patch_embed.pos_embed
        
        # Masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # Apply dropout
        x = self.encoder.dropout(x)
        
        # Collect skip features from specific encoder layers
        skip_features = []
        skip_layer_indices = self.decoder.skip_layers
        
        # Apply encoder blocks and collect skip features
        for i, block in enumerate(self.encoder.blocks):
            x = block(x)
            
            # Collect features from specified layers for skip connections
            if i in skip_layer_indices:
                skip_features.append(x.clone())
            
        x = self.encoder.norm(x)
        
        return x, mask, ids_restore, skip_features
    
    def forward_decoder(self, x, ids_restore, skip_features=None):
        """
        Forward pass through decoder with skip connections
        """
        x = self.decoder(x, ids_restore, skip_features)
        return x
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss
        Args:
            imgs: original images (B, C, H, W)
            pred: predicted patches (B, L, patch_size**2 * C)
            mask: binary mask (B, L), 0 is keep, 1 is remove
        """
        target = self.patchify(imgs)  # (B, L, patch_size**2 * C)
        
        # Normalize target (per patch)
        if self.decoder.in_channels == 3:  # RGB images
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # (B, L), mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, imgs, mask_ratio=None):
        """
        Forward pass for training with skip connections
        Args:
            imgs: input images (B, C, H, W)
            mask_ratio: masking ratio (default uses self.mask_ratio)
        Returns:
            loss: reconstruction loss
            pred: predicted patches
            mask: binary mask
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        latent, mask, ids_restore, skip_features = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore, skip_features)
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def visualize_reconstruction(self, imgs, mask_ratio=None):
        """
        Visualize reconstruction for evaluation with skip connections
        Returns original, masked, and reconstructed images
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
            
        with torch.no_grad():
            latent, mask, ids_restore, skip_features = self.forward_encoder(imgs, mask_ratio)
            pred = self.forward_decoder(latent, ids_restore, skip_features)
            
            # Reconstruct images
            reconstructed = self.unpatchify(pred)
            
            # Create masked images for visualization
            mask_patches = mask.unsqueeze(-1).repeat(1, 1, self.patch_size**2 * self.in_channels)
            masked_patches = self.patchify(imgs) * (1 - mask_patches)
            masked_imgs = self.unpatchify(masked_patches)
            
            return {
                'original': imgs,
                'masked': masked_imgs,
                'reconstructed': reconstructed,
                'mask': mask
            }


def create_vision_mamba_mae(config):
    """
    Create Vision Mamba MAE model from config
    """
    encoder_config = {
        'model_name': config.model.model_name,
        'img_size': config.model.img_size,
        'patch_size': config.model.patch_size,
        'in_channels': config.model.in_channels,
        'embed_dim': config.model.embed_dim,
        'depth': config.model.depth,
        'd_state': config.model.d_state,
        'd_conv': config.model.d_conv,
        'num_heads': config.model.num_heads,
        'mlp_ratio': config.model.mlp_ratio,
        'dropout': config.model.dropout,
    }
    
    model = VisionMambaMAE(
        encoder_config=encoder_config,
        mask_ratio=getattr(config.model, 'mask_ratio', 0.75),
        decoder_embed_dim=getattr(config.model, 'decoder_embed_dim', 512),
        decoder_depth=getattr(config.model, 'decoder_depth', 8),
        decoder_num_heads=getattr(config.model, 'decoder_num_heads', 8),
        mlp_ratio=config.model.mlp_ratio,
        dropout=config.model.dropout
    )
    
    return model


# Example usage
if __name__ == "__main__":
    # Test the MAE model
    from types import SimpleNamespace
    
    # Create a mock config
    config = SimpleNamespace()
    config.model = SimpleNamespace()
    config.model.model_name = 'vision_mamba_tiny'
    config.model.img_size = 224
    config.model.patch_size = 16
    config.model.in_channels = 3
    config.model.embed_dim = 192
    config.model.depth = 12
    config.model.d_state = 16
    config.model.d_conv = 4
    config.model.num_heads = 3
    config.model.mlp_ratio = 4.0
    config.model.dropout = 0.1
    config.model.mask_ratio = 0.75
    config.model.decoder_embed_dim = 192
    config.model.decoder_depth = 12
    config.model.decoder_num_heads = 2
    
    # Create model
    model = create_vision_mamba_mae(config)
    
    # Test with random input
    x = torch.randn(2, 3, 224, 224)
    
    # Forward pass
    loss, pred, mask = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Reconstruction loss: {loss.item():.4f}")
    print(f"Prediction shape: {pred.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask ratio: {mask.mean().item():.3f}")
    
    # Test visualization
    vis_results = model.visualize_reconstruction(x)
    print(f"Original shape: {vis_results['original'].shape}")
    print(f"Masked shape: {vis_results['masked'].shape}")
    print(f"Reconstructed shape: {vis_results['reconstructed'].shape}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    decoder_params = sum(p.numel() for p in model.decoder.parameters())
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Encoder parameters: {encoder_params:,}")
    print(f"Decoder parameters: {decoder_params:,}")
    print(f"Decoder/Total ratio: {decoder_params/total_params:.3f}")