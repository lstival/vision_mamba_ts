import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
import numpy as np


class PatchEmbedding(nn.Module):
    """
    Image to Patch Embedding for Vision Mamba
    Converts 2D image patches into 1D token sequences
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = embed_dim
        
        # Patch projection layer
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embedding
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})"
        
        # Create patches and flatten
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add position embedding
        x = x + self.pos_embed
        
        return x


class SSMLayer(nn.Module):
    """
    State Space Model Layer for Vision Mamba
    Implements the core SSM computation with learnable parameters
    """
    def __init__(self, d_model, d_state=16, dt_rank="auto", d_conv=4):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        
        if dt_rank == "auto":
            self.dt_rank = math.ceil(d_model / 16)
        else:
            self.dt_rank = dt_rank
            
        # SSM parameters
        self.A_log = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Input projections
        self.x_proj = nn.Linear(d_model, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)
        
        # Convolution for sequence modeling
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=d_conv,
            bias=True,
            padding=d_conv - 1,
            groups=d_model
        )
        
    def forward(self, x):
        # x: (B, L, d_model)
        B, L, d_model = x.shape
        
        # Apply 1D convolution along sequence dimension
        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :L].transpose(1, 2)
        
        # Project input to get dt, B, C
        x_dbl = self.x_proj(x_conv)  # (B, L, dt_rank + 2*d_state)
        
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))  # (B, L, d_model)
        
        # SSM computation
        A = -torch.exp(self.A_log.float())  # (d_model, d_state)
        
        # Discretize continuous parameters
        dA = torch.exp(torch.einsum("bld,dn->bldn", dt, A))  # (B, L, d_model, d_state)
        dB = torch.einsum("bld,bln->bldn", dt, B)  # (B, L, d_model, d_state)
        
        # State space computation
        h = torch.zeros((B.shape[0], d_model, self.d_state), dtype=x.dtype, device=x.device)
        hs = []
        
        for i in range(L):
            h = dA[:, i] * h + dB[:, i] * x_conv[:, i:i+1, :].transpose(1, 2)
            hs.append(torch.einsum("bmd,bd->bm", h, C[:, i]))
            
        y = torch.stack(hs, dim=1)  # (B, L, d_model)
        
        # Add skip connection
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0)
        
        return y


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism for enhanced feature interaction
    """
    def __init__(self, d_model, num_heads=8, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.d_k ** -0.5
        
    def forward(self, x, mask=None):
        B, L, d_model = x.shape
        
        # Generate Q, K, V
        Q = self.w_q(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, L, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention computation
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, L, d_model)
        
        # Output projection
        output = self.w_o(attn_output)
        
        return output


class VisionMambaBlock(nn.Module):
    """
    Vision Mamba Block combining SSM and Attention mechanisms
    """
    def __init__(self, d_model, d_state=16, d_conv=4, num_heads=8, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # SSM layer
        self.ssm = SSMLayer(d_model, d_state, d_conv=d_conv)
        
        # Attention layer
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        
        # MLP
        mlp_hidden_dim = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout)
        )
        
        # Gating mechanism to balance SSM and Attention
        self.gate = nn.Parameter(torch.tensor(0.5))
        
    def forward(self, x):
        # SSM branch with residual connection
        ssm_out = x + self.ssm(self.norm1(x))
        
        # Attention branch with residual connection
        attn_out = x + self.attention(self.norm2(x))
        
        # Combine SSM and Attention with learnable gating
        combined = self.gate * ssm_out + (1 - self.gate) * attn_out
        
        # MLP with residual connection
        output = combined + self.mlp(self.norm3(combined))
        
        return output


class VisionMamba(nn.Module):
    """
    Vision Mamba: A hybrid architecture combining State Space Models and Attention
    for image processing and embedding extraction
    """
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=768,
        depth=12,
        d_state=16,
        d_conv=4,
        num_heads=12,
        mlp_ratio=4.0,
        dropout=0.1,
        num_classes=1000,
        use_cls_token=True
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes
        self.use_cls_token = use_cls_token
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.num_patches
        
        # CLS token (optional)
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
            self.num_tokens = num_patches + 1
        else:
            self.num_tokens = num_patches
            
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Vision Mamba blocks
        self.blocks = nn.ModuleList([
            VisionMambaBlock(
                d_model=embed_dim,
                d_state=d_state,
                d_conv=d_conv,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            ) for _ in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward_features(self, x):
        """Extract features without classification head"""
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token if used
        if self.use_cls_token:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
        x = self.dropout(x)
        
        # Apply Vision Mamba blocks
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        
        return x
        
    def forward(self, x, return_features=False):
        """
        Forward pass
        Args:
            x: Input images (B, C, H, W)
            return_features: If True, return both features and logits
        """
        features = self.forward_features(x)
        
        # Global average pooling or use CLS token
        if self.use_cls_token:
            # Use CLS token for classification
            cls_features = features[:, 0]  # (B, embed_dim)
        else:
            # Global average pooling
            cls_features = features.mean(dim=1)  # (B, embed_dim)
            
        # Classification
        logits = self.head(cls_features)
        
        if return_features:
            return {
                'logits': logits,
                'features': cls_features,
                'all_features': features
            }
        else:
            return logits
            
    def get_embeddings(self, x):
        """Extract embeddings for downstream tasks"""
        with torch.no_grad():
            features = self.forward_features(x)
            if self.use_cls_token:
                embeddings = features[:, 0]  # CLS token embeddings
            else:
                embeddings = features.mean(dim=1)  # Average pooling
            return embeddings


def create_vision_mamba_tiny(**kwargs):
    """Create a tiny Vision Mamba model with about half the parameters."""
    for k in ['embed_dim', 'depth', 'num_heads']:
        kwargs.pop(k, None)
    model = VisionMamba(
        embed_dim=96,    # Halved from 192
        depth=6,         # Halved from 12
        num_heads=2,     # Fewer heads for smaller model
        **kwargs
    )
    return model


def create_vision_mamba_small(**kwargs):
    """Create a small Vision Mamba model"""
    # Remove keys if present to avoid multiple values for the same argument
    for k in ['embed_dim', 'depth', 'num_heads']:
        kwargs.pop(k, None)
    model = VisionMamba(
        embed_dim=384,
        depth=12,
        num_heads=6,
        **kwargs
    )
    return model


def create_vision_mamba_base(**kwargs):
    """Create a base Vision Mamba model"""
    for k in ['embed_dim', 'depth', 'num_heads']:
        kwargs.pop(k, None)
    model = VisionMamba(
        embed_dim=768,
        depth=12,
        num_heads=12,
        **kwargs
    )
    return model


def create_vision_mamba_large(**kwargs):
    """Create a large Vision Mamba model"""
    for k in ['embed_dim', 'depth', 'num_heads']:
        kwargs.pop(k, None)
    model = VisionMamba(
        embed_dim=1024,
        depth=24,
        num_heads=16,
        **kwargs
    )
    return model


# Example usage and testing
if __name__ == "__main__":
    # Create model
    model = create_vision_mamba_small(
        img_size=224,
        patch_size=16,
        in_channels=3,
        num_classes=1000
    )
    
    # Test with random input
    x = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    
    # Forward pass with features
    output = model(x, return_features=True)
    print(f"Logits shape: {output['logits'].shape}")
    print(f"CLS features shape: {output['features'].shape}")
    print(f"All features shape: {output['all_features'].shape}")
    
    # Extract embeddings
    embeddings = model.get_embeddings(x)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
