"""Branch networks for multi-domain feature extraction.

This module implements the three domain-specific branches:
1. RGB Branch - EfficientNet-based spatial feature extractor
2. Frequency Branch - CNN for FFT spectrum analysis
3. Noise Branch - SRM-initialized residual pattern analyzer
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Optional, List
import numpy as np


class RGBBranch(nn.Module):
    """RGB/Spatial domain branch using pretrained backbone.
    
    Uses EfficientNet or similar backbone to extract high-level
    spatial features from normalized RGB images.
    """
    
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        pretrained: bool = True,
        feature_dim: int = 512,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Load pretrained backbone
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,  # Remove classifier
            global_pool='avg',
        )
        
        # Get backbone output dimension
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 256, 256)
            backbone_dim = self.backbone(dummy).shape[1]
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: RGB tensor [B, 3, H, W]
        
        Returns:
            Features [B, feature_dim]
        """
        features = self.backbone(x)
        features = self.projection(features)
        return features


class ConvBlock(nn.Module):
    """Basic convolutional block with BatchNorm and ReLU."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class FrequencyBranch(nn.Module):
    """Frequency domain branch for FFT spectrum analysis.
    
    Processes the log-magnitude FFT spectrum to detect periodic
    artifacts and frequency-domain signatures of generative models.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 256,
        base_channels: int = 32,
        num_layers: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Build convolutional layers with progressive downsampling
        layers = []
        in_ch = input_channels
        out_ch = base_channels
        
        for i in range(num_layers):
            layers.append(ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 256, 256)
            conv_out = self.conv_layers(dummy)
            flat_size = conv_out.view(1, -1).shape[1]
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self.feature_dim = feature_dim
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: FFT spectrum tensor [B, 1, H, W]
        
        Returns:
            Features [B, feature_dim]
        """
        x = self.conv_layers(x)
        x = self.projection(x)
        return x


class NoiseBranch(nn.Module):
    """Noise domain branch for residual pattern analysis.
    
    Uses SRM (Spatial Rich Model) filters to extract noise residuals
    and learns discriminative features from subtle artifacts.
    """
    
    # SRM filter kernels
    SRM_KERNELS = {
        'srm1': np.array([
            [ 0,  0,  0,  0,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  2, -4,  2,  0],
            [ 0, -1,  2, -1,  0],
            [ 0,  0,  0,  0,  0]
        ], dtype=np.float32) / 4.0,
        
        'srm2': np.array([
            [-1,  2, -2,  2, -1],
            [ 2, -6,  8, -6,  2],
            [-2,  8, -12, 8, -2],
            [ 2, -6,  8, -6,  2],
            [-1,  2, -2,  2, -1]
        ], dtype=np.float32) / 12.0,
        
        'srm3': np.array([
            [ 0,  0, -1,  0,  0],
            [ 0,  0,  2,  0,  0],
            [-1,  2, -4,  2, -1],
            [ 0,  0,  2,  0,  0],
            [ 0,  0, -1,  0,  0]
        ], dtype=np.float32) / 4.0,
    }
    
    def __init__(
        self,
        input_channels: int = 3,  # Number of SRM filter outputs
        feature_dim: int = 256,
        base_channels: int = 32,
        num_layers: int = 4,
        use_srm_layer: bool = True,
        freeze_srm: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.use_srm_layer = use_srm_layer
        
        # Optional learnable SRM layer (applied to RGB input)
        if use_srm_layer:
            self.srm_conv = self._create_srm_conv(freeze=freeze_srm)
            conv_input_channels = len(self.SRM_KERNELS)
        else:
            self.srm_conv = None
            conv_input_channels = input_channels
        
        # Build convolutional layers
        layers = []
        in_ch = conv_input_channels
        out_ch = base_channels
        
        for i in range(num_layers):
            layers.append(ConvBlock(in_ch, out_ch, kernel_size=3, stride=1, padding=1))
            layers.append(nn.MaxPool2d(2, 2))
            in_ch = out_ch
            out_ch = min(out_ch * 2, 256)
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            dummy = torch.zeros(1, conv_input_channels, 256, 256)
            conv_out = self.conv_layers(dummy)
            flat_size = conv_out.view(1, -1).shape[1]
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_size, feature_dim * 2),
            nn.BatchNorm1d(feature_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(feature_dim * 2, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
        )
        
        self.feature_dim = feature_dim
    
    def _create_srm_conv(self, freeze: bool = True) -> nn.Conv2d:
        """Create convolution layer initialized with SRM filters."""
        kernels = list(self.SRM_KERNELS.values())
        num_kernels = len(kernels)
        
        conv = nn.Conv2d(
            in_channels=1,  # Applied per-channel
            out_channels=num_kernels,
            kernel_size=5,
            padding=2,
            bias=False,
        )
        
        # Initialize with SRM kernels
        with torch.no_grad():
            weight = torch.zeros(num_kernels, 1, 5, 5)
            for k, kernel in enumerate(kernels):
                weight[k, 0] = torch.from_numpy(kernel)
            conv.weight.copy_(weight)
        
        if freeze:
            conv.weight.requires_grad = False
        
        return conv
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Noise residual tensor [B, num_filters, H, W] or
               RGB tensor [B, 3, H, W] if use_srm_layer=True
        
        Returns:
            Features [B, feature_dim]
        """
        if self.use_srm_layer and self.srm_conv is not None:
            # Apply SRM filters to grayscale
            gray = x.mean(dim=1, keepdim=True)
            x = self.srm_conv(gray)
            # Truncate residuals
            x = torch.clamp(x, -3, 3) / 3.0
        
        x = self.conv_layers(x)
        x = self.projection(x)
        return x


class BranchEnsemble(nn.Module):
    """Simple ensemble of all three branches (without fusion).
    
    Useful for testing individual branch performance.
    """
    
    def __init__(
        self,
        rgb_branch: RGBBranch,
        freq_branch: FrequencyBranch,
        noise_branch: NoiseBranch,
    ):
        super().__init__()
        self.rgb_branch = rgb_branch
        self.freq_branch = freq_branch
        self.noise_branch = noise_branch
        
        self.total_dim = (
            rgb_branch.feature_dim +
            freq_branch.feature_dim +
            noise_branch.feature_dim
        )
    
    def forward(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """Extract and concatenate features from all branches."""
        rgb_feat = self.rgb_branch(rgb)
        freq_feat = self.freq_branch(freq)
        noise_feat = self.noise_branch(noise)
        
        return torch.cat([rgb_feat, freq_feat, noise_feat], dim=1)


def create_branches(
    rgb_backbone: str = "efficientnet_b0",
    rgb_pretrained: bool = True,
    rgb_feature_dim: int = 512,
    freq_feature_dim: int = 256,
    noise_feature_dim: int = 256,
) -> tuple:
    """Factory function to create all three branches.
    
    Returns:
        Tuple of (rgb_branch, freq_branch, noise_branch)
    """
    rgb_branch = RGBBranch(
        backbone=rgb_backbone,
        pretrained=rgb_pretrained,
        feature_dim=rgb_feature_dim,
    )
    
    freq_branch = FrequencyBranch(
        input_channels=1,
        feature_dim=freq_feature_dim,
    )
    
    noise_branch = NoiseBranch(
        input_channels=3,
        feature_dim=noise_feature_dim,
        use_srm_layer=False,  # Pre-computed noise residuals
    )
    
    return rgb_branch, freq_branch, noise_branch
