"""Branch networks for multi-domain feature extraction.

This module implements the three domain-specific branches:
1. RGB Branch - Swin Transformer from HuggingFace
2. Frequency Branch - ResNet for FFT spectrum analysis
3. Noise Branch - ResNet for SRM residual pattern analysis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import SwinModel, SwinConfig, AutoImageProcessor
from typing import Optional, List
import numpy as np


class RGBBranch(nn.Module):
    """RGB/Spatial domain branch using Swin Transformer.
    
    Uses pretrained Swin Transformer from HuggingFace to extract
    hierarchical spatial features from RGB images.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/swin-tiny-patch4-window7-224",
        pretrained: bool = True,
        feature_dim: int = 512,
        dropout: float = 0.2,
        image_size: int = 224,
    ):
        super().__init__()
        
        self.image_size = image_size
        
        # Load Swin Transformer
        if pretrained:
            self.swin = SwinModel.from_pretrained(model_name)
        else:
            config = SwinConfig(image_size=image_size)
            self.swin = SwinModel(config)
        
        # Get hidden size from config
        hidden_size = self.swin.config.hidden_size  # 768 for swin-tiny
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
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
        # Resize if needed (Swin expects 224x224 by default)
        if x.shape[2] != self.image_size or x.shape[3] != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        
        # Forward through Swin
        outputs = self.swin(x)
        
        # Use pooled output (CLS token equivalent)
        pooled = outputs.pooler_output  # [B, hidden_size]
        
        # Project to feature dimension
        features = self.projection(pooled)
        
        return features


class FrequencyBranch(nn.Module):
    """Frequency domain branch using ResNet for FFT spectrum analysis.
    
    Processes the log-magnitude FFT spectrum to detect periodic
    artifacts and frequency-domain signatures of generative models.
    """
    
    def __init__(
        self,
        input_channels: int = 1,
        feature_dim: int = 256,
        resnet_type: str = "resnet18",
        pretrained: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        # Load ResNet backbone
        if resnet_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            backbone_dim = 512
        elif resnet_type == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            backbone_dim = 512
        elif resnet_type == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}")
        
        # Modify first conv layer for single-channel input
        self.input_conv = nn.Conv2d(
            input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize from pretrained weights (average across RGB channels)
        if pretrained:
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
                self.input_conv.weight.copy_(pretrained_weight.repeat(1, input_channels, 1, 1))
        
        # Use ResNet layers (excluding first conv and fc)
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
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
            x: FFT spectrum tensor [B, 1, H, W]
        
        Returns:
            Features [B, feature_dim]
        """
        x = self.input_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        x = self.projection(x)
        return x


class NoiseBranch(nn.Module):
    """Noise domain branch using ResNet for residual pattern analysis.
    
    Uses SRM (Spatial Rich Model) filters to extract noise residuals
    and ResNet to learn discriminative features from subtle artifacts.
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
        resnet_type: str = "resnet18",
        pretrained: bool = True,
        use_srm_layer: bool = False,
        freeze_srm: bool = True,
        dropout: float = 0.2,
    ):
        super().__init__()
        
        self.use_srm_layer = use_srm_layer
        
        # Optional SRM layer (applied to RGB input to extract residuals)
        if use_srm_layer:
            self.srm_conv = self._create_srm_conv(freeze=freeze_srm)
            resnet_input_channels = len(self.SRM_KERNELS)
        else:
            self.srm_conv = None
            resnet_input_channels = input_channels
        
        # Load ResNet backbone
        if resnet_type == "resnet18":
            resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
            backbone_dim = 512
        elif resnet_type == "resnet34":
            resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT if pretrained else None)
            backbone_dim = 512
        elif resnet_type == "resnet50":
            resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported resnet_type: {resnet_type}")
        
        # Modify first conv layer for noise residual input
        self.input_conv = nn.Conv2d(
            resnet_input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # Initialize from pretrained weights
        if pretrained and resnet_input_channels == 3:
            with torch.no_grad():
                self.input_conv.weight.copy_(resnet.conv1.weight)
        elif pretrained:
            with torch.no_grad():
                pretrained_weight = resnet.conv1.weight.mean(dim=1, keepdim=True)
                self.input_conv.weight.copy_(pretrained_weight.repeat(1, resnet_input_channels, 1, 1))
        
        # Use ResNet layers
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(backbone_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.feature_dim = feature_dim
    
    def _create_srm_conv(self, freeze: bool = True) -> nn.Conv2d:
        """Create convolution layer initialized with SRM filters."""
        kernels = list(self.SRM_KERNELS.values())
        num_kernels = len(kernels)
        
        conv = nn.Conv2d(
            in_channels=1,
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
        
        # ResNet forward
        x = self.input_conv(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
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
    rgb_model: str = "microsoft/swin-tiny-patch4-window7-224",
    rgb_pretrained: bool = True,
    rgb_feature_dim: int = 512,
    freq_feature_dim: int = 256,
    noise_feature_dim: int = 256,
    resnet_type: str = "resnet18",
) -> tuple:
    """Factory function to create all three branches.
    
    Returns:
        Tuple of (rgb_branch, freq_branch, noise_branch)
    """
    rgb_branch = RGBBranch(
        model_name=rgb_model,
        pretrained=rgb_pretrained,
        feature_dim=rgb_feature_dim,
    )
    
    freq_branch = FrequencyBranch(
        input_channels=1,
        feature_dim=freq_feature_dim,
        resnet_type=resnet_type,
        pretrained=True,
    )
    
    noise_branch = NoiseBranch(
        input_channels=3,
        feature_dim=noise_feature_dim,
        resnet_type=resnet_type,
        pretrained=True,
        use_srm_layer=False,  # Pre-computed noise residuals
    )
    
    return rgb_branch, freq_branch, noise_branch
