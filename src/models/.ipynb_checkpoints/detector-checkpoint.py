"""Full multi-domain detector model.

This module assembles the complete detector by combining:
- Three domain-specific branches (RGB, Frequency, Noise)
- Fusion module for combining features
- Classification head for final prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from .branches import RGBBranch, FrequencyBranch, NoiseBranch, create_branches
from .fusion import create_fusion_module


class ClassificationHead(nn.Module):
    """Classification head with multiple FC layers."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [256, 128],
        num_classes: int = 2,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, num_classes))
        
        self.classifier = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


class MultiDomainDetector(nn.Module):
    """Multi-domain AI image detector.
    
    Combines RGB, frequency, and noise domain features to detect
    AI-generated images across different generative architectures.
    """
    
    def __init__(
        self,
        # RGB branch config
        rgb_backbone: str = "efficientnet_b0",
        rgb_pretrained: bool = True,
        rgb_feature_dim: int = 512,
        # Frequency branch config
        freq_feature_dim: int = 256,
        # Noise branch config
        noise_feature_dim: int = 256,
        # Fusion config
        fusion_type: str = "attention",
        fusion_hidden_dim: int = 512,
        fusion_num_heads: int = 8,
        fusion_dropout: float = 0.3,
        # Classifier config
        classifier_hidden_dims: List[int] = [256, 128],
        classifier_dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        
        # Create branches
        self.rgb_branch = RGBBranch(
            backbone=rgb_backbone,
            pretrained=rgb_pretrained,
            feature_dim=rgb_feature_dim,
        )
        
        self.freq_branch = FrequencyBranch(
            input_channels=1,
            feature_dim=freq_feature_dim,
        )
        
        self.noise_branch = NoiseBranch(
            input_channels=3,
            feature_dim=noise_feature_dim,
            use_srm_layer=False,  # Pre-computed residuals from dataset
        )
        
        # Create fusion module
        branch_dims = [rgb_feature_dim, freq_feature_dim, noise_feature_dim]
        self.fusion = create_fusion_module(
            fusion_type=fusion_type,
            input_dims=branch_dims,
            hidden_dim=fusion_hidden_dim,
            num_heads=fusion_num_heads,
            dropout=fusion_dropout,
        )
        
        # Create classifier
        self.classifier = ClassificationHead(
            input_dim=self.fusion.output_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )
        
        # Store config for later
        self.config = {
            'rgb_backbone': rgb_backbone,
            'fusion_type': fusion_type,
            'num_classes': num_classes,
        }
    
    def forward(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        noise: torch.Tensor,
        return_features: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb: RGB tensor [B, 3, H, W]
            freq: FFT spectrum tensor [B, 1, H, W]
            noise: Noise residual tensor [B, 3, H, W]
            return_features: If True, return intermediate features
        
        Returns:
            Dict with 'logits' and optionally 'features'
        """
        # Extract branch features
        rgb_feat = self.rgb_branch(rgb)
        freq_feat = self.freq_branch(freq)
        noise_feat = self.noise_branch(noise)
        
        # Fuse features
        fused = self.fusion([rgb_feat, freq_feat, noise_feat])
        
        # Classify
        logits = self.classifier(fused)
        
        output = {'logits': logits}
        
        if return_features:
            output['rgb_features'] = rgb_feat
            output['freq_features'] = freq_feat
            output['noise_features'] = noise_feat
            output['fused_features'] = fused
        
        return output
    
    def predict(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        noise: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get predictions and probabilities.
        
        Returns:
            Tuple of (predictions, probabilities)
        """
        output = self.forward(rgb, freq, noise)
        logits = output['logits']
        
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        return preds, probs
    
    def get_branch_features(
        self,
        rgb: torch.Tensor,
        freq: torch.Tensor,
        noise: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Get features from each branch for analysis."""
        return {
            'rgb': self.rgb_branch(rgb),
            'freq': self.freq_branch(freq),
            'noise': self.noise_branch(noise),
        }


class SingleBranchDetector(nn.Module):
    """Single-branch detector for ablation studies.
    
    Uses only one domain for detection to measure individual
    branch contribution.
    """
    
    def __init__(
        self,
        branch_type: str = "rgb",
        # Branch configs (same as MultiDomainDetector)
        rgb_backbone: str = "efficientnet_b0",
        rgb_pretrained: bool = True,
        feature_dim: int = 512,
        # Classifier config
        classifier_hidden_dims: List[int] = [256, 128],
        classifier_dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()
        
        self.branch_type = branch_type
        
        if branch_type == "rgb":
            self.branch = RGBBranch(
                backbone=rgb_backbone,
                pretrained=rgb_pretrained,
                feature_dim=feature_dim,
            )
        elif branch_type == "freq":
            self.branch = FrequencyBranch(
                input_channels=1,
                feature_dim=feature_dim,
            )
        elif branch_type == "noise":
            self.branch = NoiseBranch(
                input_channels=3,
                feature_dim=feature_dim,
                use_srm_layer=False,
            )
        else:
            raise ValueError(f"Unknown branch type: {branch_type}")
        
        self.classifier = ClassificationHead(
            input_dim=self.branch.feature_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: Input tensor appropriate for branch type
        
        Returns:
            Dict with 'logits'
        """
        features = self.branch(x)
        logits = self.classifier(features)
        return {'logits': logits}


def create_detector(
    model_config: Optional[Dict] = None,
    **kwargs,
) -> MultiDomainDetector:
    """Factory function to create detector from config.
    
    Args:
        model_config: Model configuration dict
        **kwargs: Override specific parameters
    
    Returns:
        MultiDomainDetector instance
    """
    # Default config
    config = {
        'rgb_backbone': 'efficientnet_b0',
        'rgb_pretrained': True,
        'rgb_feature_dim': 512,
        'freq_feature_dim': 256,
        'noise_feature_dim': 256,
        'fusion_type': 'attention',
        'fusion_hidden_dim': 512,
        'fusion_num_heads': 8,
        'fusion_dropout': 0.3,
        'classifier_hidden_dims': [256, 128],
        'classifier_dropout': 0.5,
        'num_classes': 2,
    }
    
    # Update from model_config if provided
    if model_config is not None:
        config.update(model_config)
    
    # Override with kwargs
    config.update(kwargs)
    
    return MultiDomainDetector(**config)


def load_detector(
    checkpoint_path: str,
    device: str = "cuda",
    **override_config,
) -> MultiDomainDetector:
    """Load detector from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        **override_config: Override config parameters
    
    Returns:
        Loaded MultiDomainDetector
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get config from checkpoint
    config = checkpoint.get('config', {})
    config.update(override_config)
    
    # Create model
    model = create_detector(**config)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model
