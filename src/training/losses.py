"""Loss functions for AI image detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance.
    
    Reduces the relative loss for well-classified examples,
    putting more focus on hard, misclassified examples.
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = 'mean',
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [B, C]
            targets: Labels [B]
        
        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    """Cross-entropy loss with label smoothing."""
    
    def __init__(
        self,
        smoothing: float = 0.1,
        num_classes: int = 2,
    ):
        super().__init__()
        self.smoothing = smoothing
        self.num_classes = num_classes
        self.confidence = 1.0 - smoothing
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Logits [B, C]
            targets: Labels [B]
        
        Returns:
            Smoothed cross-entropy loss
        """
        log_probs = F.log_softmax(inputs, dim=1)
        
        # Create smooth labels
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), self.confidence)
        
        return (-true_dist * log_probs).sum(dim=1).mean()


class CombinedLoss(nn.Module):
    """Combined loss with multiple components."""
    
    def __init__(
        self,
        use_focal: bool = False,
        use_smoothing: bool = True,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        smoothing: float = 0.1,
        class_weights: Optional[torch.Tensor] = None,
    ):
        super().__init__()
        
        self.use_focal = use_focal
        self.use_smoothing = use_smoothing
        
        if use_focal:
            self.loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif use_smoothing:
            self.loss_fn = LabelSmoothingLoss(smoothing=smoothing)
        else:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            outputs: Model outputs dict with 'logits' key
            targets: Labels [B]
        
        Returns:
            Dict with 'loss' and component losses
        """
        logits = outputs['logits']
        loss = self.loss_fn(logits, targets)
        
        return {
            'loss': loss,
            'ce_loss': loss,
        }


def create_loss_fn(
    loss_type: str = 'cross_entropy',
    class_weights: Optional[torch.Tensor] = None,
    **kwargs,
) -> nn.Module:
    """Factory function to create loss function.
    
    Args:
        loss_type: One of 'cross_entropy', 'focal', 'label_smoothing'
        class_weights: Optional class weights for imbalanced data
        **kwargs: Additional loss function arguments
    
    Returns:
        Loss function module
    """
    if loss_type == 'cross_entropy':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_type == 'focal':
        return FocalLoss(
            alpha=kwargs.get('focal_alpha', 0.25),
            gamma=kwargs.get('focal_gamma', 2.0),
        )
    elif loss_type == 'label_smoothing':
        return LabelSmoothingLoss(
            smoothing=kwargs.get('smoothing', 0.1),
        )
    elif loss_type == 'combined':
        return CombinedLoss(
            use_focal=kwargs.get('use_focal', False),
            use_smoothing=kwargs.get('use_smoothing', True),
            class_weights=class_weights,
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
