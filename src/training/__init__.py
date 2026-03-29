"""Training utilities for AI image detection."""

from .trainer import Trainer, Evaluator
from .losses import (
    FocalLoss,
    LabelSmoothingLoss,
    CombinedLoss,
    create_loss_fn,
)
from .metrics import (
    MetricsCalculator,
    EarlyStopping,
    compute_accuracy,
)

__all__ = [
    'Trainer',
    'Evaluator',
    'FocalLoss',
    'LabelSmoothingLoss',
    'CombinedLoss',
    'create_loss_fn',
    'MetricsCalculator',
    'EarlyStopping',
    'compute_accuracy',
]
