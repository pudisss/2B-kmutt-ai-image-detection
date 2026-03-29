"""Evaluation metrics for AI image detection."""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


class MetricsCalculator:
    """Calculate and track evaluation metrics."""
    
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        self.reset()
    
    def reset(self):
        """Reset accumulated predictions."""
        self.all_labels = []
        self.all_probs = []
        self.all_preds = []
        self.per_arch_data = defaultdict(lambda: {'labels': [], 'probs': [], 'preds': []})

    def set_threshold(self, threshold: float):
        """Set decision threshold for binary classification."""
        self.threshold = float(threshold)
    
    def update(
        self,
        labels: torch.Tensor,
        probs: torch.Tensor,
        architectures: Optional[List[str]] = None,
    ):
        """Add batch predictions.
        
        Args:
            labels: Ground truth labels [B]
            probs: Predicted probabilities [B, 2] or [B] (for class 1)
            architectures: Optional list of architecture names for each sample
        """
        labels_np = labels.cpu().numpy()
        
        if probs.dim() == 2:
            probs_np = probs[:, 1].cpu().numpy()  # Probability of class 1 (fake)
        else:
            probs_np = probs.cpu().numpy()
        
        preds_np = (probs_np >= self.threshold).astype(int)
        
        self.all_labels.extend(labels_np)
        self.all_probs.extend(probs_np)
        self.all_preds.extend(preds_np)
        
        # Track per-architecture if provided
        if architectures is not None:
            for i, arch in enumerate(architectures):
                self.per_arch_data[arch]['labels'].append(labels_np[i])
                self.per_arch_data[arch]['probs'].append(probs_np[i])
                self.per_arch_data[arch]['preds'].append(preds_np[i])
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics."""
        return self.compute_at_threshold(self.threshold)

    def compute_at_threshold(self, threshold: float) -> Dict[str, float]:
        """Compute all metrics using a specific decision threshold."""
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        preds = (probs >= float(threshold)).astype(int)
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(labels, preds)
        metrics['balanced_accuracy'] = balanced_accuracy_score(labels, preds)
        metrics['precision'] = precision_score(labels, preds, zero_division=0)
        metrics['recall'] = recall_score(labels, preds, zero_division=0)
        metrics['f1'] = f1_score(labels, preds, zero_division=0)
        
        # AUC (if both classes present)
        if len(np.unique(labels)) > 1:
            metrics['auc'] = roc_auc_score(labels, probs)
        else:
            metrics['auc'] = 0.0

        metrics['threshold'] = float(threshold)
        
        return metrics
    
    def compute_per_architecture(self, threshold: Optional[float] = None) -> Dict[str, Dict[str, float]]:
        """Compute metrics for each architecture."""
        results = {}
        use_threshold = self.threshold if threshold is None else float(threshold)
        
        for arch, data in self.per_arch_data.items():
            labels = np.array(data['labels'])
            probs = np.array(data['probs'])
            preds = (probs >= use_threshold).astype(int)
            
            if len(labels) == 0:
                continue
            
            arch_metrics = {
                'count': len(labels),
                'accuracy': accuracy_score(labels, preds),
                'balanced_accuracy': balanced_accuracy_score(labels, preds),
            }
            
            if len(np.unique(labels)) > 1:
                arch_metrics['auc'] = roc_auc_score(labels, probs)
            else:
                arch_metrics['auc'] = 0.0
            
            results[arch] = arch_metrics
        
        return results
    
    def get_confusion_matrix(self, threshold: Optional[float] = None) -> np.ndarray:
        """Get confusion matrix."""
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        use_threshold = self.threshold if threshold is None else float(threshold)
        preds = (probs >= use_threshold).astype(int)
        return confusion_matrix(labels, preds, labels=[0, 1])
    
    def get_roc_curve(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get ROC curve data.
        
        Returns:
            Tuple of (fpr, tpr, thresholds)
        """
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        return roc_curve(labels, probs)
    
    def get_classification_report(self, threshold: Optional[float] = None) -> str:
        """Get sklearn classification report string."""
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        use_threshold = self.threshold if threshold is None else float(threshold)
        preds = (probs >= use_threshold).astype(int)
        return classification_report(
            labels, preds,
            labels=[0, 1],
            target_names=['Real', 'Fake'],
            digits=4,
            zero_division=0,
        )

    def find_best_threshold(
        self,
        metric: str = 'balanced_accuracy',
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
        steps: int = 181,
    ) -> Tuple[float, Dict[str, float]]:
        """Grid-search threshold on accumulated labels/probabilities."""
        labels = np.array(self.all_labels)
        probs = np.array(self.all_probs)
        if len(labels) == 0:
            raise ValueError("No predictions accumulated for threshold tuning.")
        if len(np.unique(labels)) < 2:
            # Can't tune meaningfully if only one class is present.
            threshold = 0.5
            return threshold, self.compute_at_threshold(threshold)

        thresholds = np.linspace(min_threshold, max_threshold, steps)
        best_threshold = 0.5
        best_metrics = self.compute_at_threshold(best_threshold)
        best_score = best_metrics.get(metric, float('-inf'))

        for threshold in thresholds:
            metrics = self.compute_at_threshold(float(threshold))
            score = metrics.get(metric, float('-inf'))
            if score > best_score or (
                np.isclose(score, best_score) and abs(float(threshold) - 0.5) < abs(best_threshold - 0.5)
            ):
                best_score = score
                best_threshold = float(threshold)
                best_metrics = metrics

        return best_threshold, best_metrics


class EarlyStopping:
    """Early stopping callback."""
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.001,
        mode: str = 'max',
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        
        self.counter = 0
        self.best_value = None
        self.should_stop = False
    
    def __call__(self, value: float) -> bool:
        """Check if training should stop.
        
        Args:
            value: Current metric value
        
        Returns:
            True if training should stop
        """
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                return True
        
        return False
    
    def reset(self):
        """Reset early stopping state."""
        self.counter = 0
        self.best_value = None
        self.should_stop = False


def compute_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
    """Compute batch accuracy."""
    with torch.no_grad():
        preds = output.argmax(dim=1)
        correct = (preds == target).sum().item()
        return correct / target.size(0)
