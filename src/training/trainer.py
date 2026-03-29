"""Training pipeline for multi-domain AI image detector."""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, LinearLR
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime

from ..models.detector import MultiDomainDetector
from .metrics import MetricsCalculator, EarlyStopping, compute_accuracy
from .losses import create_loss_fn


class Trainer:
    """Training manager for multi-domain detector."""
    
    def __init__(
        self,
        model: MultiDomainDetector,
        train_loader: DataLoader,
        val_loader: DataLoader,
        # Training config
        epochs: int = 50,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        # Optimizer
        optimizer: str = "adamw",
        # Scheduler
        scheduler: str = "cosine",
        warmup_epochs: int = 5,
        min_lr: float = 1e-6,
        # Mixed precision
        use_amp: bool = True,
        # Gradient settings
        gradient_clip: float = 1.0,
        accumulation_steps: int = 1,
        # Early stopping
        early_stopping_patience: int = 10,
        # Checkpointing
        checkpoint_dir: str = "./checkpoints",
        save_best_only: bool = True,
        # Device
        device: str = "cuda",
        # Logging
        log_interval: int = 100,
        # Class weights
        class_weights: Optional[torch.Tensor] = None,
        threshold_metric: str = "balanced_accuracy",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.use_amp = use_amp and device == "cuda"
        self.gradient_clip = float(gradient_clip)
        self.accumulation_steps = int(accumulation_steps)
        self.log_interval = int(log_interval)
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer(optimizer, float(weight_decay))
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler(
            scheduler, int(warmup_epochs), float(min_lr), len(train_loader)
        )
        
        # Initialize loss function
        if class_weights is not None:
            class_weights = class_weights.to(device)
        self.criterion = create_loss_fn('label_smoothing')
        
        # Initialize AMP scaler
        self.scaler = GradScaler() if self.use_amp else None
        
        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=int(early_stopping_patience),
            mode='max',
        )
        
        # Initialize metrics
        self.metrics_calculator = MetricsCalculator()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_auc = 0.0
        self.training_history = []
        self.best_threshold = 0.5
        self.threshold_metric = threshold_metric
    
    def _create_optimizer(self, optimizer_type: str, weight_decay: float):
        """Create optimizer."""
        # Separate parameters for different learning rates
        # Pretrained backbone params get lower LR
        pretrained_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            # Swin and ResNet backbone params
            if 'swin' in name or 'layer1' in name or 'layer2' in name or 'layer3' in name or 'layer4' in name or 'bn1' in name:
                pretrained_params.append(param)
            else:
                other_params.append(param)
        
        param_groups = [
            {'params': pretrained_params, 'lr': float(self.learning_rate) * 0.1},  # Lower LR for pretrained
            {'params': other_params, 'lr': float(self.learning_rate)},
        ]
        
        if optimizer_type == "adamw":
            return AdamW(param_groups, weight_decay=weight_decay)
        elif optimizer_type == "sgd":
            return SGD(param_groups, momentum=0.9, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def _create_scheduler(
        self,
        scheduler_type: str,
        warmup_epochs: int,
        min_lr: float,
        steps_per_epoch: int,
    ):
        """Create learning rate scheduler."""
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch
        
        if scheduler_type == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=min_lr,
            )
        elif scheduler_type == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.learning_rate,
                total_steps=total_steps,
                pct_start=warmup_epochs / self.epochs,
            )
        else:
            return CosineAnnealingLR(self.optimizer, T_max=total_steps, eta_min=min_lr)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            rgb = batch['rgb'].to(self.device)
            freq = batch['freq'].to(self.device)
            noise = batch['noise'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass with AMP
            with autocast(enabled=self.use_amp):
                outputs = self.model(rgb, freq, noise)
                loss = self.criterion(outputs['logits'], labels)
                loss = loss / self.accumulation_steps
            
            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.use_amp:
                    self.scaler.unscale_(self.optimizer)
                
                # Gradient clipping
                if self.gradient_clip > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip)
                
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1
            
            # Track metrics
            total_loss += loss.item() * self.accumulation_steps
            accuracy = compute_accuracy(outputs['logits'], labels)
            total_correct += accuracy * labels.size(0)
            total_samples += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{total_loss / (batch_idx + 1):.4f}",
                'acc': f"{total_correct / total_samples:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_acc': total_correct / total_samples,
        }
    
    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        self.metrics_calculator.reset()
        
        total_loss = 0.0
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            rgb = batch['rgb'].to(self.device)
            freq = batch['freq'].to(self.device)
            noise = batch['noise'].to(self.device)
            labels = batch['label'].to(self.device)
            
            with autocast(enabled=self.use_amp):
                outputs = self.model(rgb, freq, noise)
                loss = self.criterion(outputs['logits'], labels)
            
            total_loss += loss.item()
            
            # Get probabilities
            probs = F.softmax(outputs['logits'], dim=1)
            archs = None
            if 'metadata' in batch:
                archs = [m.get('architecture', 'unknown') for m in batch['metadata']]
            self.metrics_calculator.update(labels, probs, archs)
        
        best_threshold, metrics = self.metrics_calculator.find_best_threshold(
            metric=self.threshold_metric
        )
        self.best_threshold = best_threshold
        self.metrics_calculator.set_threshold(best_threshold)
        metrics = self.metrics_calculator.compute()
        metrics['val_loss'] = total_loss / len(self.val_loader)
        metrics['best_threshold'] = best_threshold
        metrics['per_architecture'] = self.metrics_calculator.compute_per_architecture(
            threshold=best_threshold
        )
        
        return metrics
    
    def train(self) -> Dict[str, list]:
        """Full training loop."""
        print(f"\nStarting training for {self.epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Train batches: {len(self.train_loader)}")
        print(f"Val batches: {len(self.val_loader)}")
        print(f"AMP enabled: {self.use_amp}")
        print()
        
        for epoch in range(self.epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch()
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            epoch_metrics = {**train_metrics, **val_metrics}
            self.training_history.append(epoch_metrics)
            
            # Print epoch summary
            print(f"\nEpoch {epoch + 1}/{self.epochs}")
            print(f"  Train Loss: {train_metrics['train_loss']:.4f}, "
                  f"Train Acc: {train_metrics['train_acc']:.4f}")
            print(f"  Val Loss: {val_metrics['val_loss']:.4f}, "
                  f"Val Acc: {val_metrics['accuracy']:.4f}, "
                  f"Val AUC: {val_metrics['auc']:.4f}, "
                  f"Val BalAcc: {val_metrics['balanced_accuracy']:.4f}, "
                  f"Thr: {val_metrics['best_threshold']:.3f}")
            
            # Check for best model
            is_best = val_metrics['auc'] > self.best_val_auc
            if is_best:
                self.best_val_auc = val_metrics['auc']
                print(f"  New best AUC: {self.best_val_auc:.4f}")
            
            # Save checkpoint
            if is_best or not self.save_best_only:
                self.save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.early_stopping(val_metrics['auc']):
                print(f"\nEarly stopping triggered at epoch {epoch + 1}")
                break
        
        # Save final model and history
        self.save_checkpoint(is_best=False, filename='final.pt')
        self._save_history()
        
        return {k: [m[k] for m in self.training_history] for k in self.training_history[0]}
    
    def save_checkpoint(self, is_best: bool = False, filename: Optional[str] = None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_auc': self.best_val_auc,
            'best_threshold': self.best_threshold,
            'config': self.model.config,
        }
        
        if filename is None:
            filename = f"epoch_{self.current_epoch + 1}.pt"
        
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"  Saved best model to {best_path}")
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = self.checkpoint_dir / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load checkpoint and resume training."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_val_auc = checkpoint['best_val_auc']
        
        print(f"Resumed from epoch {self.current_epoch}")


class Evaluator:
    """Evaluation manager for trained models."""
    
    def __init__(
        self,
        model: MultiDomainDetector,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        per_architecture: bool = True,
        threshold: float = 0.5,
    ) -> Dict:
        """Evaluate model on a dataset.
        
        Args:
            dataloader: DataLoader to evaluate on
            per_architecture: Whether to compute per-architecture metrics
        
        Returns:
            Dict with metrics and per-architecture breakdown
        """
        metrics_calc = MetricsCalculator(threshold=threshold)
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            rgb = batch['rgb'].to(self.device)
            freq = batch['freq'].to(self.device)
            noise = batch['noise'].to(self.device)
            labels = batch['label'].to(self.device)
            
            outputs = self.model(rgb, freq, noise)
            probs = F.softmax(outputs['logits'], dim=1)
            
            # Get architectures if available
            if per_architecture and 'metadata' in batch:
                archs = [m['architecture'] for m in batch['metadata']]
            else:
                archs = None
            
            metrics_calc.update(labels, probs, archs)
        
        results = {
            'overall': metrics_calc.compute(),
            'confusion_matrix': metrics_calc.get_confusion_matrix().tolist(),
            'classification_report': metrics_calc.get_classification_report(),
        }
        
        if per_architecture:
            results['per_architecture'] = metrics_calc.compute_per_architecture(threshold=threshold)
        results['threshold'] = float(threshold)
        
        return results
    
    def evaluate_per_architecture(
        self,
        dataloaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict]:
        """Evaluate on separate dataloaders per architecture.
        
        Args:
            dataloaders: Dict mapping architecture name to DataLoader
        
        Returns:
            Dict with metrics for each architecture
        """
        results = {}
        
        for arch_name, dataloader in dataloaders.items():
            print(f"\nEvaluating {arch_name}...")
            arch_results = self.evaluate(dataloader, per_architecture=False)
            results[arch_name] = arch_results['overall']
        
        return results
