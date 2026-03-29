"""Logging utilities for AI Image Detector."""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str = "ai_image_detector",
    level: str = "INFO",
    log_file: Optional[str] = None,
) -> logging.Logger:
    """Set up and return a logger with console and optional file handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_format = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(logging.DEBUG)
        file_format = logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(file_format)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_name(prefix: str = "exp") -> str:
    """Generate a unique experiment name based on timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


class MetricLogger:
    """Simple metric logger for tracking training progress."""
    
    def __init__(self, delimiter: str = "  "):
        self.meters = {}
        self.delimiter = delimiter
    
    def update(self, **kwargs):
        """Update metrics with new values."""
        for k, v in kwargs.items():
            if k not in self.meters:
                self.meters[k] = []
            self.meters[k].append(v)
    
    def get_avg(self, key: str) -> float:
        """Get average value for a metric."""
        if key not in self.meters or len(self.meters[key]) == 0:
            return 0.0
        return sum(self.meters[key]) / len(self.meters[key])
    
    def get_last(self, key: str) -> float:
        """Get last value for a metric."""
        if key not in self.meters or len(self.meters[key]) == 0:
            return 0.0
        return self.meters[key][-1]
    
    def reset(self):
        """Reset all metrics."""
        self.meters = {}
    
    def __str__(self) -> str:
        """Format all metrics as a string."""
        entries = []
        for name, values in self.meters.items():
            if len(values) > 0:
                avg = sum(values) / len(values)
                entries.append(f"{name}: {avg:.4f}")
        return self.delimiter.join(entries)
