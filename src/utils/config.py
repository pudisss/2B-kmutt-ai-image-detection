"""Configuration management for AI Image Detector."""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def merge_configs(base: Dict, override: Dict) -> Dict:
    """Recursively merge override config into base config."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    return result


@dataclass
class DataConfig:
    """Data configuration."""
    root_dir: str = "./data/processed"
    image_size: int = 256
    normalize_mean: List[float] = field(default_factory=lambda: [0.485, 0.456, 0.406])
    normalize_std: List[float] = field(default_factory=lambda: [0.229, 0.224, 0.225])
    batch_size: int = 32
    num_workers: int = 8
    pin_memory: bool = True
    balance_classes: bool = True
    samples_per_class: Optional[int] = None
    val_split: float = 0.1
    weighted_sampling: bool = True
    augmentation: Dict[str, Any] = field(default_factory=dict)
    train_models: Dict[str, List[str]] = field(default_factory=dict)
    test_models: Dict[str, List[str]] = field(default_factory=dict)
    datasets: List[Dict] = field(default_factory=list)


@dataclass
class ModelConfig:
    """Model configuration."""
    # RGB branch (Swin Transformer)
    rgb_model: str = "microsoft/swin-tiny-patch4-window7-224"
    rgb_pretrained: bool = True
    rgb_feature_dim: int = 512
    # Frequency branch (ResNet)
    freq_feature_dim: int = 256
    freq_resnet: str = "resnet18"
    # Noise branch (ResNet)
    noise_feature_dim: int = 256
    noise_resnet: str = "resnet18"
    # Fusion
    fusion_type: str = "attention"
    fusion_hidden_dim: int = 512
    fusion_num_heads: int = 8
    fusion_dropout: float = 0.3
    # Classifier
    classifier_hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    classifier_dropout: float = 0.5
    num_classes: int = 2


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    use_amp: bool = True
    gradient_clip: float = 1.0
    accumulation_steps: int = 1
    early_stopping_patience: int = 10
    checkpoint_dir: str = "./checkpoints"
    save_best_only: bool = True
    threshold_metric: str = "balanced_accuracy"


@dataclass
class Config:
    """Main configuration container."""
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    seed: int = 42
    device: str = "cuda"
    
    @classmethod
    def from_yaml(cls, config_path: str) -> "Config":
        """Create Config from YAML file."""
        raw = load_config(config_path)
        
        # Parse data config
        data_raw = raw.get("data", {})
        data = DataConfig(
            root_dir=data_raw.get("root_dir", "./data/processed"),
            image_size=data_raw.get("image_size", 256),
            normalize_mean=data_raw.get("normalize_mean", [0.485, 0.456, 0.406]),
            normalize_std=data_raw.get("normalize_std", [0.229, 0.224, 0.225]),
            batch_size=data_raw.get("batch_size", 32),
            num_workers=data_raw.get("num_workers", 8),
            pin_memory=data_raw.get("pin_memory", True),
            balance_classes=data_raw.get("balance_classes", True),
            samples_per_class=data_raw.get("samples_per_class"),
            val_split=data_raw.get("val_split", 0.1),
            weighted_sampling=data_raw.get("weighted_sampling", True),
            augmentation=data_raw.get("augmentation", {}),
            train_models=data_raw.get("train_models", {}),
            test_models=data_raw.get("test_models", {}),
            datasets=data_raw.get("datasets", []),
        )
        
        # Parse model config
        model_raw = raw.get("model", {})
        rgb = model_raw.get("rgb_branch", {})
        freq = model_raw.get("frequency_branch", {})
        noise = model_raw.get("noise_branch", {})
        fusion = model_raw.get("fusion", {})
        classifier = model_raw.get("classifier", {})
        
        model = ModelConfig(
            rgb_model=rgb.get("model_name", "microsoft/swin-tiny-patch4-window7-224"),
            rgb_pretrained=rgb.get("pretrained", True),
            rgb_feature_dim=rgb.get("feature_dim", 512),
            freq_feature_dim=freq.get("feature_dim", 256),
            freq_resnet=freq.get("resnet_type", "resnet18"),
            noise_feature_dim=noise.get("feature_dim", 256),
            noise_resnet=noise.get("resnet_type", "resnet18"),
            fusion_type=fusion.get("type", "attention"),
            fusion_hidden_dim=fusion.get("hidden_dim", 512),
            fusion_num_heads=fusion.get("num_heads", 8),
            fusion_dropout=fusion.get("dropout", 0.3),
            classifier_hidden_dims=classifier.get("hidden_dims", [256, 128]),
            classifier_dropout=classifier.get("dropout", 0.5),
            num_classes=classifier.get("num_classes", 2),
        )
        
        # Parse training config
        train_raw = raw.get("training", {})
        early = train_raw.get("early_stopping", {})
        
        training = TrainingConfig(
            epochs=train_raw.get("epochs", 50),
            learning_rate=train_raw.get("learning_rate", 1e-4),
            weight_decay=train_raw.get("weight_decay", 1e-5),
            optimizer=train_raw.get("optimizer", "adamw"),
            scheduler=train_raw.get("scheduler", "cosine"),
            warmup_epochs=train_raw.get("warmup_epochs", 5),
            min_lr=train_raw.get("min_lr", 1e-6),
            use_amp=train_raw.get("use_amp", True),
            gradient_clip=train_raw.get("gradient_clip", 1.0),
            accumulation_steps=train_raw.get("accumulation_steps", 1),
            early_stopping_patience=early.get("patience", 10),
            checkpoint_dir=train_raw.get("checkpoint_dir", "./checkpoints"),
            save_best_only=train_raw.get("save_best_only", True),
            threshold_metric=train_raw.get("threshold_metric", "balanced_accuracy"),
        )
        
        return cls(
            data=data,
            model=model,
            training=training,
            seed=raw.get("seed", 42),
            device=raw.get("device", "cuda"),
        )


def get_default_config() -> Config:
    """Get default configuration."""
    default_path = Path(__file__).parent.parent.parent / "configs" / "default.yaml"
    if default_path.exists():
        return Config.from_yaml(str(default_path))
    return Config(
        data=DataConfig(),
        model=ModelConfig(),
        training=TrainingConfig(),
    )
