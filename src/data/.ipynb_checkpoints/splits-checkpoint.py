"""Model-aware train/test splitting for generalization testing.

This module implements dataset splitting strategies that ensure models
used for testing are completely unseen during training, enabling
proper evaluation of cross-architecture generalization.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import random


# Architecture type mapping
ARCHITECTURE_TYPES = {
    # GAN models (from CNNDetection)
    'progan': 'gan',
    'stylegan': 'gan',
    'stylegan2': 'gan',
    'biggan': 'gan',
    'cyclegan': 'gan',
    'stargan': 'gan',
    'gaugan': 'gan',
    'deepfake': 'gan',
    'seeingdark': 'gan',
    'san': 'gan',
    'crn': 'gan',
    'imle': 'gan',
    'sitd': 'gan',
    'natural': 'real',
    
    # Diffusion models (from DiffusionForensics)
    'ddpm': 'diffusion',
    'iddpm': 'diffusion',
    'adm': 'diffusion',
    'pndm': 'diffusion',
    'ldm': 'diffusion',
    'sdv1': 'diffusion',
    'sdv1_new1': 'diffusion',
    'sdv1_new2': 'diffusion',
    'sdv2': 'diffusion',
    'dalle2': 'diffusion',
    'midjourney': 'diffusion',
    'if': 'diffusion',
    'vqdiffusion': 'diffusion',
    'natural_images': 'real',
}


def get_architecture_type(arch: str) -> str:
    """Get the architecture type (gan, diffusion, real) for a model."""
    arch_lower = arch.lower().strip()
    return ARCHITECTURE_TYPES.get(arch_lower, 'unknown')


class ModelAwareSplitter:
    """Handles train/test splitting with model-level separation.
    
    Ensures that models used for testing are completely unseen during
    training to properly evaluate generalization across architectures.
    """
    
    def __init__(
        self,
        train_models: Dict[str, List[str]],
        test_models: Dict[str, List[str]],
        seed: int = 42,
    ):
        """
        Args:
            train_models: Dict mapping architecture types to list of model names for training
            test_models: Dict mapping architecture types to list of model names for testing
            seed: Random seed for reproducibility
        """
        self.train_models = {k: set(v) for k, v in train_models.items()}
        self.test_models = {k: set(v) for k, v in test_models.items()}
        self.seed = seed
        
        # Flatten for quick lookup
        self._train_set = set()
        self._test_set = set()
        for models in self.train_models.values():
            self._train_set.update(m.lower() for m in models)
        for models in self.test_models.values():
            self._test_set.update(m.lower() for m in models)
        
        # Validate no overlap
        overlap = self._train_set & self._test_set
        if overlap:
            raise ValueError(f"Models appear in both train and test: {overlap}")
    
    def is_train_model(self, model: str) -> bool:
        """Check if model should be used for training."""
        return model.lower().strip() in self._train_set
    
    def is_test_model(self, model: str) -> bool:
        """Check if model should be used for testing."""
        return model.lower().strip() in self._test_set
    
    def split_dataframe(
        self,
        df: pd.DataFrame,
        architecture_col: str = 'architecture',
        label_col: str = 'label',
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split dataframe into train and test based on model assignments.
        
        Args:
            df: DataFrame with image metadata
            architecture_col: Column containing model/architecture name
            label_col: Column containing label (0=real, 1=fake)
            
        Returns:
            Tuple of (train_df, test_df)
        """
        def assign_split(row):
            arch = row[architecture_col].lower().strip()
            label = row[label_col]
            
            # Real images go to both train and test (split by random sampling)
            if label == 0:
                return 'either'
            
            # Fake images assigned by model
            if self.is_train_model(arch):
                return 'train'
            elif self.is_test_model(arch):
                return 'test'
            else:
                return 'unknown'
        
        df = df.copy()
        df['_split_assignment'] = df.apply(assign_split, axis=1)
        
        # Handle unknown models
        unknown = df[df['_split_assignment'] == 'unknown']
        if len(unknown) > 0:
            unknown_archs = unknown[architecture_col].unique()
            print(f"Warning: Unknown models (assigned to train): {list(unknown_archs)}")
            df.loc[df['_split_assignment'] == 'unknown', '_split_assignment'] = 'train'
        
        # Split real images proportionally (80/20)
        real_mask = df['_split_assignment'] == 'either'
        real_df = df[real_mask].copy()
        
        np.random.seed(self.seed)
        real_indices = real_df.index.tolist()
        np.random.shuffle(real_indices)
        split_idx = int(len(real_indices) * 0.8)
        
        train_real_idx = set(real_indices[:split_idx])
        df.loc[real_mask & df.index.isin(train_real_idx), '_split_assignment'] = 'train'
        df.loc[real_mask & ~df.index.isin(train_real_idx), '_split_assignment'] = 'test'
        
        # Create final splits
        train_df = df[df['_split_assignment'] == 'train'].drop(columns=['_split_assignment'])
        test_df = df[df['_split_assignment'] == 'test'].drop(columns=['_split_assignment'])
        
        return train_df, test_df
    
    def get_split_summary(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        architecture_col: str = 'architecture',
        label_col: str = 'label',
    ) -> Dict:
        """Generate summary statistics for the split."""
        summary = {
            'train': {
                'total': len(train_df),
                'real': len(train_df[train_df[label_col] == 0]),
                'fake': len(train_df[train_df[label_col] == 1]),
                'architectures': train_df[architecture_col].value_counts().to_dict(),
            },
            'test': {
                'total': len(test_df),
                'real': len(test_df[test_df[label_col] == 0]),
                'fake': len(test_df[test_df[label_col] == 1]),
                'architectures': test_df[architecture_col].value_counts().to_dict(),
            },
        }
        
        # Check for architecture overlap
        train_archs = set(train_df[train_df[label_col] == 1][architecture_col].unique())
        test_archs = set(test_df[test_df[label_col] == 1][architecture_col].unique())
        summary['overlap'] = list(train_archs & test_archs)
        
        return summary


class BalancedSampler:
    """Handles balanced sampling across classes and architectures."""
    
    def __init__(
        self,
        samples_per_class: Optional[int] = None,
        balance_architectures: bool = True,
        seed: int = 42,
    ):
        self.samples_per_class = samples_per_class
        self.balance_architectures = balance_architectures
        self.seed = seed
    
    def sample(
        self,
        df: pd.DataFrame,
        label_col: str = 'label',
        architecture_col: str = 'architecture',
    ) -> pd.DataFrame:
        """Sample from dataframe with class balancing.
        
        Args:
            df: Input dataframe
            label_col: Column containing labels
            architecture_col: Column containing architecture names
            
        Returns:
            Sampled dataframe
        """
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Group by label
        real_df = df[df[label_col] == 0]
        fake_df = df[df[label_col] == 1]
        
        # Determine target count per class
        if self.samples_per_class is not None:
            target_per_class = self.samples_per_class
        else:
            # Use minimum class count for balance
            target_per_class = min(len(real_df), len(fake_df))
        
        # Sample real images
        if len(real_df) <= target_per_class:
            sampled_real = real_df
        else:
            sampled_real = real_df.sample(n=target_per_class, random_state=self.seed)
        
        # Sample fake images (optionally balanced across architectures)
        if self.balance_architectures:
            sampled_fake = self._sample_balanced_architectures(
                fake_df, target_per_class, architecture_col
            )
        else:
            if len(fake_df) <= target_per_class:
                sampled_fake = fake_df
            else:
                sampled_fake = fake_df.sample(n=target_per_class, random_state=self.seed)
        
        # Combine and shuffle
        result = pd.concat([sampled_real, sampled_fake], ignore_index=True)
        result = result.sample(frac=1, random_state=self.seed).reset_index(drop=True)
        
        return result
    
    def _sample_balanced_architectures(
        self,
        df: pd.DataFrame,
        target_count: int,
        architecture_col: str,
    ) -> pd.DataFrame:
        """Sample with balanced representation across architectures."""
        architectures = df[architecture_col].unique()
        n_archs = len(architectures)
        
        if n_archs == 0:
            return df.iloc[:0]
        
        # Target samples per architecture
        per_arch = target_count // n_archs
        remainder = target_count % n_archs
        
        samples = []
        for i, arch in enumerate(architectures):
            arch_df = df[df[architecture_col] == arch]
            n_samples = per_arch + (1 if i < remainder else 0)
            
            if len(arch_df) <= n_samples:
                samples.append(arch_df)
            else:
                samples.append(arch_df.sample(n=n_samples, random_state=self.seed + i))
        
        return pd.concat(samples, ignore_index=True)


def load_and_combine_metadata(
    root_dir: str,
    dataset_configs: List[Dict],
) -> pd.DataFrame:
    """Load and combine metadata from multiple datasets.
    
    Args:
        root_dir: Root directory for processed data
        dataset_configs: List of dataset configuration dicts with
                        'name', 'metadata', and 'architecture_type' keys
    
    Returns:
        Combined DataFrame with unified schema
    """
    root = Path(root_dir)
    dfs = []
    
    for config in dataset_configs:
        metadata_path = root / config['metadata']
        if not metadata_path.exists():
            print(f"Warning: Metadata not found: {metadata_path}")
            continue
        
        df = pd.read_csv(metadata_path)
        df['dataset_name'] = config['name']
        df['architecture_type'] = df['architecture'].apply(get_architecture_type)
        
        # Ensure filepath is absolute/relative to root
        if 'filepath' in df.columns:
            df['full_path'] = df['filepath'].apply(
                lambda x: str(root / x) if not x.startswith(str(root)) else x
            )
        
        dfs.append(df)
    
    if not dfs:
        raise ValueError("No datasets loaded!")
    
    combined = pd.concat(dfs, ignore_index=True)
    return combined
