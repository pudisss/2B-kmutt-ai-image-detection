"""Multi-domain fusion modules.

This module implements various fusion strategies for combining
features from RGB, frequency, and noise domain branches.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple
import math


class ConcatFusion(nn.Module):
    """Simple concatenation fusion with optional projection."""
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: Optional[int] = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        total_dim = sum(input_dims)
        self.output_dim = hidden_dim or total_dim
        
        if hidden_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(total_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.projection = nn.Identity()
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors [B, dim_i]
        
        Returns:
            Fused features [B, output_dim]
        """
        concat = torch.cat(features, dim=1)
        return self.projection(concat)


class AttentionFusion(nn.Module):
    """Attention-based fusion using multi-head self-attention.
    
    Treats each domain's features as a token and applies
    self-attention to learn cross-domain relationships.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_domains = len(input_dims)
        
        # Project each domain to common dimension
        self.domain_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Learnable domain embeddings
        self.domain_embeddings = nn.Parameter(
            torch.randn(self.num_domains, hidden_dim) * 0.02
        )
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * self.num_domains, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors [B, dim_i]
        
        Returns:
            Fused features [B, hidden_dim]
        """
        batch_size = features[0].shape[0]
        
        # Project to common dimension and add domain embeddings
        tokens = []
        for i, (feat, proj) in enumerate(zip(features, self.domain_projections)):
            projected = proj(feat)  # [B, hidden_dim]
            projected = projected + self.domain_embeddings[i]
            tokens.append(projected)
        
        # Stack as sequence [B, num_domains, hidden_dim]
        tokens = torch.stack(tokens, dim=1)
        
        # Apply transformer
        attended = self.transformer(tokens)  # [B, num_domains, hidden_dim]
        
        # Flatten and project
        flat = attended.reshape(batch_size, -1)  # [B, num_domains * hidden_dim]
        output = self.output_projection(flat)
        
        return output


class GatedFusion(nn.Module):
    """Gated fusion with learnable domain importance.
    
    Uses gating mechanism to dynamically weight each domain's
    contribution based on input-dependent importance scores.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 512,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_domains = len(input_dims)
        self.hidden_dim = hidden_dim
        
        # Project each domain to common dimension
        self.domain_projections = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
            )
            for dim in input_dims
        ])
        
        # Gating network
        total_dim = sum(input_dims)
        self.gate = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, self.num_domains),
            nn.Softmax(dim=1),
        )
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors [B, dim_i]
        
        Returns:
            Fused features [B, hidden_dim]
        """
        # Compute gating weights from concatenated features
        concat = torch.cat(features, dim=1)
        gates = self.gate(concat)  # [B, num_domains]
        
        # Project and weight each domain
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.domain_projections)):
            proj_feat = proj(feat)  # [B, hidden_dim]
            weighted = proj_feat * gates[:, i:i+1]  # [B, hidden_dim]
            projected.append(weighted)
        
        # Sum weighted features
        fused = sum(projected)
        output = self.output_projection(fused)
        
        return output


class BilinearFusion(nn.Module):
    """Bilinear fusion for capturing pairwise domain interactions.
    
    Computes bilinear interactions between domain pairs to capture
    second-order relationships.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 512,
        rank: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_domains = len(input_dims)
        self.hidden_dim = hidden_dim
        self.rank = rank
        
        # Project each domain
        self.domain_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Low-rank bilinear interactions for each domain pair
        num_pairs = self.num_domains * (self.num_domains - 1) // 2
        
        # Use factorized bilinear: W_1^T x * W_2^T y
        self.bilinear_w1 = nn.Parameter(torch.randn(num_pairs, hidden_dim, rank) * 0.02)
        self.bilinear_w2 = nn.Parameter(torch.randn(num_pairs, hidden_dim, rank) * 0.02)
        
        # Output combination
        interaction_dim = num_pairs * rank
        linear_dim = hidden_dim * self.num_domains
        
        self.output_projection = nn.Sequential(
            nn.Linear(interaction_dim + linear_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors [B, dim_i]
        
        Returns:
            Fused features [B, hidden_dim]
        """
        batch_size = features[0].shape[0]
        
        # Project features
        projected = [proj(feat) for proj, feat in zip(self.domain_projections, features)]
        
        # Compute bilinear interactions
        interactions = []
        pair_idx = 0
        
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                # Factorized bilinear: (x^T W1) * (y^T W2)
                x_proj = torch.matmul(projected[i], self.bilinear_w1[pair_idx])  # [B, rank]
                y_proj = torch.matmul(projected[j], self.bilinear_w2[pair_idx])  # [B, rank]
                interaction = x_proj * y_proj  # [B, rank]
                interactions.append(interaction)
                pair_idx += 1
        
        # Concatenate interactions and linear features
        interaction_concat = torch.cat(interactions, dim=1)  # [B, num_pairs * rank]
        linear_concat = torch.cat(projected, dim=1)  # [B, num_domains * hidden_dim]
        
        combined = torch.cat([interaction_concat, linear_concat], dim=1)
        output = self.output_projection(combined)
        
        return output


class CrossAttentionFusion(nn.Module):
    """Cross-attention fusion where each domain attends to others.
    
    Each domain serves as query to attend to the other domains,
    allowing for asymmetric information flow.
    """
    
    def __init__(
        self,
        input_dims: List[int],
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.num_domains = len(input_dims)
        self.hidden_dim = hidden_dim
        
        # Project each domain
        self.domain_projections = nn.ModuleList([
            nn.Linear(dim, hidden_dim) for dim in input_dims
        ])
        
        # Cross-attention for each domain
        self.cross_attention = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(self.num_domains)
        ])
        
        # Layer norms
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(self.num_domains)
        ])
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * self.num_domains, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features: List of feature tensors [B, dim_i]
        
        Returns:
            Fused features [B, hidden_dim]
        """
        batch_size = features[0].shape[0]
        
        # Project features
        projected = [proj(feat).unsqueeze(1) for proj, feat in 
                    zip(self.domain_projections, features)]  # Each [B, 1, hidden_dim]
        
        # Stack key-value tokens (all domains)
        kv = torch.cat(projected, dim=1)  # [B, num_domains, hidden_dim]
        
        # Cross-attention for each domain as query
        attended = []
        for i, (q, attn, ln) in enumerate(zip(projected, self.cross_attention, self.layer_norms)):
            # Exclude self from key-value
            kv_others = torch.cat([projected[j] for j in range(self.num_domains) if j != i], dim=1)
            
            # Cross-attention
            attn_out, _ = attn(q, kv_others, kv_others)  # [B, 1, hidden_dim]
            
            # Residual and norm
            out = ln(q + attn_out)  # [B, 1, hidden_dim]
            attended.append(out.squeeze(1))  # [B, hidden_dim]
        
        # Concatenate and project
        concat = torch.cat(attended, dim=1)  # [B, num_domains * hidden_dim]
        output = self.output_projection(concat)
        
        return output


def create_fusion_module(
    fusion_type: str,
    input_dims: List[int],
    hidden_dim: int = 512,
    num_heads: int = 8,
    dropout: float = 0.3,
) -> nn.Module:
    """Factory function to create fusion module.
    
    Args:
        fusion_type: One of 'concat', 'attention', 'gated', 'bilinear', 'cross_attention'
        input_dims: List of input dimensions from each branch
        hidden_dim: Output dimension
        num_heads: Number of attention heads (for attention-based fusion)
        dropout: Dropout rate
    
    Returns:
        Fusion module
    """
    if fusion_type == 'concat':
        return ConcatFusion(input_dims, hidden_dim, dropout)
    elif fusion_type == 'attention':
        return AttentionFusion(input_dims, hidden_dim, num_heads, dropout=dropout)
    elif fusion_type == 'gated':
        return GatedFusion(input_dims, hidden_dim, dropout)
    elif fusion_type == 'bilinear':
        return BilinearFusion(input_dims, hidden_dim, dropout=dropout)
    elif fusion_type == 'cross_attention':
        return CrossAttentionFusion(input_dims, hidden_dim, num_heads, dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")
