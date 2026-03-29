"""Model architectures for AI image detection."""

from .branches import (
    RGBBranch,
    FrequencyBranch,
    NoiseBranch,
    BranchEnsemble,
    create_branches,
)
from .fusion import (
    ConcatFusion,
    AttentionFusion,
    GatedFusion,
    BilinearFusion,
    CrossAttentionFusion,
    create_fusion_module,
)
from .detector import (
    MultiDomainDetector,
    SingleBranchDetector,
    ClassificationHead,
    create_detector,
    load_detector,
)

__all__ = [
    'RGBBranch',
    'FrequencyBranch',
    'NoiseBranch',
    'BranchEnsemble',
    'create_branches',
    'ConcatFusion',
    'AttentionFusion',
    'GatedFusion',
    'BilinearFusion',
    'CrossAttentionFusion',
    'create_fusion_module',
    'MultiDomainDetector',
    'SingleBranchDetector',
    'ClassificationHead',
    'create_detector',
    'load_detector',
]
