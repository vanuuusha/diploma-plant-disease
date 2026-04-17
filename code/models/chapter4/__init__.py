"""Chapter 4: CGFM (Context-Gated Feature Modulation) and baselines."""
from .film_layer import FiLMLayer
from .context_encoder import ContextEncoder
from .se_block import SEBlock
from .cbam_block import CBAMBlock
from .late_fusion_head import LateFusionClassifier

__all__ = [
    "FiLMLayer",
    "ContextEncoder",
    "SEBlock",
    "CBAMBlock",
    "LateFusionClassifier",
]
