"""RCA Layers."""

from .ssm import SelectiveStateSpaceModel, SimpleStateSpaceModel
from .attention import EfficientAttention
from .gla import GatedLinearAttention
from .sliding_attention import SlidingWindowAttention
from .scan import parallel_scan_linear, compute_parallel_scan, TRITON_AVAILABLE
from .norm import RMSNorm, DeepNorm
from .positions import ALiBiPositionEmbedding, RotaryPositionEmbedding
# RCA-Mythos v3.0 layers
from .lti_injection import LTIInjection
from .act_halting import ACTHalting
from .loop_embedding import LoopIndexEmbedding
from .lora_depth import DepthLoRAAdapter

__all__ = [
    # SSM
    "SelectiveStateSpaceModel",
    "SimpleStateSpaceModel",
    # Attention
    "EfficientAttention",
    "GatedLinearAttention",
    "SlidingWindowAttention",
    # Scan
    "parallel_scan_linear",
    "compute_parallel_scan",
    "TRITON_AVAILABLE",
    # Norm
    "RMSNorm",
    "DeepNorm",
    # Positional
    "ALiBiPositionEmbedding",
    "RotaryPositionEmbedding",
    # RCA-Mythos v3.0
    "LTIInjection",
    "ACTHalting",
    "LoopIndexEmbedding",
    "DepthLoRAAdapter",
]
