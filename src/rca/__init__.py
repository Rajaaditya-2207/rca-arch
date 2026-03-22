"""
RCA — Recursive Compression Architecture v2.0
==============================================

Ultra-Reasoning Architecture combining Mamba SSM, Gated Linear
Attention (GLA), and Sliding Window Attention across specialized
cognitive zones — with Triton/XLA-accelerated parallel scan.

Author: Rajaaditya.R
Contact: rajaaditya.aadhi@gmail.com

Quick start::

    from rca import RCAModel, RCAConfig

    config = RCAConfig.rca_100m()
    model = RCAModel(config)

    print(f"Parameters: {model.count_parameters():,}")
"""
"""
RCA (Recurrent Cross Attention) Architecture
A hybrid sequence modeling architecture combining sliding window attention, 
linear recurrent mechanisms (GLA/SSM), and long-context cross-attention.
"""

__version__ = "1.0.2"
__author__ = "Rajaaditya.R"
__email__ = "rajaaditya.aadhi@gmail.com"

from .config import RCAConfig
from .modeling.rca_model import RCAModel, RCAForCausalLM
from .modeling.outputs import CausalLMOutput, ModelOutput, BaseModelOutput
from .trainer import RCATrainer, TrainingArguments
from .generator import RCAGenerator
from .utils.benchmark import RCABenchmark
from .utils.export import export_to_onnx, save_pretrained, load_pretrained
from .converter import (
    export_safetensors, load_safetensors,
    export_gguf,
)
from .layers.scan import compute_parallel_scan, parallel_scan_linear, TRITON_AVAILABLE
from .layers.ssm import SelectiveStateSpaceModel, SimpleStateSpaceModel
from .layers.attention import EfficientAttention
from .layers.gla import GatedLinearAttention
from .layers.sliding_attention import SlidingWindowAttention
from .layers.norm import RMSNorm, DeepNorm
from .layers.positions import ALiBiPositionEmbedding, RotaryPositionEmbedding

__all__ = [
    # Config
    "RCAConfig",
    # Models
    "RCAModel",
    "RCAForCausalLM",
    # Outputs
    "CausalLMOutput",
    "ModelOutput",
    "BaseModelOutput",
    # Training
    "RCATrainer",
    "TrainingArguments",
    # Generation
    "RCAGenerator",
    # Converter / Export
    "export_safetensors",
    "load_safetensors",
    "export_gguf",
    # Utilities
    "RCABenchmark",
    "export_to_onnx",
    "save_pretrained",
    "load_pretrained",
    # Layers (advanced)
    "compute_parallel_scan",
    "parallel_scan_linear",
    "TRITON_AVAILABLE",
    "SelectiveStateSpaceModel",
    "SimpleStateSpaceModel",
    "EfficientAttention",
    "GatedLinearAttention",
    "SlidingWindowAttention",
    "RMSNorm",
    "DeepNorm",
    "ALiBiPositionEmbedding",
    "RotaryPositionEmbedding",
]
