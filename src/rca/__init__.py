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

__version__ = "1.3.0"
__author__ = "Rajaaditya.R"
__email__ = "rajaaditya.aadhi@gmail.com"

from .config import RCAConfig
from .modeling.rca_model import RCAModel, RCAForCausalLM
from .modeling.rca_mythos_model import RCAMythosModel, RCAMythosForCausalLM
from .modeling.recurrent_core import RecurrentCore
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
# RCA-Mythos v3.0 layers
from .layers.lti_injection import LTIInjection
from .layers.act_halting import ACTHalting
from .layers.loop_embedding import LoopIndexEmbedding
from .layers.lora_depth import DepthLoRAAdapter

__all__ = [
    # Config
    "RCAConfig",
    # ── v2.0 Models ─────────────────────────────────────────────────────────
    "RCAModel",
    "RCAForCausalLM",
    # ── v3.0 Mythos Models ──────────────────────────────────────────────────
    "RCAMythosModel",
    "RCAMythosForCausalLM",
    "RecurrentCore",
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
    # ── Core Layers (v2.0) ───────────────────────────────────────────────────
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
    # ── Mythos Layers (v3.0) ─────────────────────────────────────────────────
    "LTIInjection",
    "ACTHalting",
    "LoopIndexEmbedding",
    "DepthLoRAAdapter",
]
