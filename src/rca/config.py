"""
RCA Configuration
=================

Model configuration with presets and validation.

Author: Rajaaditya.R
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import json


@dataclass
class RCAConfig:
    """
    Configuration for RCA models.

    Supports preset sizes and full customization.
    """

    # Core dimensions
    vocab_size: int = 50257
    state_dim: int = 768
    n_layers: int = 12
    ssm_expand: int = 2

    # Attention
    n_heads: int = 12
    num_attention_layers: int = 4
    attention_every_n: int = 3
    use_hybrid_attention: bool = True

    # SSM options
    use_selective_scan: bool = True
    use_full_matrix: bool = False

    # Regularization
    dropout: float = 0.1

    # Positional encoding
    use_alibi: bool = True
    use_rotary: bool = True
    alibi_learnable: bool = True
    max_seq_len: int = 8192

    # Training
    initializer_range: float = 0.02
    tie_word_embeddings: bool = False

    # Misc
    pad_token_id: Optional[int] = None
    bos_token_id: int = 1
    eos_token_id: int = 2

    # Ultra-Reasoning Architecture
    use_ultra_reasoning: bool = False
    use_glu_ffn: bool = False
    gla_heads: int = 8
    gla_expand_k: float = 1.0
    gla_expand_v: float = 2.0
    sliding_window_size: int = 512
    num_memory_tokens: int = 32
    ssm_zone_end: float = 0.6
    gla_zone_end: float = 0.85
    use_mqa: bool = False

    # Performance / Training Optimization
    gradient_checkpointing: bool = False
    use_torch_compile: bool = False
    compile_mode: str = "reduce-overhead"

    # =========================================================================
    # RCA-Mythos (v3.0) — Recurrent-Depth Architecture
    # =========================================================================
    # Set use_recurrent_depth=True to use RCAMythosModel instead of RCAModel.

    use_recurrent_depth: bool = False

    # Prelude: SSM blocks run once before the recurrent loop.
    # These encode the input sequence into rich representation e.
    mythos_prelude_layers: int = 4

    # Coda: Reasoning blocks (SlidingWindow + MemTokens) run once after loop.
    # These do the final precision pass on the loop-refined hidden state.
    mythos_coda_layers: int = 2

    # Recurrent Core: maximum T at training/inference.
    # - Training: T ~ Uniform(1, max_loops) per batch step [Parcae optimal]
    # - Inference: T = max_loops (or set explicitly via n_loops)
    # Parcae scaling law: optimal μ_rec ∝ C^0.40 (FLOP budget C)
    # Rule of thumb: 4 loops for 100M, 8 for 500M, 16 for 1B+
    mythos_max_loops: int = 8

    # LoRA depth adapter rank. Parameter overhead ≈ 2×dim×rank + max_loops×rank.
    # Recommended: rank=16 for dim≤1024, rank=32 for dim≥2048.
    mythos_lora_rank: int = 16

    # ACT halting threshold. Positions with cumulative_p ≥ threshold stop looping.
    # 0.99 is the Graves (2016) recommendation. Lower = fewer loops on average.
    mythos_act_threshold: float = 0.99

    # Fraction of hidden channels receiving loop index embedding signal.
    # 0.125 = dim // 8 channels. More channels → stronger depth signal,
    # but disrupts more of the existing feature space.
    mythos_loop_embed_fraction: float = 0.125

    # Random-loop training (Parcae): sample T ~ Uniform(1, max_loops) each step.
    # This is the KEY to stable training and depth extrapolation at inference.
    # Set False only for debugging or fixed-depth ablations.
    mythos_random_loop_training: bool = True

    def __post_init__(self):
        """Validate config after initialization."""
        assert self.state_dim > 0, "state_dim must be positive"
        assert self.n_layers > 0, "n_layers must be positive"
        assert self.n_heads > 0, "n_heads must be positive"
        assert self.state_dim % self.n_heads == 0, (
            f"state_dim ({self.state_dim}) must be divisible by n_heads ({self.n_heads})"
        )
        assert 0 <= self.dropout < 1, "dropout must be in [0, 1)"

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "RCAConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def from_json(cls, path: str) -> "RCAConfig":
        with open(path, "r") as f:
            return cls.from_dict(json.load(f))

    # =========================================================================
    # Legacy Presets (kept for backward compat)
    # =========================================================================

    @classmethod
    def rca_tiny(cls) -> "RCAConfig":
        """~10M params — for testing and debugging."""
        return cls(
            state_dim=256,
            n_layers=4,
            n_heads=4,
            ssm_expand=2,
            num_attention_layers=1,
            attention_every_n=4,
            dropout=0.1,
        )

    @classmethod
    def rca_small(cls) -> "RCAConfig":
        """~50M params — lightweight model."""
        return cls(
            state_dim=512,
            n_layers=8,
            n_heads=8,
            ssm_expand=2,
            num_attention_layers=2,
            attention_every_n=4,
            dropout=0.1,
        )

    @classmethod
    def rca_base(cls) -> "RCAConfig":
        """~125M params — standard model."""
        return cls(
            state_dim=768,
            n_layers=12,
            n_heads=12,
            ssm_expand=2,
            num_attention_layers=4,
            attention_every_n=3,
            dropout=0.1,
        )

    @classmethod
    def rca_large(cls) -> "RCAConfig":
        """~350M params — high capacity."""
        return cls(
            state_dim=1024,
            n_layers=24,
            n_heads=16,
            ssm_expand=2,
            num_attention_layers=8,
            attention_every_n=3,
            dropout=0.1,
        )

    @classmethod
    def rca_xl(cls) -> "RCAConfig":
        """~1B params — extra-large model."""
        return cls(
            state_dim=1280,
            n_layers=36,
            n_heads=20,
            ssm_expand=2,
            num_attention_layers=12,
            attention_every_n=3,
            dropout=0.05,
        )

    @classmethod
    def rca_ultra(cls) -> "RCAConfig":
        """~300M params — ultra-reasoning architecture.

        Brain Analogy:
        - Layers 1–19 (60%): SSM backbone (stream of consciousness)
        - Layers 20–27 (25%): GLA (working memory)
        - Layers 28–32 (15%): Sliding Window + Memory Tokens (focus)
        - All layers: GLU-FFN (active neurons / world knowledge)
        """
        return cls(
            state_dim=768,
            n_layers=32,
            n_heads=12,
            ssm_expand=2,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.1,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=12,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=False,
        )

    # =========================================================================
    # Production Presets — Ultra-Reasoning with seq_len=4096
    # =========================================================================

    @classmethod
    def rca_100m(cls) -> "RCAConfig":
        """~100M params — Ultra-Reasoning, fits easily on T4/P100.

        Training budget (7 hrs):
          T4:   ~800M tokens   (batch=8,  grad_accum=4,  fp16)
          P100: ~1.2B tokens   (batch=8,  grad_accum=4,  fp16)
        """
        return cls(
            state_dim=512,
            n_layers=12,
            n_heads=8,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.1,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=8,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=False,
            gradient_checkpointing=False,
        )

    @classmethod
    def rca_500m(cls) -> "RCAConfig":
        """~500M params — Ultra-Reasoning, T4/P100 with checkpointing.

        Training budget (7 hrs):
          T4:   ~300M tokens   (batch=2, grad_accum=16, fp16, grad_ckpt)
          P100: ~500M tokens   (batch=2, grad_accum=16, fp16, grad_ckpt)
        """
        return cls(
            state_dim=1024,
            n_layers=20,
            n_heads=16,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.1,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=16,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=False,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_1b(cls) -> "RCAConfig":
        """~1B params — Ultra-Reasoning, T4/P100 with aggressive checkpointing.

        Training budget (7 hrs):
          T4:   ~150M tokens   (batch=1, grad_accum=32, fp16, grad_ckpt)
          P100: ~250M tokens   (batch=1, grad_accum=32, fp16, grad_ckpt)
        """
        return cls(
            state_dim=1280,
            n_layers=28,
            n_heads=20,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=20,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_5b(cls) -> "RCAConfig":
        """~5B params — Ultra-Reasoning, requires multi-GPU (FSDP).

        Training budget (1T tokens):
          4×A100 80GB: ~21 days  (FSDP, bf16)
        """
        return cls(
            state_dim=2048,
            n_layers=48,
            n_heads=32,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=32,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=64,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_10b(cls) -> "RCAConfig":
        """~10B params — Ultra-Reasoning, requires 8×A100 (FSDP).

        Training budget (1T tokens):
          8×A100 80GB: ~30 days  (FSDP, bf16)
        """
        return cls(
            state_dim=3072,
            n_layers=48,
            n_heads=32,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=32,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=64,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_100b(cls) -> "RCAConfig":
        """~100B params — Ultra-Reasoning, requires large cluster or TPU pod.

        Training budget (1T tokens):
          TPU v4-256 pod: ~45 days  (XLA FSDP)
          64×A100 80GB:  ~45 days  (FSDP, bf16)
        """
        return cls(
            state_dim=7680,
            n_layers=80,
            n_heads=64,
            ssm_expand=2,
            max_seq_len=4096,
            num_attention_layers=0,
            attention_every_n=0,
            use_hybrid_attention=False,
            dropout=0.05,
            use_ultra_reasoning=True,
            use_glu_ffn=True,
            gla_heads=64,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=128,
            ssm_zone_end=0.6,
            gla_zone_end=0.85,
            use_mqa=True,
            gradient_checkpointing=True,
        )

    # =========================================================================
    # RCA-Mythos Presets — Recurrent-Depth Architecture (v3.0)
    # =========================================================================
    # Loop counts follow Parcae scaling law: μ_rec ∝ C^0.40
    # A 770M-param mythos model ≈ 1.3B flat-depth model quality.
    #
    # Architecture: SSM Prelude (once) → GLA Core (×T loops) → Reasoning Coda (once)
    # =========================================================================

    @classmethod
    def rca_mythos_100m(cls) -> "RCAConfig":
        """~100M params — RCA-Mythos, fits T4/P100 with room to spare.

        Effective depth: 4 SSM (prelude) + 4 loops × 1 GLA + 2 Reasoning (coda)
        Parcae equivalent: ≈ 130M flat model quality

        Training budget (7 hrs):
          T4:   ~700M tokens   (batch=8, grad_accum=4, fp16)
          P100: ~1.1B tokens   (batch=8, grad_accum=4, fp16)

        Optimal μ_rec = 4 loops (Parcae C^0.40 at ~100M FLOP budget).
        """
        return cls(
            # Core dimensions
            state_dim=512,
            n_layers=6,
            n_heads=8,
            ssm_expand=2,
            max_seq_len=4096,
            dropout=0.1,
            # Attention (used in ReasoningBlock coda)
            use_hybrid_attention=False,
            num_attention_layers=0,
            attention_every_n=0,
            # GLA (recurrent core)
            gla_heads=8,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            # Reasoning coda
            sliding_window_size=512,
            num_memory_tokens=32,
            use_mqa=False,
            # Ultra-reasoning zone (unused; mythos uses its own zones)
            use_ultra_reasoning=False,
            use_glu_ffn=True,
            # SSM
            use_selective_scan=True,
            use_full_matrix=False,
            # Mythos v3.0
            use_recurrent_depth=True,
            mythos_prelude_layers=4,
            mythos_coda_layers=2,
            mythos_max_loops=4,          # μ_rec=4 for ~100M budget
            mythos_lora_rank=16,
            mythos_act_threshold=0.99,
            mythos_loop_embed_fraction=0.125,
            mythos_random_loop_training=True,
            gradient_checkpointing=False,
        )

    @classmethod
    def rca_mythos_500m(cls) -> "RCAConfig":
        """~500M params — RCA-Mythos, T4/P100 with gradient checkpointing.

        Effective depth: 6 SSM + 8 loops × 1 GLA + 3 Reasoning
        Parcae equivalent: ≈ 700M flat model quality

        Training budget (7 hrs):
          T4:   ~250M tokens   (batch=2, grad_accum=16, fp16, grad_ckpt)
          P100: ~400M tokens   (batch=2, grad_accum=16, fp16, grad_ckpt)

        Optimal μ_rec = 8 loops (Parcae C^0.40 at ~500M FLOP budget).
        """
        return cls(
            state_dim=1024,
            n_layers=10,
            n_heads=16,
            ssm_expand=2,
            max_seq_len=4096,
            dropout=0.1,
            use_hybrid_attention=False,
            num_attention_layers=0,
            attention_every_n=0,
            gla_heads=16,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=32,
            use_mqa=False,
            use_ultra_reasoning=False,
            use_glu_ffn=True,
            use_selective_scan=True,
            use_full_matrix=False,
            use_recurrent_depth=True,
            mythos_prelude_layers=6,
            mythos_coda_layers=3,
            mythos_max_loops=8,          # μ_rec=8 for ~500M budget
            mythos_lora_rank=16,
            mythos_act_threshold=0.99,
            mythos_loop_embed_fraction=0.125,
            mythos_random_loop_training=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_mythos_1b(cls) -> "RCAConfig":
        """~1B params — RCA-Mythos, requires A100 or T4 with aggressive ckpt.

        Effective depth: 8 SSM + 12 loops × 1 GLA + 4 Reasoning
        Parcae equivalent: ≈ 1.5B flat model quality

        Training budget (7 hrs):
          T4:   ~100M tokens   (batch=1, grad_accum=32, fp16, grad_ckpt)
          A100: ~800M tokens   (batch=4, grad_accum=8, bf16, grad_ckpt)

        Optimal μ_rec = 12 loops (Parcae C^0.40 at ~1B FLOP budget).
        """
        return cls(
            state_dim=1280,
            n_layers=14,
            n_heads=20,
            ssm_expand=2,
            max_seq_len=4096,
            dropout=0.05,
            use_hybrid_attention=False,
            num_attention_layers=0,
            attention_every_n=0,
            gla_heads=20,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=512,
            num_memory_tokens=64,
            use_mqa=True,
            use_ultra_reasoning=False,
            use_glu_ffn=True,
            use_selective_scan=True,
            use_full_matrix=False,
            use_recurrent_depth=True,
            mythos_prelude_layers=8,
            mythos_coda_layers=4,
            mythos_max_loops=12,         # μ_rec=12 for ~1B budget
            mythos_lora_rank=32,
            mythos_act_threshold=0.99,
            mythos_loop_embed_fraction=0.125,
            mythos_random_loop_training=True,
            gradient_checkpointing=True,
        )

    @classmethod
    def rca_mythos_3b(cls) -> "RCAConfig":
        """~3B params — RCA-Mythos, requires multi-GPU or single A100 80GB.

        Effective depth: 10 SSM + 16 loops × 1 GLA + 6 Reasoning
        Parcae equivalent: ≈ 5B flat model quality

        Training budget (1T tokens):
          2×A100 80GB: ~18 days  (FSDP, bf16, grad_ckpt)
          4×A100 40GB: ~20 days  (FSDP, bf16, grad_ckpt)

        Optimal μ_rec = 16 loops (Parcae C^0.40 at ~3B FLOP budget).
        """
        return cls(
            state_dim=2048,
            n_layers=22,
            n_heads=32,
            ssm_expand=2,
            max_seq_len=8192,
            dropout=0.05,
            use_hybrid_attention=False,
            num_attention_layers=0,
            attention_every_n=0,
            gla_heads=32,
            gla_expand_k=1.0,
            gla_expand_v=2.0,
            sliding_window_size=1024,
            num_memory_tokens=64,
            use_mqa=True,
            use_ultra_reasoning=False,
            use_glu_ffn=True,
            use_selective_scan=True,
            use_full_matrix=False,
            use_recurrent_depth=True,
            mythos_prelude_layers=10,
            mythos_coda_layers=6,
            mythos_max_loops=16,         # μ_rec=16 for ~3B budget
            mythos_lora_rank=32,
            mythos_act_threshold=0.99,
            mythos_loop_embed_fraction=0.125,
            mythos_random_loop_training=True,
            gradient_checkpointing=True,
        )
