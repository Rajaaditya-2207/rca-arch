"""
RCA-Mythos Model
================

RCA v3.0 — Hybrid Recurrent-Depth Architecture combining:
    - RCA v2.0 brain-analogy zones (SSM / GLA / Reasoning)
    - OpenMythos Recurrent-Depth Transformer innovations (Parcae 2026)

Architecture (3-stage Prelude → Loop → Coda):

    Input Token IDs
          ↓
    [Embedding]
          ↓
    ╔══════════════════════════════════════╗
    ║  PRELUDE  (SSM Zone — run once)      ║   Stream of consciousness
    ║  MambaMixBlocks × prelude_layers     ║   Encodes input → e, h₀
    ╚══════════════════════════════════════╝
          ↓  e = prelude output (saved for injection)
    ╔══════════════════════════════════════╗
    ║  RECURRENT CORE (GLA Zone — T loops) ║   Working memory × depth
    ║  Single GLA block, looped T times    ║
    ║  Per step:                           ║
    ║    h = loop_embed(h, t)              ║   Depth-aware representations
    ║    gla_out = GLA(norm(h + e))        ║   Input injection every loop
    ║    gla_out += LoRA(gla_out, t)       ║   Per-loop specialization
    ║    h = A·h + B·e + gla_out          ║   LTI-stable update ρ(A)<1
    ║    p_halt = ACT(h)                   ║   Adaptive early exit
    ╚══════════════════════════════════════╝
          ↓
    ╔══════════════════════════════════════╗
    ║  CODA  (Reasoning Zone — run once)   ║   Focus / precision
    ║  ReasoningBlocks × coda_layers       ║   SlidingWindow + MemTokens
    ╚══════════════════════════════════════╝
          ↓
    [Final Norm → LM Head]

Why this wins:
  - SSM Prelude: O(1) memory, captures long-range context
  - GLA Loop:    linear-time per loop, O(T·L) total, associative memory
  - Reasoning Coda: precise local attention with global memory token bookmarks
  - T loops × 1 GLA block ≈ quality of T×GLA stacked at 1× parameter cost
  - Depth extrapolation: train on T₁ loops, solve T₂ > T₁ hop problems at test
  - ACT: 2-3× throughput gain from per-position adaptive compute depth

Training loop count (Parcae optimal):
  - T ~ Uniform(1, max_loops) per batch step during training
  - Optimal μ_rec ∝ C^0.40 (where C = training FLOP budget)
  - Performance ceiling at inference = μ_rec used during training

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import os
from typing import Optional, Tuple, Dict, List

from ..config import RCAConfig
from ..layers.ssm import SelectiveStateSpaceModel, SimpleStateSpaceModel
from ..layers.norm import RMSNorm
from ..layers.sliding_attention import SlidingWindowAttention
from .outputs import CausalLMOutput
from .recurrent_core import RecurrentCore


# ===========================================================================
# GLU-FFN (reused from rca_model.py — kept inline for module independence)
# ===========================================================================

class GLUFFN(nn.Module):
    """SwiGLU feed-forward network. World knowledge storage."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = ((dim * expand * 2 // 3 + 63) // 64) * 64
        self.up_proj   = nn.Linear(dim, hidden, bias=False)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


# ===========================================================================
# Prelude Block (SSM — run once)
# ===========================================================================

class PreludeBlock(nn.Module):
    """
    SSM-based Prelude block.

    Identical to MambaMixBlock from RCA v2.0 — encodes the input sequence
    into a rich representation e that is then injected at every loop step.
    """

    def __init__(self, config: RCAConfig, layer_idx: int):
        super().__init__()
        d = config.state_dim
        expand_d = d * config.ssm_expand

        self.norm1    = RMSNorm(d)
        self.in_proj  = nn.Linear(d, expand_d, bias=False)
        self.gate_proj= nn.Linear(d, expand_d, bias=False)

        self.ssm = (
            SelectiveStateSpaceModel(expand_d, expand_d, use_full_matrix=config.use_full_matrix)
            if config.use_selective_scan
            else SimpleStateSpaceModel(expand_d, expand_d)
        )

        self.out_proj = nn.Linear(expand_d, d, bias=False)
        self.ffn_norm = RMSNorm(d)
        self.ffn      = GLUFFN(d, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_cuda: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x
        h = self.norm1(x)
        z = self.in_proj(h)
        gate = torch.sigmoid(self.gate_proj(h))
        z = z * gate

        if use_cache and z.size(1) == 1:
            ssm_out, new_state = self.ssm.forward_sequential(z.squeeze(1), ssm_state)
            ssm_out = ssm_out.unsqueeze(1)
        else:
            ssm_out = self.ssm.forward_parallel(z, ssm_state, use_cuda=use_cuda)
            new_state = None

        x = residual + self.out_proj(ssm_out)
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_state


# ===========================================================================
# Coda Block (Reasoning Zone — run once)
# ===========================================================================

class CodaBlock(nn.Module):
    """
    Reasoning Zone Coda block.

    Sliding Window Attention + Memory Tokens + GLU-FFN.
    Identical to ReasoningBlock from RCA v2.0 — provides precise local
    attention and global context via memory tokens after the recurrent loop.
    """

    def __init__(self, config: RCAConfig, layer_idx: int):
        super().__init__()
        d = config.state_dim
        self.norm1 = RMSNorm(d)
        self.attention = SlidingWindowAttention(
            dim=d,
            num_heads=config.n_heads,
            window_size=config.sliding_window_size,
            num_memory_tokens=config.num_memory_tokens,
            dropout=config.dropout,
            use_mqa=config.use_mqa,
        )
        self.ffn_norm = RMSNorm(d)
        self.ffn = GLUFFN(d, dropout=config.dropout)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_cuda: bool = True,
    ) -> Tuple[torch.Tensor, None]:
        h = self.norm1(x)
        x = x + self.attention(h, is_causal=True)
        x = x + self.ffn(self.ffn_norm(x))
        return x, None


# ===========================================================================
# RCA-Mythos Model
# ===========================================================================

class RCAMythosModel(nn.Module):
    """
    RCA v3.0 — Recurrent-Depth Architecture (RCA-Mythos).

    Three-stage Prelude → RecurrentCore → Coda architecture:
        1. Prelude:  SSM blocks encode input → e (run once)
        2. Core:     GLA looped T times with LTI injection (depth reasoning)
        3. Coda:     Reasoning blocks for local precision (run once)

    Key advantages over RCA v2.0:
        - Depth extrapolation: more inference loops → harder problems solved
        - Stable training: LTI injection guarantees ρ(A) < 1 (no loss spikes)
        - Adaptive compute: ACT halting (2-3× throughput at same quality)
        - Weight efficiency: k GLA params × T loops ≈ k×T flat GLA stacking

    Parameters
    ----------
    config : RCAConfig
        Must have use_recurrent_depth=True and the new mythos fields set.
        Use RCAConfig.rca_mythos_*() presets for validated configurations.
    """

    def __init__(self, config: RCAConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # ── Token embeddings ────────────────────────────────────────────
        self.embeddings   = nn.Embedding(config.vocab_size, config.state_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # ── Stage 1: Prelude (SSM zone, run once) ───────────────────────
        self.prelude = nn.ModuleList([
            PreludeBlock(config, i)
            for i in range(config.mythos_prelude_layers)
        ])

        # ── Stage 2: Recurrent Core (GLA looped T times) ────────────────
        self.recurrent_core = RecurrentCore(
            dim          = config.state_dim,
            num_heads    = config.gla_heads,
            max_loops    = config.mythos_max_loops,
            lora_rank    = config.mythos_lora_rank,
            act_threshold= config.mythos_act_threshold,
            loop_dim_fraction = config.mythos_loop_embed_fraction,
            expand_k     = config.gla_expand_k,
            expand_v     = config.gla_expand_v,
            dropout      = config.dropout,
            random_loop_training = config.mythos_random_loop_training,
        )

        # ── Stage 3: Coda (Reasoning zone, run once) ─────────────────────
        self.coda = nn.ModuleList([
            CodaBlock(config, i)
            for i in range(config.mythos_coda_layers)
        ])

        # ── Output ────────────────────────────────────────────────────────
        self.final_norm = RMSNorm(config.state_dim)
        self.lm_head    = nn.Linear(config.state_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

        # ── Weight init ───────────────────────────────────────────────────
        self.apply(self._init_weights)

    # ------------------------------------------------------------------
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    # ------------------------------------------------------------------
    def _prelude_forward(self, block, h, state, use_cache, use_cuda):
        """Helper for gradient checkpointing in the prelude."""
        return block(h, state, use_cache=use_cache, use_cuda=use_cuda)

    def _coda_forward(self, block, h, state, use_cache, use_cuda):
        """Helper for gradient checkpointing in the coda."""
        return block(h, state, use_cache=use_cache, use_cuda=use_cuda)

    # ------------------------------------------------------------------
    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ssm_states: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
        n_loops: Optional[int] = None,
        use_cache: bool = False,
        use_cuda: bool = True,
        act_loss_weight: float = 0.01,
    ) -> CausalLMOutput:
        """
        Forward pass through Prelude → RecurrentCore → Coda.

        Parameters
        ----------
        input_ids : torch.Tensor, shape (B, S)
        labels : torch.Tensor, shape (B, S), optional
            If provided, computes cross-entropy loss.
        ssm_states : tuple of tensors, optional
            Prelude SSM states for cached generation.
        n_loops : int, optional
            Number of recurrent loops. None = auto (Uniform training / max inference).
        use_cache : bool
            Return SSM states for autoregressive generation.
        use_cuda : bool
            Use Triton/CUDA kernels in SSM if available.
        act_loss_weight : float
            Weight for ACT ponder regularization loss. Set 0.0 to disable.

        Returns
        -------
        CausalLMOutput with loss, logits, last_hidden_state, ssm_states.
        """
        # ── Embed ─────────────────────────────────────────────────────────
        h = self.embed_dropout(self.embeddings(input_ids))

        # ── Stage 1: Prelude ──────────────────────────────────────────────
        new_ssm_states: List[Optional[torch.Tensor]] = []
        if ssm_states is None:
            ssm_states = [None] * len(self.prelude)

        for i, (block, state) in enumerate(zip(self.prelude, ssm_states)):
            if self.gradient_checkpointing and self.training and not use_cache:
                h, new_state = grad_checkpoint(
                    self._prelude_forward,
                    block, h, state, use_cache, use_cuda,
                    use_reentrant=False,
                )
            else:
                h, new_state = block(h, state, use_cache=use_cache, use_cuda=use_cuda)
            new_ssm_states.append(new_state)

        # Save encoded input e for injection throughout the loop
        e = h.detach() if not self.training else h  # keep grad in training

        # ── Stage 2: Recurrent Core ────────────────────────────────────────
        return_ponder = act_loss_weight > 0.0 and self.training
        h, ponder_loss = self.recurrent_core(
            h, e,
            n_loops=n_loops,
            return_ponder_loss=return_ponder,
        )

        # ── Stage 3: Coda ──────────────────────────────────────────────────
        for block in self.coda:
            if self.gradient_checkpointing and self.training:
                h, _ = grad_checkpoint(
                    self._coda_forward,
                    block, h, None, False, use_cuda,
                    use_reentrant=False,
                )
            else:
                h, _ = block(h)

        # ── Output ─────────────────────────────────────────────────────────
        h = self.final_norm(h)
        logits = self.lm_head(h)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            # Add ACT ponder regularization (encourages compute efficiency)
            if ponder_loss is not None:
                loss = loss + act_loss_weight * ponder_loss

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=h,
            ssm_states=tuple(new_ssm_states) if use_cache else None,
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
        n_loops: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Autoregressive generation with O(1) SSM memory and recurrent depth.

        n_loops at inference controls reasoning depth:
            - n_loops = config.mythos_max_loops: standard depth
            - n_loops > max_loops:  deeper (ACT trained to generalize)
            - n_loops = 1:          fastest, shallowest
        """
        self.eval()
        eos = eos_token_id or self.config.eos_token_id

        # Process prefix
        out = self.forward(input_ids, use_cache=True, n_loops=n_loops, act_loss_weight=0.0)
        logits = out.logits[:, -1, :]
        ssm_states = out.ssm_states

        generated = []
        for _ in range(max_new_tokens):
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_logits[cum_probs - F.softmax(sorted_logits, dim=-1) >= top_p] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            if (next_token == eos).all():
                break

            out = self.forward(
                next_token, ssm_states=ssm_states, use_cache=True,
                n_loops=n_loops, act_loss_weight=0.0,
            )
            logits = out.logits[:, -1, :]
            ssm_states = out.ssm_states

        return torch.cat([input_ids] + generated, dim=1)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_spectral_radius(self) -> float:
        """Monitor LTI stability — ρ(A) must stay < 1.0 during training."""
        return self.recurrent_core.get_spectral_radius()

    def get_architecture_summary(self) -> Dict:
        """Return a summary of the 3-stage architecture."""
        prelude_params = sum(p.numel() for b in self.prelude for p in b.parameters())
        core_params    = sum(p.numel() for p in self.recurrent_core.parameters())
        coda_params    = sum(p.numel() for b in self.coda for p in b.parameters())
        embed_params   = sum(p.numel() for p in self.embeddings.parameters())
        head_params    = sum(p.numel() for p in self.lm_head.parameters())
        return {
            "total_parameters": self.count_parameters(),
            "prelude_parameters": prelude_params,
            "core_parameters": core_params,
            "coda_parameters": coda_params,
            "embedding_parameters": embed_params,
            "head_parameters": head_params,
            "max_loops": self.config.mythos_max_loops,
            "lora_rank": self.config.mythos_lora_rank,
            "act_threshold": self.config.mythos_act_threshold,
            "spectral_radius_rho_A": self.get_spectral_radius(),
            "prelude_layers": self.config.mythos_prelude_layers,
            "coda_layers": self.config.mythos_coda_layers,
        }

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.config.to_json(os.path.join(path, "config.json"))
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str) -> "RCAMythosModel":
        config = RCAConfig.from_json(os.path.join(path, "config.json"))
        model = cls(config)
        state_dict = torch.load(
            os.path.join(path, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model


# Alias
RCAMythosForCausalLM = RCAMythosModel
