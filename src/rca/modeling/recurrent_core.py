"""
Recurrent Core
==============

The heart of the RCA-Mythos architecture: a single GLA block looped T times
with LTI-stable injection, depth-wise LoRA, loop index embedding, and
adaptive computation time (ACT) halting.

This converts the flat GLA zone from RCA v2.0 into a recurrent depth module:
    - Same GLA block weights used at every loop (parameter efficiency)
    - LoRA depth adapter: per-loop specialization (tiny overhead)
    - LTI injection: stable hidden state update, ρ(A) < 1 guaranteed
    - Loop index embedding: depth-aware representations
    - ACT halting: easy tokens halt early, hard tokens get more loops

Recurrent update rule (per loop step t):
    1.  h = loop_index_embedding(h, t)
    2.  gla_out = GLA(norm(h + e))          ← input injection prevents drift
    3.  gla_out += LoRAAdapter(gla_out, t)   ← depth specialization
    4.  h = LTIInjection(h, e, gla_out)     ← A·h + B·e + gla_out, stable
    5.  p_halt = ACT(h)                      ← adaptive exit per position
    6.  accumulate weighted output

Training loop count (Parcae recommendation):
    T ~ Uniform(1, max_loops) per batch step.
    This is the key to stable training and depth extrapolation.

Test-time scaling:
    Performance follows saturating exponential decay up to μ_rec (training mean).
    Beyond that, gains are minimal. Set n_loops at inference = μ_rec or use ACT.

Author: Rajaaditya.R
References:
    - Parcae (Prairie et al., 2026) — stable LTI injection
    - Relaxed Recursive Transformers (Bae et al., 2024) — depth LoRA
    - Universal Transformers (Graves/Dehghani, 2018) — ACT halting
    - "Loop, Think & Generalize" (arXiv 2604.07822) — depth extrapolation
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import random

from ..layers.gla import GatedLinearAttention
from ..layers.norm import RMSNorm
from ..layers.lti_injection import LTIInjection
from ..layers.act_halting import ACTHalting
from ..layers.loop_embedding import LoopIndexEmbedding
from ..layers.lora_depth import DepthLoRAAdapter


class RecurrentCore(nn.Module):
    """
    RCA-Mythos Recurrent Core: GLA block looped T times.

    Architecture per loop step:
        h = loop_embed(h, t)
        gla_out = GLA(norm(h + e))
        gla_out = gla_out + LoRA(gla_out, t)
        h = LTI(h, e, gla_out)        →  A·h + B·e + gla_out
        p_halt = ACT(h)               →  per-position halt decision

    Parameters
    ----------
    dim : int
        Hidden state dimension.
    num_heads : int
        Number of GLA attention heads.
    max_loops : int
        Maximum loop iterations at training and inference.
    lora_rank : int
        Rank of the depth-wise LoRA adapter. Default 16.
    act_threshold : float
        ACT cumulative halting threshold. Default 0.99.
    loop_dim_fraction : float
        Fraction of dim channels receiving loop index embedding. Default 0.125.
    expand_k : float
        GLA key expansion factor. Default 1.0.
    expand_v : float
        GLA value expansion factor. Default 2.0.
    dropout : float
        Dropout on GLA output. Default 0.1.
    random_loop_training : bool
        If True, sample T ~ Uniform(1, max_loops) during training.
        This is the Parcae recommendation for stable depth extrapolation.
        Default True.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        max_loops: int = 8,
        lora_rank: int = 16,
        act_threshold: float = 0.99,
        loop_dim_fraction: float = 0.125,
        expand_k: float = 1.0,
        expand_v: float = 2.0,
        dropout: float = 0.1,
        random_loop_training: bool = True,
    ):
        super().__init__()
        self.max_loops = max_loops
        self.act_threshold = act_threshold
        self.random_loop_training = random_loop_training

        # Shared GLA block (same weights every loop)
        self.norm = RMSNorm(dim)
        self.gla = GatedLinearAttention(
            dim=dim,
            num_heads=num_heads,
            expand_k=expand_k,
            expand_v=expand_v,
        )
        self.drop = nn.Dropout(dropout)

        # Depth-specialization: per-loop LoRA (tiny overhead)
        self.lora = DepthLoRAAdapter(dim=dim, rank=lora_rank, max_loops=max_loops)

        # Stable injection: h_{t+1} = A·h + B·e + gla_out
        self.injection = LTIInjection(dim)

        # Adaptive halting: per-position early exit
        self.act = ACTHalting(dim, threshold=act_threshold)

        # Loop-depth positional signal
        self.loop_embed = LoopIndexEmbedding(dim, loop_dim_fraction=loop_dim_fraction)

        # FFN norm for after injection (pre-coda normalization)
        self.output_norm = RMSNorm(dim)

    # ------------------------------------------------------------------
    def _get_n_loops(self, n_loops: Optional[int]) -> int:
        """Resolve loop count: random during training, fixed at inference."""
        if n_loops is not None:
            return n_loops
        if self.training and self.random_loop_training:
            # Parcae: sample T ~ Uniform(1, max_loops) per forward pass
            return random.randint(1, self.max_loops)
        return self.max_loops

    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        n_loops: Optional[int] = None,
        return_ponder_loss: bool = False,
        return_per_loop_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Run the recurrent loop.

        Parameters
        ----------
        h : torch.Tensor, shape (B, T, dim)
            Initial hidden state from the Prelude (SSM zone output).
        e : torch.Tensor, shape (B, T, dim)
            Encoded input from the Prelude, frozen and injected every step.
        n_loops : int, optional
            Number of loops to run. If None:
              - Training: sampled from Uniform(1, max_loops) [Parcae]
              - Inference: uses max_loops
        return_ponder_loss : bool
            If True, return the ACT ponder regularization loss tensor.
        return_per_loop_states : bool
            If True, return list of hidden states per loop (for analysis).

        Returns
        -------
        h_out : torch.Tensor, shape (B, T, dim)
            ACT-weighted sum of hidden states across iterations.
        ponder_loss : torch.Tensor or None
            ACT ponder regularization loss (mean N+R over positions).
        """
        n_loops = self._get_n_loops(n_loops)
        B, T, D = h.shape
        device, dtype = h.device, h.dtype

        # ACT state accumulators
        halted = torch.zeros(B, T, device=device, dtype=torch.bool)
        cumulative_p = torch.zeros(B, T, device=device, dtype=dtype)
        h_out = torch.zeros_like(h)
        ponder_steps = torch.zeros(B, T, device=device, dtype=dtype)

        per_loop = [] if return_per_loop_states else None

        for t in range(n_loops):
            if halted.all():
                break  # All positions have halted — early exit

            # ── Step 1: Loop index embedding ──────────────────────────
            h = self.loop_embed(h, t)

            # ── Step 2: GLA forward (weight-shared block) ─────────────
            # Inject encoded input e at every loop to prevent drift
            h_norm = self.norm(h + e)
            gla_out = self.drop(self.gla(h_norm))

            # ── Step 3: Depth-wise LoRA delta ──────────────────────────
            gla_out = gla_out + self.lora(gla_out, t)

            # ── Step 4: LTI-stable hidden state update ──────────────────
            # h_{t+1} = A·h + B·e + gla_out   (ρ(A) < 1 guaranteed)
            h = self.injection(h, e, gla_out)

            # ── Step 5: ACT halting ──────────────────────────────────────
            p_halt = self.act(h)           # (B, T)

            cumulative_p, halted, h_out, remainder = ACTHalting.accumulate(
                p_halt, cumulative_p, halted, h, h_out,
                threshold=self.act_threshold,
            )
            ponder_steps = ponder_steps + (~halted).float()

            if per_loop is not None:
                per_loop.append(h.detach().clone())

        # Final normalization before Coda
        h_out = self.output_norm(h_out)

        pond_loss = None
        if return_ponder_loss:
            pond_loss = ACTHalting.ponder_loss(ponder_steps)

        if return_per_loop_states:
            return h_out, pond_loss, per_loop
        return h_out, pond_loss

    # ------------------------------------------------------------------
    def get_spectral_radius(self) -> float:
        """Monitor ρ(A) during training — should stay below 1.0."""
        return self.injection.get_spectral_radius()
