"""
Loop Index Embedding
====================

Sinusoidal positional signal injected into the hidden state at each loop
iteration of the recurrent core.

Without this, the shared GLA recurrent block weights must handle BOTH
early-stage pattern matching (loop 0) and late-stage refinement (loop T-1)
with no signal to distinguish which iteration they're on.

Adding the loop index embedding lets the same parameters implement
functionally distinct operations at different depths — analogous to how
RoPE enables position-aware attention with shared projection weights.

Design
------
- Applied to the first `loop_dim` channels only (default: dim // 8)
- Uses sinusoidal embedding (sin/cos pairs) over loop index t
- Additive: does NOT change the norm of the rest of the hidden state
- Supports depth extrapolation: at inference, t > max_training_loops
  is handled gracefully (sinusoids are defined for any integer t)

Author: Rajaaditya.R
Reference: davidad (2026); "Loop, Think & Generalize" (arXiv 2604.07822)
"""

import torch
import torch.nn as nn
import math


class LoopIndexEmbedding(nn.Module):
    """
    Sinusoidal loop-index embedding injected into hidden state h.

    Parameters
    ----------
    dim : int
        Full hidden dimension.
    loop_dim_fraction : float
        Fraction of channels receiving the loop embedding. Default 0.125
        (dim // 8 channels). Smaller → less disruption to existing features.
    theta : float
        Sinusoidal base frequency. Default 10000.0 (same as original
        Transformer positional encoding).
    """

    def __init__(
        self,
        dim: int,
        loop_dim_fraction: float = 0.125,
        theta: float = 10000.0,
    ):
        super().__init__()
        self.dim = dim
        self.loop_dim = max(2, int(dim * loop_dim_fraction) // 2 * 2)  # even
        self.theta = theta

    # ------------------------------------------------------------------
    def _compute_embedding(self, loop_t: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """
        Compute sinusoidal embedding vector for loop index t.

        Returns
        -------
        torch.Tensor of shape (dim,) — zero-padded outside loop_dim channels.
        """
        half = self.loop_dim // 2
        freqs = 1.0 / (
            self.theta ** (
                torch.arange(0, half, device=device, dtype=dtype) / half
            )
        )
        angles = loop_t * freqs                              # (half,)
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)  # (loop_dim,)

        # Zero-pad to full dim
        emb_full = torch.zeros(self.dim, device=device, dtype=dtype)
        emb_full[: self.loop_dim] = emb
        return emb_full

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Inject loop index signal into hidden state h.

        Parameters
        ----------
        h : torch.Tensor, shape (B, T, dim)
            Current hidden state before this loop's computation.
        loop_t : int
            Current loop iteration index (0-based).

        Returns
        -------
        torch.Tensor, shape (B, T, dim)
            h with sinusoidal loop-index bias added to first `loop_dim` channels.
        """
        emb = self._compute_embedding(loop_t, h.device, h.dtype)
        # Broadcast over (B, T)
        return h + emb.unsqueeze(0).unsqueeze(0)
