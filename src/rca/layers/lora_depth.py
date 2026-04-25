"""
Depth-wise LoRA Adapter
========================

Per-loop low-rank adaptation for the recurrent core block.
From "Relaxed Recursive Transformers" (Bae et al., 2024).

Problem: pure weight-tying (same weights every loop) limits expressiveness.
The GLA block must simultaneously handle early-stage pattern matching AND
late-stage refinement with identical weights — a tight constraint.

Solution: add a small depth-wise LoRA delta at each iteration:

    delta(x, t) = down(x) @ diag(scale[t]) @ B

where:
    - down: shared (dim → rank) projection  — one copy for all loops
    - B:    shared (rank → dim) projection  — one copy for all loops
    - scale[t]: per-loop (rank,) scaling vector — tiny per-depth adaptation

Parameter overhead:
    - Shared: 2 × dim × rank
    - Per-loop: max_loops × rank
    - Total ≈ 2 × dim × rank + max_loops × rank
    - For dim=768, rank=16, max_loops=16: 24,576 + 256 = 24,832 params
      vs. 589,824 params per full GLA head — ~4% overhead for full depth-
      aware expressiveness.

Initialization (Relaxed Recursive Transformers best practice)
-------------------------------------------------------------
    - down.weight: normal(0, 0.02)
    - B: normal(0, 0.02)
    - scale: ones (start as identity-like adapter)
    → The adapter starts near-zero output, so the model begins training
      identically to pure weight-tying before gradually specializing.

Author: Rajaaditya.R
Reference: Relaxed Recursive Transformers — Effective Parameter Sharing
           with Layer-wise LoRA (Bae et al., 2024, arXiv:2410.20672)
"""

import torch
import torch.nn as nn


class DepthLoRAAdapter(nn.Module):
    """
    Depth-wise LoRA adapter for the looped recurrent core.

    Applies a per-loop low-rank transformation to the GLA output,
    giving the shared block depth-specific behavior.

    Parameters
    ----------
    dim : int
        Model hidden dimension (input and output size of delta).
    rank : int
        Low-rank bottleneck. Recommended: 8–32 (default 16).
    max_loops : int
        Maximum number of loop iterations (size of the scale embedding table).
        At inference, loops > max_loops reuse the last scale (depth extrapolation).
    """

    def __init__(self, dim: int, rank: int = 16, max_loops: int = 16):
        super().__init__()
        self.rank = rank
        self.max_loops = max_loops

        # Shared low-rank projections (same across all loop depths)
        self.down = nn.Linear(dim, rank, bias=False)   # A: dim → rank
        self.B = nn.Parameter(torch.randn(rank, dim) * 0.02)  # B: rank → dim

        # Per-loop scale vector — the only depth-specific parameters
        # shape: (max_loops, rank)
        self.scale = nn.Embedding(max_loops, rank)

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.down.weight, std=0.02)
        nn.init.ones_(self.scale.weight)   # start near identity

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, loop_t: int) -> torch.Tensor:
        """
        Compute the depth-wise LoRA delta for loop iteration t.

        Parameters
        ----------
        x : torch.Tensor, shape (B, T, dim)
            Output of the GLA block at this loop step.
        loop_t : int
            Current loop index (0-based).

        Returns
        -------
        torch.Tensor, shape (B, T, dim)
            Delta to be added to the GLA output before LTI injection.
        """
        # Clamp for depth extrapolation: if inference loops > max_loops,
        # reuse the last learned per-loop scale.
        t_idx = min(loop_t, self.max_loops - 1)
        idx = torch.tensor(t_idx, device=x.device, dtype=torch.long)

        s = self.scale(idx)           # (rank,)
        down = self.down(x) * s       # (B, T, rank) — per-loop scaled projection
        return down @ self.B          # (B, T, dim)
