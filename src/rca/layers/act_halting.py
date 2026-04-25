"""
Adaptive Computation Time (ACT) Halting
========================================

Graves (2016) adaptive halting mechanism for recurrent depth models.

Each position in the sequence accumulates a halting probability across loop
iterations. When a position's cumulative probability exceeds the threshold,
it stops contributing to further loop updates.

Benefits:
  - Easy tokens stop early → compute budget goes to hard tokens
  - Makes the model Turing-complete under mild assumptions
  - 2–3× throughput improvement with continuous depth-wise batching

Training
--------
The ACT mechanism is differentiable. To encourage efficient use of compute,
add the optional ``ponder_loss`` as a regularizer:

    loss = lm_loss + act_loss_weight * ponder_loss

where ponder_loss = mean(N + R) over positions (N = loops used, R = remainder).

Author: Rajaaditya.R
Reference: Universal Transformers (Graves, 2016); Dehghani et al., 2018
"""

import torch
import torch.nn as nn
from typing import Tuple


class ACTHalting(nn.Module):
    """
    Adaptive Computation Time halting for looped (recurrent depth) models.

    Learns a per-position halting scalar at each loop iteration. Positions
    that have "converged" (high cumulative halting probability) stop
    contributing to the ACT-weighted output.

    Parameters
    ----------
    dim : int
        Hidden state dimension (input to the halting predictor MLP).
    threshold : float
        Cumulative probability above which a position is considered halted.
        Default 0.99 (Graves 2016 recommendation).
    """

    def __init__(self, dim: int, threshold: float = 0.99):
        super().__init__()
        self.threshold = threshold
        # Small MLP: single linear → sigmoid. Bias initialised negative so
        # the model initially prefers fewer loops (learns to halt later).
        self.halt_proj = nn.Linear(dim, 1)
        nn.init.constant_(self.halt_proj.bias, -1.0)

    # ------------------------------------------------------------------
    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Predict per-position halting probability.

        Parameters
        ----------
        h : torch.Tensor, shape (B, T, dim)
            Current hidden state at this loop iteration.

        Returns
        -------
        torch.Tensor, shape (B, T)
            Halting probabilities in (0, 1) for each position.
        """
        return torch.sigmoid(self.halt_proj(h)).squeeze(-1)

    # ------------------------------------------------------------------
    @staticmethod
    def accumulate(
        p_halt: torch.Tensor,           # (B, T)  halting prob this step
        cumulative_p: torch.Tensor,     # (B, T)  cumulative so far
        halted: torch.Tensor,           # (B, T)  bool: already halted
        h: torch.Tensor,               # (B, T, dim) current hidden
        h_out: torch.Tensor,           # (B, T, dim) accumulated weighted output
        threshold: float = 0.99,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Update ACT accumulators for one loop step.

        Returns
        -------
        (new_cumulative_p, new_halted, new_h_out, remainder)
            remainder : (B, T) weight assigned to this step's contribution.
        """
        # Positions that would cross the threshold on this step
        will_halt = (cumulative_p + p_halt >= threshold) & ~halted

        # Remainder weight for positions that halt exactly here
        remainder = torch.where(will_halt, 1.0 - cumulative_p, p_halt)

        # Accumulate output: weighted sum of h across iterations
        # Halted positions get weight 0 (already frozen)
        weight = torch.where(halted, torch.zeros_like(remainder), remainder)
        h_out = h_out + weight.unsqueeze(-1) * h

        # Update accumulators
        new_cumulative_p = cumulative_p + torch.where(halted, torch.zeros_like(p_halt), p_halt)
        new_halted = halted | will_halt

        return new_cumulative_p, new_halted, h_out, remainder

    # ------------------------------------------------------------------
    @staticmethod
    def ponder_loss(
        ponder_steps: torch.Tensor,   # (B, T) float — N + R per position
    ) -> torch.Tensor:
        """
        ACT regularization loss = mean(N + R) over all positions.

        Add to your training loss as:
            total_loss = lm_loss + act_loss_weight * ACTHalting.ponder_loss(steps)

        where steps is accumulated during the recurrent loop.
        Typical act_loss_weight: 0.01 – 0.001.
        """
        return ponder_steps.mean()
