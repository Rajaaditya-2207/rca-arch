"""
LTI Stable Injection
====================

Parcae-style (Prairie et al., 2026) Linear Time-Invariant injection for
the recurrent core update rule:

    h_{t+1} = A · h_t  +  B · e  +  transformer_out

Stability is guaranteed by construction:
    A_continuous = Diag(-exp(log_A))         always negative diagonal
    A_discrete   = exp(Δt · A_continuous)    values strictly in (0, 1)
    → ρ(A) < 1 always, regardless of learning rate or batch noise

This prevents:
  - Residual explosion (hidden state growing unboundedly across loops)
  - Loss spikes from large spectral norms

Author: Rajaaditya.R
Reference: Parcae — Scaling Laws for Stable Looped Language Models (2026)
"""

import torch
import torch.nn as nn


class LTIInjection(nn.Module):
    """
    Stable input injection for the recurrent hidden state update.

    The recurrent hidden state evolves per loop step as:
        h_{t+1} = A · h_t  +  B · e  +  transformer_out

    where:
        e   = encoded input from the Prelude (SSM zone output), frozen
              and injected at every loop step to prevent state drift.
        A   = ZOH-discretized diagonal decay matrix, ρ(A) < 1 always
        B   = learned per-channel input coupling vector

    Parameters
    ----------
    dim : int
        Hidden state dimension. One scalar parameter per channel for A and B.
    clamp_range : tuple
        Log-space clamp to keep float32 arithmetic finite. Default (-20, 20).
    """

    def __init__(self, dim: int, clamp_range: tuple = (-20, 20)):
        super().__init__()
        self.clamp_lo, self.clamp_hi = clamp_range

        # log_A: log magnitude of the continuous negative diagonal.
        # Initialized to 0 → A_discrete ≈ exp(-1) ≈ 0.37 initially.
        self.log_A = nn.Parameter(torch.zeros(dim))

        # log_dt: log of the discretization time step Δt.
        # Initialized to 0 → Δt = 1 initially.
        self.log_dt = nn.Parameter(torch.zeros(1))

        # B: input coupling. Initialized small to let transformer dominate early.
        self.B = nn.Parameter(torch.ones(dim) * 0.1)

    # ------------------------------------------------------------------
    def get_A(self) -> torch.Tensor:
        """
        Compute the discretized diagonal A matrix.

        Returns
        -------
        torch.Tensor of shape (dim,), values strictly in (0, 1).
            ρ(A) = max(A) < 1 is guaranteed for any learned parameter values.

        Implementation note
        -------------------
        We compute in log-space to avoid 0 * inf = NaN when
        log_dt → -∞ and log_A → +∞ simultaneously.

            dt * A_c = -exp(log_dt) * exp(log_A) = -exp(log_dt + log_A)
            A_discrete = exp(-exp(log_dt + log_A))

        Clamping keeps the product finite in float32 under any gradient step.
        """
        return torch.exp(
            -torch.exp((self.log_dt + self.log_A).clamp(self.clamp_lo, self.clamp_hi))
        )

    def get_spectral_radius(self) -> float:
        """Convenience: return ρ(A) = max(A_discrete) for monitoring."""
        return self.get_A().abs().max().item()

    # ------------------------------------------------------------------
    def forward(
        self,
        h: torch.Tensor,
        e: torch.Tensor,
        transformer_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the next hidden state.

        Parameters
        ----------
        h : torch.Tensor, shape (B, T, dim)
            Current hidden state from the previous loop iteration.
        e : torch.Tensor, shape (B, T, dim)
            Encoded input from the Prelude. Frozen and re-injected each step.
        transformer_out : torch.Tensor, shape (B, T, dim)
            Output of the GLA recurrent block at this loop step.

        Returns
        -------
        torch.Tensor, shape (B, T, dim)
            h_{t+1} = A · h_t + B · e + transformer_out
        """
        A = self.get_A()          # (dim,) broadcast over (B, T, dim)
        return A * h + self.B * e + transformer_out
