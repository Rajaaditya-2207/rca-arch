"""
RCA Model
=========

Full RCA v2.0 model implementation with Ultra-Reasoning Architecture.

Brain Analogy Zones:
- SSM Zone (stream of consciousness): Long-range sequence processing
- GLA Zone (working memory): Associative recall with linear attention
- Reasoning Zone (focus): Local precision with sliding window + memory tokens
- All zones use GLU-FFN (active neurons) for world knowledge

Author: Rajaaditya.R
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
import math
import json
import os
from typing import Optional, Tuple, Dict

from ..config import RCAConfig
from ..layers.ssm import SelectiveStateSpaceModel, SimpleStateSpaceModel
from ..layers.attention import EfficientAttention
from ..layers.gla import GatedLinearAttention
from ..layers.sliding_attention import SlidingWindowAttention
from ..layers.norm import RMSNorm, DeepNorm
from .outputs import CausalLMOutput, ModelOutput


# =========================================================================
# FFN Variants
# =========================================================================

class GLUFFN(nn.Module):
    """
    Gated Linear Unit Feed-Forward Network (Active Neurons).

    GLU variant: out = (Wup · x ⊙ σ(Wgate · x)) · Wdown
    Stores 'world knowledge' more effectively than standard FFN.
    """

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        hidden = dim * expand * 2 // 3  # Adjusted so total params ≈ standard FFN
        # Round to nearest multiple of 64 for efficiency
        hidden = ((hidden + 63) // 64) * 64

        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x)))


class StandardFFN(nn.Module):
    """Standard GELU Feed-Forward Network."""

    def __init__(self, dim: int, expand: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * expand, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * expand, dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def make_ffn(config: RCAConfig) -> nn.Module:
    """Create FFN module based on config."""
    if config.use_glu_ffn:
        return GLUFFN(config.state_dim, dropout=config.dropout)
    return StandardFFN(config.state_dim, dropout=config.dropout)


# =========================================================================
# Block Types
# =========================================================================

class MambaMixBlock(nn.Module):
    """
    Original RCA block: SSM + optional attention + FFN.

    Used for both standard RCA and as the SSM zone in ultra-reasoning.
    """

    def __init__(self, config: RCAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        d = config.state_dim
        expand_d = d * config.ssm_expand

        # Pre-norm
        self.norm1 = RMSNorm(d)

        # SSM branch
        self.in_proj = nn.Linear(d, expand_d, bias=False)
        self.gate_proj = nn.Linear(d, expand_d, bias=False)

        if config.use_selective_scan:
            self.ssm = SelectiveStateSpaceModel(
                expand_d, expand_d, use_full_matrix=config.use_full_matrix
            )
        else:
            self.ssm = SimpleStateSpaceModel(expand_d, expand_d)

        self.out_proj = nn.Linear(expand_d, d, bias=False)

        # Attention branch (on selected layers only) — for non-ultra mode
        self.has_attention = (
            config.use_hybrid_attention
            and config.attention_every_n > 0
            and layer_idx % config.attention_every_n == 0
            and layer_idx < config.num_attention_layers * config.attention_every_n
        )
        if self.has_attention:
            self.attn_norm = RMSNorm(d)
            self.attention = EfficientAttention(
                d,
                config.n_heads,
                dropout=config.dropout,
                use_rotary=config.use_rotary,
                use_mqa=config.use_mqa,
            )
            self.attn_gate = nn.Parameter(torch.tensor(0.5))

        # FFN
        self.ffn_norm = RMSNorm(d)
        self.ffn = make_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        ssm_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_cuda: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        residual = x

        # SSM branch
        h = self.norm1(x)
        z = self.in_proj(h)
        gate = torch.sigmoid(self.gate_proj(h))
        z = z * gate

        if use_cache and z.size(1) == 1:
            ssm_out, new_ssm_state = self.ssm.forward_sequential(
                z.squeeze(1), ssm_state
            )
            ssm_out = ssm_out.unsqueeze(1)
        else:
            ssm_out = self.ssm.forward_parallel(z, ssm_state, use_cuda=use_cuda)
            new_ssm_state = None

        x = residual + self.out_proj(ssm_out)

        # Attention branch
        if self.has_attention:
            h = self.attn_norm(x)
            attn_out = self.attention(h, is_causal=True)
            gate_val = torch.sigmoid(self.attn_gate)
            x = x + gate_val * attn_out

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_ssm_state


class GLABlock(nn.Module):
    """
    GLA Block: Gated Linear Attention + GLU-FFN.

    Acts as 'Working Memory' — associative recall with linear-time complexity.
    """

    def __init__(self, config: RCAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        d = config.state_dim

        self.norm1 = RMSNorm(d)
        self.gla = GatedLinearAttention(
            dim=d,
            num_heads=config.gla_heads,
            expand_k=config.gla_expand_k,
            expand_v=config.gla_expand_v,
        )

        self.ffn_norm = RMSNorm(d)
        self.ffn = make_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        gla_state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_cuda: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # GLA branch
        h = self.norm1(x)

        if use_cache and x.size(1) == 1:
            gla_out, new_state = self.gla.forward_recurrent(h, gla_state)
        else:
            gla_out = self.gla(h)
            new_state = None

        x = x + gla_out

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x, new_state


class ReasoningBlock(nn.Module):
    """
    Reasoning Block: Sliding Window Attention + Memory Tokens + GLU-FFN.

    Acts as 'Focus' — perfect local vision with global context bookmarks.
    """

    def __init__(self, config: RCAConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
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
        self.ffn = make_ffn(config)

    def forward(
        self,
        x: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        use_cache: bool = False,
        use_cuda: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Sliding Window + Memory Tokens
        h = self.norm1(x)
        attn_out = self.attention(h, is_causal=True)
        x = x + attn_out

        # FFN
        x = x + self.ffn(self.ffn_norm(x))

        return x, None


# =========================================================================
# Main Model
# =========================================================================

class RCAModel(nn.Module):
    """
    RCA v2.0 — Recursive Compression Architecture.

    Standard mode:
    - Selective SSM backbone (Mamba-style)
    - Hybrid attention at select layers
    - O(1) generation memory

    Ultra-Reasoning mode (Brain Analogy):
    - SSM Zone (60%): stream of consciousness
    - GLA Zone (25%): working memory
    - Reasoning Zone (15%): focus with sliding window + memory tokens
    - All zones: GLU-FFN active neurons

    Performance features:
    - Gradient checkpointing (config.gradient_checkpointing)
    - torch.compile support (config.use_torch_compile)
    """

    def __init__(self, config: RCAConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Token embeddings
        self.embeddings = nn.Embedding(config.vocab_size, config.state_dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Build layers based on mode
        if config.use_ultra_reasoning:
            self.layers = self._build_ultra_layers(config)
        else:
            self.layers = nn.ModuleList(
                [MambaMixBlock(config, i) for i in range(config.n_layers)]
            )

        self.final_norm = RMSNorm(config.state_dim)

        # LM head
        self.lm_head = nn.Linear(config.state_dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.embeddings.weight

        # Init weights
        self.apply(self._init_weights)

    def _build_ultra_layers(self, config: RCAConfig) -> nn.ModuleList:
        """Build layer zones for ultra-reasoning architecture."""
        layers = nn.ModuleList()
        n = config.n_layers

        ssm_end = int(n * config.ssm_zone_end)
        gla_end = int(n * config.gla_zone_end)

        for i in range(n):
            if i < ssm_end:
                # SSM Zone: stream of consciousness
                layers.append(MambaMixBlock(config, i))
            elif i < gla_end:
                # GLA Zone: working memory
                layers.append(GLABlock(config, i))
            else:
                # Reasoning Zone: focus
                layers.append(ReasoningBlock(config, i))

        return layers

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)

    def _layer_forward(self, layer, h, state, use_cache, use_cuda):
        """Helper for gradient checkpointing compatibility."""
        return layer(h, state, use_cache=use_cache, use_cuda=use_cuda)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        ssm_states: Optional[Tuple[Optional[torch.Tensor], ...]] = None,
        use_cache: bool = False,
        use_cuda: bool = True,
    ) -> CausalLMOutput:
        """
        Forward pass.

        Args:
            input_ids: [B, S] token ids
            labels: [B, S] optional labels for loss computation
            ssm_states: optional tuple of layer states
            use_cache: if True, return updated states
            use_cuda: if True, use Triton/CUDA when available
        """
        h = self.embed_dropout(self.embeddings(input_ids))

        new_ssm_states = []
        if ssm_states is None:
            ssm_states = [None] * len(self.layers)

        for i, (layer, state) in enumerate(zip(self.layers, ssm_states)):
            if self.gradient_checkpointing and self.training and not use_cache:
                # Gradient checkpointing: trades compute for memory
                # Cannot use with use_cache since checkpoint doesn't support
                # returning non-tensor outputs reliably
                h, new_state = grad_checkpoint(
                    self._layer_forward,
                    layer, h, state, use_cache, use_cuda,
                    use_reentrant=False,
                )
            else:
                h, new_state = layer(h, state, use_cache=use_cache, use_cuda=use_cuda)
            new_ssm_states.append(new_state)

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

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            last_hidden_state=h,
            ssm_states=tuple(new_ssm_states) if use_cache else None,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive generation with O(1) memory."""
        self.eval()
        B = input_ids.shape[0]
        device = input_ids.device
        eos = eos_token_id or self.config.eos_token_id

        # Process prefix (in parallel)
        out = self.forward(input_ids, use_cache=True, use_cuda=True)
        logits = out.logits[:, -1, :]
        ssm_states = out.ssm_states

        generated = []
        for _ in range(max_new_tokens):
            # Sample next token
            logits = logits / max(temperature, 1e-8)

            if top_k > 0:
                values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < values[:, -1:]] = float("-inf")

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated.append(next_token)

            if (next_token == eos).all():
                break

            # Process next token (O(1) per step)
            out = self.forward(next_token, ssm_states=ssm_states, use_cache=True)
            logits = out.logits[:, -1, :]
            ssm_states = out.ssm_states

        return torch.cat([input_ids] + generated, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_layer_zones(self) -> Dict[str, list]:
        """Return which layers belong to which zone."""
        if not self.config.use_ultra_reasoning:
            return {"ssm": list(range(len(self.layers)))}

        zones = {"ssm": [], "gla": [], "reasoning": []}
        for i, layer in enumerate(self.layers):
            if isinstance(layer, MambaMixBlock):
                zones["ssm"].append(i)
            elif isinstance(layer, GLABlock):
                zones["gla"].append(i)
            elif isinstance(layer, ReasoningBlock):
                zones["reasoning"].append(i)
        return zones

    def save_pretrained(self, path: str):
        """Save model and config to directory."""
        os.makedirs(path, exist_ok=True)
        self.config.to_json(os.path.join(path, "config.json"))
        torch.save(self.state_dict(), os.path.join(path, "model.pt"))

    @classmethod
    def from_pretrained(cls, path: str, **kwargs) -> "RCAModel":
        """Load model from directory."""
        config = RCAConfig.from_json(os.path.join(path, "config.json"))
        model = cls(config)
        state_dict = torch.load(
            os.path.join(path, "model.pt"),
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(state_dict)
        return model


# Convenient alias
RCAForCausalLM = RCAModel
