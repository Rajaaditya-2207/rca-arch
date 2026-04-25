"""
Smoke test for RCA-Mythos v3.0
================================
Verifies:
  1. Import of all new classes
  2. Config creation (all 4 presets)
  3. RCAMythosModel instantiation (tiny config)
  4. Forward pass with loss
  5. LTI spectral radius ρ(A) < 1
  6. Generation (greedy)
  7. Architecture summary
  8. Backward compatibility: RCAModel still works
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch

# ── 1. Imports ─────────────────────────────────────────────────────────────
print("=" * 60)
print("RCA-Mythos v3.0 Smoke Test")
print("=" * 60)

from rca import (
    RCAConfig, RCAModel, RCAForCausalLM,
    RCAMythosModel, RCAMythosForCausalLM, RecurrentCore,
    LTIInjection, ACTHalting, LoopIndexEmbedding, DepthLoRAAdapter,
)
print("[OK] All imports successful")

# ── 2. Config presets ──────────────────────────────────────────────────────
for name, fn in [
    ("mythos_100m", RCAConfig.rca_mythos_100m),
    ("mythos_500m", RCAConfig.rca_mythos_500m),
    ("mythos_1b",   RCAConfig.rca_mythos_1b),
    ("mythos_3b",   RCAConfig.rca_mythos_3b),
]:
    cfg = fn()
    assert cfg.use_recurrent_depth, f"{name}: use_recurrent_depth should be True"
    print(f"[OK] Config {name}: max_loops={cfg.mythos_max_loops}, "
          f"prelude={cfg.mythos_prelude_layers}, coda={cfg.mythos_coda_layers}")

# ── 3. Tiny model instantiation ────────────────────────────────────────────
tiny_cfg = RCAConfig(
    vocab_size=256,
    state_dim=64,
    n_layers=2,
    n_heads=4,
    ssm_expand=2,
    max_seq_len=32,
    dropout=0.0,
    use_hybrid_attention=False,
    num_attention_layers=0,
    attention_every_n=0,
    gla_heads=4,
    gla_expand_k=1.0,
    gla_expand_v=1.0,
    sliding_window_size=16,
    num_memory_tokens=4,
    use_mqa=False,
    use_ultra_reasoning=False,
    use_glu_ffn=True,
    use_selective_scan=True,
    use_full_matrix=False,
    use_recurrent_depth=True,
    mythos_prelude_layers=2,
    mythos_coda_layers=1,
    mythos_max_loops=3,
    mythos_lora_rank=8,
    mythos_act_threshold=0.99,
    mythos_loop_embed_fraction=0.125,
    mythos_random_loop_training=True,
    gradient_checkpointing=False,
)

model = RCAMythosModel(tiny_cfg)
total_params = model.count_parameters()
print(f"[OK] RCAMythosModel instantiated: {total_params:,} parameters")

# ── 4. Forward pass with loss ──────────────────────────────────────────────
B, S = 2, 16
ids    = torch.randint(0, 256, (B, S))
labels = ids.clone()

model.train()
out = model(ids, labels=labels, n_loops=3, act_loss_weight=0.01)

assert out.loss is not None,             "Loss should not be None"
assert out.logits.shape == (B, S, 256),  f"Logits shape wrong: {out.logits.shape}"
assert not torch.isnan(out.loss),        f"Loss is NaN: {out.loss}"
assert not torch.isinf(out.loss),        f"Loss is inf: {out.loss}"
print(f"[OK] Forward pass: loss={out.loss.item():.4f}, logits={out.logits.shape}")

# ── 5. Backward pass ──────────────────────────────────────────────────────
out.loss.backward()
print("[OK] Backward pass (gradients computed successfully)")

# ── 6. LTI spectral radius rho(A) < 1 ───────────────────────────────────────
rho = model.get_spectral_radius()
assert rho < 1.0, f"LTI stability violated! rho(A) = {rho:.6f} >= 1.0"
print(f"[OK] LTI stability: rho(A) = {rho:.6f} (must be < 1.0) STABLE")

# ── 7. Generation ─────────────────────────────────────────────────────────
model.eval()
prompt = torch.randint(0, 256, (1, 4))
with torch.no_grad():
    generated = model.generate(prompt, max_new_tokens=8, n_loops=3)
assert generated.shape[0] == 1
assert generated.shape[1] >= 4
print(f"[OK] Generation: {prompt.shape} -> {generated.shape}")

# ── 8. Architecture summary ───────────────────────────────────────────────
summary = model.get_architecture_summary()
print("[OK] Architecture summary:")
for k, v in summary.items():
    print(f"       {k:35s} = {v}")

# ── 9. Backward compatibility: RCAModel v2.0 ──────────────────────────────
old_cfg = RCAConfig.rca_100m()
old_model = RCAModel(old_cfg)
old_ids = torch.randint(0, old_cfg.vocab_size, (1, 8))
old_out = old_model(old_ids)
assert old_out.logits.shape[0] == 1
print(f"[OK] Backward compat: RCAModel v2.0 still works — logits {old_out.logits.shape}")

# ── 10. Fixed-loop inference (depth extrapolation) ────────────────────────
model.eval()
with torch.no_grad():
    out2 = model(ids, n_loops=6)  # 2× training max_loops
assert not torch.isnan(out2.logits).any(), "NaN in extrapolated logits"
print(f"[OK] Depth extrapolation: n_loops=6 (2x training max=3) - no NaN OK")

print()
print("=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
