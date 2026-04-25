# RCA 1.3.0 — Recurrent Cross Attention Architecture

**RCA-Mythos (v1.3.0)**: A hybrid **Recurrent-Depth Architecture** combining Mamba SSM, Gated Linear Attention (GLA), and Sliding Window Attention across specialized cognitive zones — heavily inspired by OpenMythos and Parcae scaling laws.

> RCA v1.3.0 introduces the **Mythos** architecture, which replaces deep parameter stacking with a **recurrent depth loop**. A single block of parameters is looped multiple times at inference, allowing the model to "think deeper" about hard problems without consuming extra memory.

---

## Why RCA?

| Feature | Transformer | RCA v1.0/v2.0 | **RCA-Mythos v1.3.0** |
|---|---|---|---|
| Training complexity | O(N²) | O(N) | **O(N)** |
| Generation memory | O(N) KV cache | O(1) | **O(1)** |
| Reasoning Depth | Fixed (depth = layers) | Fixed | **Dynamic (Depth Extrapolation)** |
| Parameter Efficiency | 1x | 1x | **~5x (via recurrent looping)** |
| Generation speed | Slows with context | Constant | **Adaptive (ACT Halting)** |

### Version History & Upgrade Guide

* **v1.0 / v2.0 (Ultra-Reasoning):** Introduced the 3-zone architecture (SSM $\rightarrow$ GLA $\rightarrow$ Reasoning). This is a "flat" architecture where every layer has unique weights. Best for standard workflows where deterministic layer-by-layer execution is desired.
* **v1.3.0 (RCA-Mythos):** Introduced the **Recurrent-Depth** architecture. The GLA zone is replaced by a `RecurrentCore` that loops a single set of weights $T$ times. This introduces **Depth Extrapolation** (the model can think deeper at inference by looping more times) and massive parameter efficiency.

### Architecture: The 3-Stage Recurrent Pipeline (Mythos)

RCA-Mythos divides processing into three stages, running the GLA zone inside a recurrent loop. The Open-Mythos "thinking" mechanism is **globally integrated** as the central Recurrent Core of the model, enhancing complex reasoning for all tokens.

```
┌──────────────────────────────────────────────────────────────┐
│   PRELUDE                 │ Stream of Consciousness         │
│   SSM Blocks (run once)   │ Encodes input context → O(1)    │
├──────────────────────────────────────────────────────────────┤
│   RECURRENT CORE          │ Working Memory & Depth          │
│   GLA Block (run T times) │ Associative recall              │
│                           │ LTI-stable injection            │
│                           │ LoRA depth-wise adaptation      │
│                           │ ACT early-halting per-token     │
├──────────────────────────────────────────────────────────────┤
│   CODA                    │ Focus & Precision               │
│   Reasoning (run once)    │ Sliding Window + Memory Tokens  │
└──────────────────────────────────────────────────────────────┘
```

---

## Installation

```bash
# Core (CPU/GPU)
pip install rca-arch

# With GPU acceleration (Triton kernels)
pip install rca-arch[gpu]

# With export support (safetensors)
pip install rca-arch[export]

# With training utilities
pip install rca-arch[training]

# Everything
pip install rca-arch[all]
```

**Requirements:** Python ≥ 3.9, PyTorch ≥ 2.0.0

---

## Quick Start

### 1. Create a Model from Presets

Both the classic `RCAModel` and the new `RCAMythosModel` are available.

```python
from rca import RCAConfig, RCAModel, RCAMythosModel

# --- Option A: Classic RCA (v1/v2 flat architecture) ---
classic_config = RCAConfig.rca_100m()
classic_model = RCAModel(classic_config)

# --- Option B: RCA-Mythos (v1.3.0 recurrent architecture) ---
mythos_config = RCAConfig.rca_mythos_100m()
mythos_config.vocab_size = 32000  # match your tokenizer

model = RCAMythosModel(mythos_config)
print(f"Parameters: {model.count_parameters():,}")
# → Parameters: ~100,000,000 (but performs like a 130M+ model)

# Inspect the architecture zones
print(model.get_architecture_summary())
```

### 2. Forward Pass

```python
import torch

x = torch.randint(0, 32000, (2, 4096))  # [batch, seq_len]
output = model(x)

print(output.logits.shape)  # [2, 4096, 32000]
print(output.loss)          # None (no labels provided)
```

### 3. Forward Pass with Loss

```python
x = torch.randint(0, 32000, (2, 4096))
labels = x.clone()

output = model(x, labels=labels)
print(f"Loss: {output.loss.item():.4f}")
```

### 4. Generate Text (with Depth Extrapolation)

With `RCAMythosModel`, you can dynamically increase the reasoning depth at inference time by passing a higher `n_loops` argument.

```python
prompt = torch.randint(0, 32000, (1, 64))  # [1, prompt_len]

# Generate with test-time depth extrapolation (e.g. 16 loops)
generated = model.generate(
    prompt,
    max_new_tokens=200,
    n_loops=16,          # Force deeper reasoning than training default
    temperature=0.8,
    top_k=50,
    top_p=0.9,
)
print(generated.shape)  # [1, 264]
```

---

## Model Presets

### RCA-Mythos Presets (v1.3.0)

All Mythos presets use the Recurrent-Depth architecture.
A 770M parameter Mythos model reaches the quality of a 1.3B flat-depth model.

| Preset | Params | Equiv. Flat | Loops | Prelude | Coda | Hardware |
|---|---|---|---|---|---|---|
| `rca_mythos_100m()` | ~100M | ~130M | 4 | 4 | 2 | T4 / P100 |
| `rca_mythos_500m()` | ~500M | ~700M | 8 | 6 | 3 | T4 / P100 (ckpt) |
| `rca_mythos_1b()` | ~1B | ~1.5B | 12 | 8 | 4 | A100 |
| `rca_mythos_3b()` | ~3B | ~5B | 16 | 10 | 6 | Multi-GPU / A100 80G |

### Estimated Training Budget (7-hour window)

*Optimal Loop scaling law follows Parcae: μ_rec ∝ C^0.40*

| Preset | T4 (16GB) | P100 (16GB) | Settings |
|---|---|---|---|
| `rca_mythos_100m` | ~700M tokens | ~1.1B tokens | batch=8, grad_accum=4, fp16 |
| `rca_mythos_500m` | ~250M tokens | ~400M tokens | batch=2, grad_accum=16, fp16, grad_ckpt |
| `rca_mythos_1b` | ~100M tokens | ~250M tokens | batch=1, grad_accum=32, fp16, grad_ckpt |

---

## In-Depth: How RCA-Mythos Works

The "Thinking" part of the architecture is fully globally integrated into the model, processing every single token. It is not an add-on; it is the core engine.

1. **The Prelude (SSM)**: Fast, $O(1)$ recurrent scan that rapidly ingests the input context and compresses it into a high-density vector $e$.
2. **The Recurrent Core (GLA Loop)**: Instead of stacking 20 different layers, we take **one** Gated Linear Attention layer and loop it $T$ times (e.g., $T=8$). 
   - **LTI Stable Injection**: At each loop, the context $e$ is injected into the state. We strictly enforce $\rho(A) < 1$ (Linear Time-Invariant stability) so the gradients never explode, even if you loop it 100 times.
   - **Depth LoRA**: A tiny parameter-efficient adapter tells the shared weights *which loop iteration* it is currently on, allowing the layer to act differently at loop 1 vs loop 8.
   - **ACT Halting (Adaptive Computation)**: "Easy" tokens (like "the", "and") accumulate high confidence quickly and exit the loop early. "Hard" tokens (complex math, logic) stay in the loop for all $T$ iterations to "think deeper".
3. **The Coda (Reasoning)**: A final Sliding Window Attention pass that grounds the refined abstract thoughts back into precise token predictions.

---

## Training RCA-Mythos

Training the Mythos architecture is identical to the classic architecture, but with a few built-in advantages. By training with a random number of loops (Parcae strategy), the model learns to extrapolate depth at inference time.

### Training on a Single GPU (e.g., T4, P100, RTX 4050)

```python
from rca import RCAConfig, RCAMythosModel, RCATrainer, TrainingArguments

# 1. Config: Use a Mythos preset
config = RCAConfig.rca_mythos_500m()
config.vocab_size = 32000

# 2. Model: Instantiate the Recurrent-Depth model
model = RCAMythosModel(config)

# 3. Dataset
from torch.utils.data import Dataset
class TextDataset(Dataset):
    def __init__(self, data, seq_len=4096):
        self.data = data
        self.seq_len = seq_len
    def __len__(self): return len(self.data) // self.seq_len
    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}

# 4. Training args (Gradient Checkpointing is enabled by default in 500M)
args = TrainingArguments(
    output_dir="./checkpoints",
    num_train_epochs=1,
    per_device_train_batch_size=2,      # Small batch size to fit in VRAM
    gradient_accumulation_steps=16,     # Accumulate to get effective batch=32
    learning_rate=3e-4,
    warmup_steps=200,
    fp16=True,                          # Essential for 6GB VRAM
    logging_steps=10,
)

# 5. Train
trainer = RCATrainer(model=model, args=args, train_dataset=train_dataset)
trainer.train()
```

### Hardware Estimates: Laptop GPU (RTX 4050 6GB)

#### Scenario A: 100M Model at Chinchilla Optimal Limit (2B Tokens)
The Chinchilla scaling law dictates that optimal training requires ~20 tokens per parameter. For a 100M model, this is **2 Billion tokens**.
- **Compute Required**: A 100M Mythos model effectively computes like a 130M flat model.
  - $FLOPs \approx 6 \times 130,000,000 \times 2,000,000,000 \approx 1.56 \text{ ExaFLOPs}$
- **Hardware Speed**: RTX 4050 Laptop (~30 effective TFLOPs in mixed precision).
- **Time Estimate**: $1.56 \times 10^{18} / 30 \times 10^{12} \approx 52,000 \text{ seconds}$.
- $\approx \mathbf{14.5 \text{ hours}}$. *(You can easily train this overnight on a laptop!)*

#### Scenario B: 500M Model on 10B Tokens
If you scale up to the **500M RCA-Mythos model** on a dataset of **10 Billion tokens** (world knowledge, math, coding):
- **Compute Required**: Acts like a 700M flat model.
  - $FLOPs \approx 6 \times 700,000,000 \times 10,000,000,000 \approx 42 \text{ ExaFLOPs}$
- **Time Estimate**: $4.2 \times 10^{16} / 30 \text{ TFLOPs} \approx 1.4 \text{ million seconds}$.
- $\approx \mathbf{388 \text{ hours}}$ (or about **16 days** of continuous 24/7 training).

**VRAM Note**: To fit these models in **6GB of VRAM**, you MUST use:
1. `fp16=True` or `bf16=True`
2. `per_device_train_batch_size=1` or `2` (use `gradient_accumulation_steps` to compensate)
3. `gradient_checkpointing=True` 
4. An 8-bit optimizer (like `bitsandbytes.optim.AdamW8bit`) to save optimizer state memory.

### Multi-GPU Training (DDP)

```bash
torchrun --nproc_per_node=4 train.py
```

```python
args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    fp16=True,
    use_ddp=True,
    # ...
)
```

### Large Model Training (FSDP — 5B+)

```bash
torchrun --nproc_per_node=8 train.py
```

```python
config = RCAConfig.rca_5b()
config.vocab_size = 32000

args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=64,
    bf16=True,
    use_fsdp=True,
    # ...
)
```

### TPU Training (XLA)

```python
args = TrainingArguments(
    output_dir="./checkpoints",
    per_device_train_batch_size=8,
    use_xla=True,
    bf16=True,
    # ...
)
```

---

## Custom Architecture

Full control over every parameter:

```python
from rca import RCAConfig, RCAModel

config = RCAConfig(
    vocab_size=50257,
    state_dim=768,
    n_layers=24,
    n_heads=12,

    # Ultra-Reasoning zones
    use_ultra_reasoning=True,
    use_glu_ffn=True,
    ssm_zone_end=0.6,        # First 60% = SSM (stream of consciousness)
    gla_zone_end=0.85,       # Next 25%  = GLA (working memory)
    # Remaining 15%          = Reasoning (focus)

    # GLA settings
    gla_heads=12,
    gla_expand_k=1.0,
    gla_expand_v=2.0,

    # Reasoning settings
    sliding_window_size=512,
    num_memory_tokens=32,

    # Performance
    gradient_checkpointing=True,
    max_seq_len=4096,
    dropout=0.1,
)

model = RCAModel(config)
print(model.get_layer_zones())
```

### Key Configuration Parameters

| Parameter | Default | Description |
|---|---|---|
| `use_ultra_reasoning` | `False` | Enable 3-zone architecture |
| `use_glu_ffn` | `False` | SwiGLU FFN instead of standard GELU |
| `ssm_zone_end` | `0.6` | Fraction of layers for SSM zone |
| `gla_zone_end` | `0.85` | Fraction of layers for SSM + GLA zones |
| `gradient_checkpointing` | `False` | Trade compute for memory savings |
| `sliding_window_size` | `512` | Local attention window in reasoning zone |
| `num_memory_tokens` | `32` | Global context bookmarks in reasoning zone |
| `use_mqa` | `False` | Multi-Query Attention for KV savings |

---

## Model Export

### Safetensors (recommended for fast loading)

```python
from rca import export_safetensors, load_safetensors, RCAModel

# Export
export_safetensors(model, "./my_model_safetensors/")

# Load
model = load_safetensors(RCAModel, "./my_model_safetensors/")
```

### GGUF (for llama.cpp / edge inference)

```python
from rca import export_gguf

# Full precision
export_gguf(model, "./my_model.gguf", quantization="f16")

# Quantized (smaller, faster on CPU)
export_gguf(model, "./my_model_q8.gguf", quantization="q8_0")
export_gguf(model, "./my_model_q4.gguf", quantization="q4_0")
```

**Quantization options:**

| Format | Size vs f32 | Quality | Use Case |
|---|---|---|---|
| `f32` | 1× | Lossless | Research / debugging |
| `f16` | 0.5× | Near-lossless | GPU inference |
| `q8_0` | 0.25× | Minimal loss | CPU / edge inference |
| `q4_0` | 0.125× | Some loss | Mobile / embedded |

### PyTorch Native Save/Load

```python
# Save
model.save_pretrained("./my_model/")

# Load
model = RCAModel.from_pretrained("./my_model/")
```

---

## Performance Features

### Gradient Checkpointing

Trades ~30% compute for ~60% memory savings. Enabled by default for 500M+ presets.

```python
config = RCAConfig.rca_500m()
# config.gradient_checkpointing is already True

# Or enable manually:
config.gradient_checkpointing = True
```

### Triton-Accelerated Parallel Scan

The SSM parallel scan automatically uses Triton kernels on NVIDIA GPUs:

```python
from rca import TRITON_AVAILABLE
print(f"Triton available: {TRITON_AVAILABLE}")
# Automatic — no code changes needed
```

### torch.compile

Fuses operations for additional speedup:

```python
args = TrainingArguments(
    use_torch_compile=True,
    compile_mode="reduce-overhead",  # or "max-autotune"
    # ...
)
```

### Fused RMSNorm

All normalization layers use an optimized `rsqrt(mean(x²))` implementation that is both faster and compatible with `torch.compile` kernel fusion.

---

## Kaggle / Colab Quick Training

Complete training script for free-tier GPUs:

```python
# Install
# !pip install rca-arch[gpu]

import torch
from rca import RCAConfig, RCAModel, RCATrainer, TrainingArguments

# Use 100M preset for T4
config = RCAConfig.rca_100m()
config.vocab_size = 32000

model = RCAModel(config)
print(f"Model: {model.count_parameters():,} params")
print(f"Zones: {model.get_layer_zones()}")

# Create a simple dataset (replace with your data)
from torch.utils.data import TensorDataset
data = torch.randint(0, 32000, (1000, 4097))
dataset = TensorDataset(data[:, :-1], data[:, 1:])

# Train
args = TrainingArguments(
    output_dir="/kaggle/working/rca_output",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=3e-4,
    warmup_steps=100,
    fp16=True,
    logging_steps=5,
    save_steps=200,
)

trainer = RCATrainer(model=model, args=args, train_dataset=dataset)
trainer.train()

# Export
from rca import export_safetensors
export_safetensors(model, "/kaggle/working/rca_safetensors/")
```

---

## Running Tests

```bash
# Run the full test suite
python run_tests.py

# Run a standalone integration test
python run_single_test.py
```

---

## Project Structure

```
src/rca/
├── __init__.py          # Public API
├── config.py            # RCAConfig with presets
├── modeling/
│   ├── rca_model.py     # RCAModel (SSM/GLA/Reasoning blocks)
│   └── outputs.py       # Output dataclasses
├── layers/
│   ├── ssm.py           # Selective State Space Model
│   ├── gla.py           # Gated Linear Attention (vectorized)
│   ├── sliding_attention.py  # Sliding Window + Memory Tokens
│   ├── attention.py     # Efficient Attention (MQA/Rotary)
│   ├── scan.py          # Parallel scan (PyTorch/Triton/XLA)
│   ├── norm.py          # Fused RMSNorm, DeepNorm
│   └── positions.py     # ALiBi, Rotary embeddings
├── trainer.py           # RCATrainer (DDP/FSDP/XLA/compile)
├── converter.py         # Safetensors + GGUF export
├── generator.py         # Text generation utilities
└── utils/
    ├── benchmark.py     # Performance benchmarking
    └── export.py        # ONNX export, save/load
```

---

## Citation

```bibtex
@software{rca2024,
  title={RCA: Recursive Compression Architecture},
  author={Rajaaditya, R.},
  year={2024},
  url={https://github.com/rajaaditya/rca-arch}
}
```

## License

MIT License — see [LICENSE](LICENSE) for details.
