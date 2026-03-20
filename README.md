# RCA Architecture

RCA (Ultra-Reasoning Continuous Architecture) is a bleeding-edge, highly efficient hybrid sequence model framework. It is expressly designed to radically solve the quadratic attention bottleneck of traditional Transformers. 

By unifying linear-time State Space Models (SSM) and Gated Linear Attention with highly targeted local Sliding Window Attention, RCA provides an effectively infinite context window, flat constant-time generation speeds, and transformer-level reasoning capabilities.

## Key Capabilities
- **O(1) Memory Footprint:** The vast majority of the network runs natively in a compressed state matrix, allowing you to process millions of tokens using a flat, constant amount of memory.
- **Ultra-Fast Generation:** Because historical context does not continuously grow the Key-Value cache, text generation speed is significantly faster than standard Transformers at long contexts.
- **Long-Term Reasoning:** Highly targeted local attention modules maintain sharp, specific reasoning facts that pure linear models natively struggle with. It is perfectly tuned for multi-turn agentic workflows.

## Installation

You can install the official package directly from PyPI:
```bash
pip install rca-arch
```

## Quick Start & Usage

Using the model in your PyTorch workflow is incredibly straightforward. Define a configuration based on our scaled presets, and initialize the architecture:

```python
import torch
from rca import RCAConfig, RCAModel

# 1. Select your target scale
# Available presets: "rca_tiny", "rca_small", "rca_base", "rca_large", "rca_xl"
config = RCAConfig.from_preset("rca_small")
config.vocab_size = 32000

# 2. Initialize the model
model = RCAModel(config)

# 3. Standard forward pass
x = torch.randint(0, 32000, (2, 1024)) # [batch_size, sequence_length]
logits, cache = model(x)

print("Output shape:", logits.shape)
```

## Running Tests & Verifications

If you clone the repository from GitHub, you can execute the comprehensive test suite to verify the custom kernel behavior, chunkwise parallel processing, and mathematical stability.

```bash
# Run the entire test suite across all architecture layers (SSM, GLA, Slidng Window)
python run_tests.py

# Run a localized standalone integration test
python run_single_test.py
```
