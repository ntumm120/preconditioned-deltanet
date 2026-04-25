# Baseline Model Configurations

This directory contains two sets of baseline configurations for comparing different linear attention architectures at ~340M parameters.

## Configuration Sets

### 1. `deltanet_paper_config/` - No Output Gate (~4H² attention)

Based on the original DeltaNet paper architecture without output gating.

| Setting | Value |
|---------|-------|
| hidden_size | 1024 |
| num_heads | 8 |
| head_dim | 128 |
| key_dim | 1024 (= H) |
| value_dim | 1024 (= H) |
| expand_v | 1.0 |
| output_gate | No |
| num_layers | 24 |
| tie_word_embeddings | Yes |

**Attention params per layer:** ~4H² (q, k, v, o projections each H×H)

### 2. `comba_paper_config/` - With Output Gate (~4H² attention)

Based on the Comba/GatedDeltaNet paper architecture with output gating. Uses parameter reallocation to maintain the same total param count as the no-gate config.

| Setting | Value |
|---------|-------|
| hidden_size | 1024 |
| num_heads | 4 |
| head_dim | 128 |
| key_dim | 512 (= 0.5H) |
| value_dim | 1024 (= H) |
| expand_v | 2.0 |
| output_gate | Yes |
| num_layers | 24 |
| tie_word_embeddings | Yes |

**Attention params per layer:** ~4H² (smaller q, k; larger v, g, o)

## Parameter Reallocation Math

To add an output gate while maintaining the same parameter budget:

**Without gate (4H²):**
- q_proj: H × H = H²
- k_proj: H × H = H²
- v_proj: H × H = H²
- o_proj: H × H = H²
- **Total: 4H²**

**With gate, using multipliers α (key) and β (value):**
- q_proj: H × αH = αH²
- k_proj: H × αH = αH²
- v_proj: H × βH = βH²
- g_proj: H × βH = βH²
- o_proj: βH × H = βH²
- **Total: 2α + 3β**

To match 4H² while preserving the docstring's architectural ratio (β = 2α):
```
2α + 3(2α) = 4
8α = 4
α = 0.5, β = 1.0
```

This gives:
- key_dim = 0.5 × 1024 = 512
- value_dim = 1.0 × 1024 = 1024

## Parameter Counts

| Model | DeltaNet Paper Config | Comba Paper Config |
|-------|----------------------|-------------------|
| delta_net | 341.6M | 341.4M |
| gated_deltanet | 341.8M | 341.5M |
| comba | 341.8M | 341.5M |
| kda | 354.2M | 338.3M |

Note: KDA has a different architecture (bottleneck MLPs for gates) which causes slight param count differences.

## Usage

These configs can be loaded directly with the FLA library:

```python
import json
from fla.models import GatedDeltaNetConfig, GatedDeltaNetForCausalLM

with open('comba_paper_config/gated_deltanet.json') as f:
    config = GatedDeltaNetConfig(**json.load(f))
model = GatedDeltaNetForCausalLM(config)
```
