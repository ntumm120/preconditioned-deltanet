#!/usr/bin/env python3
"""Generate 8 experiment configs for x sweep ablation (no m_norm)."""
import json
from pathlib import Path

base_config = {
    "model_type": "precond_gated_deltanet",
    "attn_mode": "chunk",
    "hidden_size": 1024,
    "num_heads": 8,
    "head_dim": 128,
    "expand_v": 1.0,
    "num_hidden_layers": 24,
    "hidden_ratio": 4,
    "use_gate": False,
    "use_short_conv": True,
    "conv_size": 4,
    "fuse_norm": True,
    "fuse_swiglu": True,
    "fuse_cross_entropy": True,
    "vocab_size": 32000,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "tie_word_embeddings": True,
    "initializer_range": 0.02,
    "norm_eps": 1e-06,
    "qk_norm": True,
    "k_precond_norm": True,
    "fix_r_one": True,
    "fixed_lambda": 0.1,
    "dt_bias_offset": 0.0,
    "untie": True,
    "squash_eps": 1e-06,
    "precond_mode": "symmetric",
    "log_A_scale_init": -0.2,  # init0p2
    "use_m_norm": False,  # NO m_norm for all runs
}

# 8 experiments
experiments = [
    # x sweep - fast (3 runs)
    {"name": "fast_x1p25", "squash_type": "fast", "x": 1.25, "learn_scale": True, "learn_gain": False},
    {"name": "fast_x1p75", "squash_type": "fast", "x": 1.75, "learn_scale": True, "learn_gain": False},
    {"name": "fast_x2", "squash_type": "fast", "x": 2.0, "learn_scale": True, "learn_gain": False},

    # x sweep - smooth (3 runs)
    {"name": "smooth_x1p25", "squash_type": "smooth", "x": 1.25, "learn_scale": True, "learn_gain": False},
    {"name": "smooth_x1p75", "squash_type": "smooth", "x": 1.75, "learn_scale": True, "learn_gain": False},
    {"name": "smooth_x2", "squash_type": "smooth", "x": 2.0, "learn_scale": True, "learn_gain": False},

    # Ablations (2 runs)
    {"name": "fast_x1p5_no_learn_scale", "squash_type": "fast", "x": 1.5, "learn_scale": False, "learn_gain": False},
    {"name": "fast_x1p5_learn_gain", "squash_type": "fast", "x": 1.5, "learn_scale": True, "learn_gain": True},
]

config_dir = Path(__file__).parent

for exp in experiments:
    config = base_config.copy()
    config["squash_type"] = exp["squash_type"]
    config["squash_x"] = exp["x"]
    config["learn_scale"] = exp["learn_scale"]
    config["learn_gain"] = exp["learn_gain"]

    config_path = config_dir / f"{exp['name']}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Created {config_path.name}")

print(f"\nCreated {len(experiments)} configs")
