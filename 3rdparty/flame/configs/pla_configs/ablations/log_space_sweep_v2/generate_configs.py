#!/usr/bin/env python3
"""Generate all 12 experiment configs for log_space_sweep_v2."""
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
    "learn_scale": True,
    "learn_gain": False,
    "use_m_norm": True
}

# 12 experiments
experiments = [
    # Squash ablation (old init)
    {"name": "amp_x2_smooth", "mode": "squash_amplify", "squash_type": "smooth", "x": 2.0, "init": -1.0},
    {"name": "sym_x1p5_smooth", "mode": "squash_symmetric", "squash_type": "smooth", "x": 1.5, "init": -1.0},

    # Init ablation (fast squash)
    {"name": "amp_x2_fast_init0p2", "mode": "squash_amplify", "squash_type": "fast", "x": 2.0, "init": -0.2},
    {"name": "sym_x1p5_fast_init0p2", "mode": "squash_symmetric", "squash_type": "fast", "x": 1.5, "init": -0.2},

    # x sweep amplify (fast, better init)
    {"name": "amp_x3_fast_init0p2", "mode": "squash_amplify", "squash_type": "fast", "x": 3.0, "init": -0.2},
    {"name": "amp_x4_fast_init0p2", "mode": "squash_amplify", "squash_type": "fast", "x": 4.0, "init": -0.2},
    {"name": "amp_x6_fast_init0p2", "mode": "squash_amplify", "squash_type": "fast", "x": 6.0, "init": -0.2},

    # x sweep symmetric (fast, better init)
    {"name": "sym_x2_fast_init0p2", "mode": "squash_symmetric", "squash_type": "fast", "x": 2.0, "init": -0.2},
    {"name": "sym_x2p5_fast_init0p2", "mode": "squash_symmetric", "squash_type": "fast", "x": 2.5, "init": -0.2},
    {"name": "sym_x3_fast_init0p2", "mode": "squash_symmetric", "squash_type": "fast", "x": 3.0, "init": -0.2},

    # Best combo candidates (smooth + better init)
    {"name": "amp_x2_smooth_init0p2", "mode": "squash_amplify", "squash_type": "smooth", "x": 2.0, "init": -0.2},
    {"name": "sym_x1p5_smooth_init0p2", "mode": "squash_symmetric", "squash_type": "smooth", "x": 1.5, "init": -0.2},
]

config_dir = Path(__file__).parent

for exp in experiments:
    config = base_config.copy()
    config["precond_mode"] = exp["mode"]
    config["squash_type"] = exp["squash_type"]
    config["squash_x"] = exp["x"]
    config["log_A_scale_init"] = exp["init"]

    config_path = config_dir / f"{exp['name']}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Created {config_path.name}")

print(f"\nCreated {len(experiments)} configs")
