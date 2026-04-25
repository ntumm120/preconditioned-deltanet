#!/usr/bin/env python3
"""Generate 15 experiment configs for m_norm ablation."""
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
}

# 15 experiments for m_norm ablation
experiments = [
    # === Fast (7 runs) ===
    # 1. fast_x1p5_m_norm
    {"name": "fast_x1p5_m_norm", "squash_type": "fast", "x": 1.5,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # 2. fast_x1p5_curvature_m_norm
    {"name": "fast_x1p5_curvature_m_norm", "squash_type": "fast", "x": 1.5,
     "use_m_norm": True, "curvature_aware_m_norm": True, "learn_scale": True, "learn_gain": False},

    # 3. fast_x2_m_norm
    {"name": "fast_x2_m_norm", "squash_type": "fast", "x": 2.0,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # 4. fast_x2_curvature_m_norm
    {"name": "fast_x2_curvature_m_norm", "squash_type": "fast", "x": 2.0,
     "use_m_norm": True, "curvature_aware_m_norm": True, "learn_scale": True, "learn_gain": False},

    # 5. fast_x1p5_m_norm_no_learn_scale
    {"name": "fast_x1p5_m_norm_no_learn_scale", "squash_type": "fast", "x": 1.5,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": False, "learn_gain": False},

    # 6. fast_x1p5_m_norm_learn_gain
    {"name": "fast_x1p5_m_norm_learn_gain", "squash_type": "fast", "x": 1.5,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": True},

    # 7. fast_x3_m_norm
    {"name": "fast_x3_m_norm", "squash_type": "fast", "x": 3.0,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # === Smooth (8 runs) ===
    # 8. smooth_x1p5_m_norm
    {"name": "smooth_x1p5_m_norm", "squash_type": "smooth", "x": 1.5,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # 9. smooth_x2_m_norm
    {"name": "smooth_x2_m_norm", "squash_type": "smooth", "x": 2.0,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # 10. smooth_x3_m_norm
    {"name": "smooth_x3_m_norm", "squash_type": "smooth", "x": 3.0,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # 11. smooth_x4_m_norm
    {"name": "smooth_x4_m_norm", "squash_type": "smooth", "x": 4.0,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # 12. smooth_x5_m_norm
    {"name": "smooth_x5_m_norm", "squash_type": "smooth", "x": 5.0,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": False},

    # 13. smooth_x2_curvature_m_norm
    {"name": "smooth_x2_curvature_m_norm", "squash_type": "smooth", "x": 2.0,
     "use_m_norm": True, "curvature_aware_m_norm": True, "learn_scale": True, "learn_gain": False},

    # 14. smooth_x3_curvature_m_norm
    {"name": "smooth_x3_curvature_m_norm", "squash_type": "smooth", "x": 3.0,
     "use_m_norm": True, "curvature_aware_m_norm": True, "learn_scale": True, "learn_gain": False},

    # 15. smooth_x1p5_m_norm_learn_gain
    {"name": "smooth_x1p5_m_norm_learn_gain", "squash_type": "smooth", "x": 1.5,
     "use_m_norm": True, "curvature_aware_m_norm": False, "learn_scale": True, "learn_gain": True},
]

config_dir = Path(__file__).parent

for exp in experiments:
    config = base_config.copy()
    config["squash_type"] = exp["squash_type"]
    config["squash_x"] = exp["x"]
    config["use_m_norm"] = exp["use_m_norm"]
    config["curvature_aware_m_norm"] = exp["curvature_aware_m_norm"]
    config["learn_scale"] = exp["learn_scale"]
    config["learn_gain"] = exp["learn_gain"]

    config_path = config_dir / f"{exp['name']}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"Created {config_path.name}")

print(f"\nCreated {len(experiments)} configs")
