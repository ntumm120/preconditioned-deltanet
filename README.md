# Preconditioned DeltaNet

Training code for [*Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences*](https://arxiv.org/abs/2604.21100).

This repo provides model configs and training scripts to train the models from the paper. It bundles a fork of [FLA](https://github.com/fla-org/flash-linear-attention) (with our preconditioned recurrence Triton kernels) and uses [flame](https://github.com/fla-org/flame) to train them. 

> **Note:** A PR to merge preconditioned DeltaNet into upstream [FLA](https://github.com/fla-org/flash-linear-attention) is in progress, including integration of our preconditioned recurrence with the Tilelang GDN and FlashKDA backends. Once merged, upstream FLA will be the de facto reference implementation for these models, and you can install FLA directly instead of using the bundled fork. This repo will remain useful for the training configs and scripts.

## Where the preconditioned recurrences live

Our contributions in the bundled FLA fork at `3rdparty/flash-linear-attention/`:

- **ATK preconditioner kernels** (shared by PGDN and PKDA): [`fla/ops/atk/`](3rdparty/flash-linear-attention/fla/ops/atk/)
- **Preconditioned Gated DeltaNet (PGDN) kernels**: [`fla/ops/precond_gated_delta_rule/`](3rdparty/flash-linear-attention/fla/ops/precond_gated_delta_rule/) — asymmetric KKT forward, WY backward, fused recurrent / chunk forms
- **Preconditioned KDA (PKDA) kernels**: [`fla/ops/precond_kda/`](3rdparty/flash-linear-attention/fla/ops/precond_kda/) — asymmetric intra-chunk, fused backward, fused recurrent / chunk forms
- **Layer modules**: [`fla/layers/precond_gated_deltanet.py`](3rdparty/flash-linear-attention/fla/layers/precond_gated_deltanet.py), [`fla/layers/precond_kda.py`](3rdparty/flash-linear-attention/fla/layers/precond_kda.py)
- **HuggingFace models**: [`fla/models/precond_gated_deltanet/`](3rdparty/flash-linear-attention/fla/models/precond_gated_deltanet/), [`fla/models/precond_kda/`](3rdparty/flash-linear-attention/fla/models/precond_kda/)
- **Op tests**: [`tests/ops/test_precond_gated_delta.py`](3rdparty/flash-linear-attention/tests/ops/test_precond_gated_delta.py), [`tests/ops/test_precond_kda.py`](3rdparty/flash-linear-attention/tests/ops/test_precond_kda.py)

Configs and training scripts for both the baseline (GDN, KDA) and preconditioned (PGDN, PKDA) models live in [`configs/`](configs/) and [`scripts/`](scripts/) in this repo.

## Installation

```bash
# Clone with all contents
git clone https://github.com/ntumm120/preconditioned-deltanet.git
cd preconditioned-deltanet

# Install the FLA fork
pip install 3rdparty/flash-linear-attention

# Install flame dependencies
pip install 3rdparty/flame
```

## Training

All training scripts are in `scripts/`. Each script launches distributed training using [flame](https://github.com/fla-org/flame). The dataset ([SlimPajama-627B](https://huggingface.co/datasets/gmongaras/SlimPajama-627B_Reupload)) streams from HuggingFace automatically.

```bash
# Basic usage (8 GPUs, outputs to exp/<model_name>/)
bash scripts/train_precond_gated_deltanet_340M.sh
```

Each script accepts the following optional arguments:

| Argument | Description | Default |
|----------|-------------|---------|
| `--output DIR` | Checkpoint and log output directory | `exp/<model_name>/` |
| `--wandb-project NAME` | Weights & Biases project name | *(disabled)* |
| `--ngpu N` | Number of GPUs | `8` |

```bash
# Custom output directory and W&B logging
bash scripts/train_precond_kda_1B.sh --output /data/experiments/pkda --wandb-project my-project

# Use 4 GPUs instead of 8
bash scripts/train_gated_deltanet_340M.sh --ngpu 4
```

Outputs (checkpoints, logs) are saved to `exp/<model_name>/` by default, or to the directory specified by `--output`. The environment variables `NNODE`, `NGPU`, and `LOG_RANK` can also be set externally and will take precedence over script defaults.

## Model Configs

| Config | Model Type | Size | Hidden | Heads | Head Dim | Layers |
|--------|-----------|------|--------|-------|----------|--------|
| `gated_deltanet_340M` | Gated DeltaNet | 340M | 1024 | 8 | 128 | 24 |
| `gated_deltanet_1B` | Gated DeltaNet | 1B | 1792 | 14 | 128 | 24 |
| `kda_340M` | KDA | 355M* | 1024 | 8 | 128 | 24 |
| `kda_1B` | KDA | 1B* | 1792 | 14 | 120 | 24 |
| `precond_gated_deltanet_340M` | Precond. Gated DeltaNet | 340M | 1024 | 8 | 128 | 24 |
| `precond_gated_deltanet_1B` | Precond. Gated DeltaNet | 1B | 1792 | 14 | 128 | 24 |
| `precond_kda_340M` | Precond. KDA | 355M* | 1024 | 8 | 128 | 24 |
| `precond_kda_1B` | Precond. KDA | 1B* | 1792 | 14 | 120 | 24 |

\* KDA and Precond. KDA configs have ~15M additional parameters from the output gate. Reducing the head dimension such that the KDA model has 340M params reduces throughput as kernel is optimized for strides of size 16. See paper for more details. 

### Training Hyperparameters

| | 340M (15B tokens) | 1B (50B tokens) |
|---|---|---|
| Steps | 30,000 | 95,000 |
| Batch size | 16 | 8 |
| Gradient accumulation | 2 | 4 |
| Sequence length | 2,048 | 2,048 |
| Learning rate | 4e-4 | 4e-4 |
| Warmup steps | 1,024 | 1,024 |
| Weight decay | 0.01 | 0.01 |
| GPUs | 8 | 8 |

## Citation

```bibtex
@misc{tumma2026preconditioneddeltanetcurvatureawaresequence,
      title={Preconditioned DeltaNet: Curvature-aware Sequence Modeling for Linear Recurrences},
      author={Neehal Tumma and Noel Loo and Daniela Rus},
      year={2026},
      eprint={2604.21100},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2604.21100},
}
```

## Acknowledgements

This repo builds on [FLA](https://github.com/fla-org/flash-linear-attention) and [flame](https://github.com/fla-org/flame). We thank the FLA team for their excellent infrastructure.
