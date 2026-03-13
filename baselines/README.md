# CHIMERA-Bench Baselines

This directory contains 11 antibody design methods retrained on CHIMERA-Bench under identical conditions.

## Methods

| Method | Dir | Paradigm | Venue | Numbering | Epi-Cond? | Multi-CDR? |
|--------|-----|----------|-------|-----------|:---------:|:----------:|
| DiffAb | `diffab/` | Diffusion | NeurIPS 2022 | Chothia | Yes | Yes |
| AbFlowNet | `abflownet/` | Flow matching | ICML 2025 | Chothia | Yes | Yes |
| AbMEGD | `abmedg/` | Diffusion | IJCAI 2025 | Chothia | Yes | Yes |
| RADAb | `radab/` | Retrieval + diffusion | ICLR 2025 | Chothia | Yes | Yes |
| dyAb | `dyab/` | Flow matching | AAAI 2025 | IMGT | Yes | Yes |
| MEAN | `mean/` | Equivariant GNN | ICLR 2023 | IMGT | Yes | No (H3) |
| dyMEAN | `dymean/` | Equivariant GNN | ICML 2023 | IMGT | Yes | Yes |
| RAAD | `raad/` | Equivariant GNN | AAAI 2025 | IMGT | Yes | Yes |
| RefineGNN | `refinegnn/` | Autoregressive GNN | ICLR 2022 | Chothia | No | Yes |
| AbODE | `abode/` | Conjoined ODE | ICML 2023 | Chothia | No | Yes |
| AbDockGen | `abdockgen/` | Hierarchical ENN | ICML 2022 | Chothia | Yes (H3) | No |

## Setup

```bash
# Set data root (required)
export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0
```

Each baseline also requires its own Python dependencies. See the baseline's original README for installation instructions.

## Per-Baseline Integration Files

Each baseline directory contains 5 CHIMERA integration files:

| File | Purpose |
|------|---------|
| `config.yaml` | Hyperparameters (inherits paths from `shared_config.yaml`) |
| `preprocess.py` | Converts CHIMERA data to baseline's native format |
| `chimera_trainer.py` | Training + test inference (`--split`, `--gpu`, `--test_only`) |
| `chimera_evaluate.py` | 12-metric evaluation (`--predictions` or `--aggregate`) |
| `chimera_train.sh` | End-to-end: preprocess, train on all splits, evaluate |

## Shared Infrastructure

- **`shared_config.yaml`**: Common paths (resolved from `CHIMERA_DATA_ROOT` env var), split names, evaluation config
- **`chimera_utils.py`**: Shared utilities -- `load_shared_config()`, `load_split_ids()`, `save_predictions()`, `run_full_evaluation()`, `EarlyStopping`, `ModelCheckpoint`, `setup_wandb()`

## Usage

### Full Pipeline (all splits)

```bash
cd baselines/diffab
bash chimera_train.sh
```

### Single Split

```bash
cd baselines/diffab

# 1. Preprocess
python preprocess.py

# 2. Train
python chimera_trainer.py --split epitope_group --gpu 0

# 3. Evaluate
python chimera_evaluate.py --aggregate
```

### Test-Only (with existing checkpoint)

```bash
python chimera_trainer.py --split epitope_group --test_only --checkpoint /path/to/best.pt
```

## Data Format Categories

Baselines use different input formats. Each `preprocess.py` converts from the unified CHIMERA format:

| Category | Format | Methods |
|----------|--------|---------|
| A | SAbDab-style LMDB | DiffAb, AbFlowNet, AbMEGD, RADAb, RAAD |
| B | JSONL | RefineGNN |
| C | PDB + FASTA | (IgGM, ProteinMPNN -- not included) |
| D | 6D coordinates | (AbSGM -- not included) |
| E | CDR-only | AbODE, AbDockGen |

## Prediction Format

All baselines save predictions as `.pt` files via `save_predictions()`:

```python
{
    "complex_id": str,         # e.g., "7zwi_E_F_D"
    "cdr_type": str,           # "H1", "H2", ..., "L3"
    "pred_sequence": str,      # predicted amino acid sequence
    "true_sequence": str,      # ground truth sequence
    "pred_coords": (L, 3),    # predicted CA coordinates
    "true_coords": (L, 3),    # ground truth CA coordinates
    "ppl": float,              # perplexity (0.0 for diffusion models)
}
```

## 12 CHIMERA Metrics

| Group | Metrics |
|-------|---------|
| Sequence | AAR, CAAR, PPL |
| Structure | RMSD (Kabsch CA), TM-score |
| Interface | Fnat, iRMSD, DockQ |
| Epitope | Epitope F1 (precision, recall) |
| Designability | n_liabilities |
| Composites | CHIMERA-S, CHIMERA-B |

## Notes

- **Model checkpoints** are not included in this repository. Train from scratch using the provided scripts, or download pre-trained weights from the original baseline repositories.
- **WandB logging** is enabled by default (project: `chimera_baselines`). Use `--no-wandb` to disable.
- **Early stopping patience** is standardized to 10 epochs across all baselines.
- Baselines not included (no CHIMERA integration yet): AbX, AbEgDiffuser, LEAD, IgGM, DiffAbXL, AbSGM, ProteinMPNN.
