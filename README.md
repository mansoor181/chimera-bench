# CHIMERA-Bench

A benchmark dataset for **epitope-specific antibody design**.

[![CI](https://github.com/mansoor181/chimera-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/mansoor181/chimera-bench/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Data: CC BY 4.0](https://img.shields.io/badge/Data-CC%20BY%204.0-green.svg)](LICENSE-DATA)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-HuggingFace-orange)](https://huggingface.co/datasets/mansoorbaloch/chimera-bench)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mansoor181/chimera-bench/blob/main/notebooks/demo.ipynb)

**Paper**: [CHIMERA-Bench: A Benchmark Dataset for Epitope-Specific Antibody Design](https://arxiv.org/abs/2603.13431v3) (ICLR 2026 GEM Workshop - [openreview](https://openreview.net/forum?id=PyZvVIJbSy))

```bash
pip install -e .                 # install the chimera_bench package
pytest tests/ -q                 # smoke-test against the committed sample (no download)
```

```python
import chimera_bench as cb       # set CHIMERA_DATA_ROOT first
split = cb.load_split("epitope_group")
feat = cb.load_complex(split["test"][0])
```

## Overview

CHIMERA-Bench provides:
- **2,922 antibody-antigen complexes** with epitope/paratope annotations, multi-scheme numbering (IMGT + Chothia), and pre-computed structural features
- **3 generalization splits**: epitope-group, antigen-fold, temporal
- **12 evaluation metrics** spanning sequence quality, structural accuracy, binding interface, and epitope specificity
- **11 retrained baselines** across 6 design paradigms, all evaluated under identical conditions
- **Format converters** for 5 data categories used by different baseline methods

| Property | Value |
|----------|-------|
| Complexes | 2,922 |
| PDB structures | 2,721 |
| Pre-computed features | 2,922 `.pt` files |
| Splits | 3 (epitope-group, antigen-fold, temporal) |
| Numbering schemes | IMGT, Chothia |
| Baselines evaluated | 11 methods, 6 paradigms |
| Resolution cutoff | 4.0 A |
| Contact cutoff | 4.5 A |

## Leaderboard

Primary task: **CDR-H3 design on the epitope-group split** (best-of-K, mean±std
over the test set). Full results for all methods, splits, and CDR types are in
[`leaderboard.csv`](leaderboard.csv). Regenerate with
`python scripts/build_leaderboard.py --results results/`.

| Rank | Method | AAR | CAAR | RMSD | Fnat | DockQ | Epitope F1 | CHIMERA-S | CHIMERA-B |
|---:|---|---|---|---|---|---|---|---|---|
| 1 | MEAN | 0.42±0.15 | 0.21±0.23 | 2.01±0.89 | 0.48±0.31 | 0.63±0.21 | 0.67±0.29 | 0.47±0.11 | 0.48±0.22 |
| 2 | RAAD | 0.38±0.13 | 0.21±0.23 | 1.95±0.87 | 0.49±0.30 | 0.64±0.21 | 0.67±0.27 | 0.48±0.11 | 0.48±0.20 |
| 3 | AbODE | 0.31±0.13 | 0.22±0.21 | 16.40±4.67 | 0.08±0.12 | 0.34±0.08 | 0.20±0.19 | 0.12±0.03 | 0.13±0.11 |
| 4 | dyAb | 0.27±0.10 | 0.12±0.17 | 3.31±0.84 | 0.35±0.28 | 0.54±0.18 | 0.50±0.32 | 0.37±0.07 | 0.35±0.23 |
| 5 | AbDockGen | 0.25±0.11 | 0.10±0.18 | 3.97±1.25 | 0.35±0.27 | 0.52±0.17 | 0.63±0.26 | 0.34±0.07 | 0.42±0.19 |
| 6 | RefineGNN | 0.23±0.12 | 0.17±0.23 | 3.07±0.72 | 0.62±0.20 | 0.71±0.09 | 0.76±0.10 | 0.44±0.06 | 0.57±0.09 |
| 7 | AbFlowNet | 0.22±0.12 | 0.13±0.18 | 2.70±1.24 | 0.49±0.33 | 0.57±0.22 | 0.55±0.28 | 0.41±0.11 | 0.42±0.22 |
| 8 | AbMEGD | 0.22±0.11 | 0.11±0.18 | 2.76±1.26 | 0.50±0.32 | 0.57±0.22 | 0.56±0.27 | 0.41±0.11 | 0.43±0.21 |
| 9 | DiffAb | 0.22±0.12 | 0.11±0.18 | 2.64±1.19 | 0.48±0.32 | 0.58±0.23 | 0.56±0.27 | 0.42±0.11 | 0.42±0.21 |
| 10 | RADAb | 0.22±0.12 | 0.09±0.16 | 12.28±75.62 | 0.47±0.32 | 0.57±0.22 | 0.57±0.28 | 0.40±0.12 | 0.43±0.21 |

To submit a method, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Dataset

The dataset is hosted on HuggingFace Hub and Zenodo:

- **HuggingFace**: [`mansoorbaloch/chimera-bench`](https://huggingface.co/datasets/mansoorbaloch/chimera-bench)
- **Zenodo**: [DOI: 20598827](https://zenodo.org/records/20598827)
- **Pre-computed residue graphs** (optional, 5.8 GB): available as a separate download

### Download

```bash
# Set the data root (used by all scripts)
export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0

# Option 1: HuggingFace CLI
hf download mansoorbaloch/chimera-bench --repo-type dataset --local-dir $CHIMERA_DATA_ROOT

# Option 2: Direct download from Zenodo
# wget <20598827> -O chimera-bench-v1.0.zip && unzip chimera-bench-v1.0.zip
```

### Dataset Structure

```
chimera-bench-v1.0/
  metadata/
    final_summary.csv           # 2,922 complexes with 32 columns
    excluded_complexes.csv      # 59 excluded complexes with reasons
    antibody_sequences.fasta    # VH+VL sequences for all complexes
  splits/
    epitope_group.json          # Primary split (2338/292/292)
    antigen_fold.json           # Fold-based generalization (2338/292/292)
    temporal.json               # Prospective evaluation (2337/292/293)
  complex_features/             # Per-complex PyTorch tensors (2,922 files)
    {complex_id}.pt
  structures/                   # PDB structure files (2,721 files)
    {pdb}.pdb
```

### Complex Features Format

Each `.pt` file contains:

| Key | Shape | Description |
|-----|-------|-------------|
| `complex_id` | str | Unique ID: `{pdb}_{Hchain}_{Lchain}_{Agchain}` |
| `heavy_sequence` | str | Heavy chain amino acid sequence |
| `light_sequence` | str | Light chain amino acid sequence |
| `antigen_sequence` | str | Antigen amino acid sequence |
| `heavy_atom14_coords` | (N, 14, 3) | Heavy chain 14-atom coordinates |
| `heavy_ca_coords` | (N, 3) | Heavy chain CA coordinates |
| `epitope_residues` | list | (chain, resid, resname) tuples |
| `paratope_residues` | list | (chain, resid, resname) tuples |
| `contact_pairs` | list | Ab-Ag contact pairs with distances |
| `numbering` | dict | IMGT and Chothia numbering for H and L chains |
| `cdr_masks` | dict | Per-residue CDR annotations (-1=FR, 0-2=H1-H3, 3-5=L1-L3) |
| `ag_surface_points` | (128, 3) | Sampled antigen surface points |
| `ag_surface_chemical_feats` | (128, 6) | Hydropathy, charge, H-bond, aromaticity, polarity |

See [`notebooks/demo.ipynb`](notebooks/demo.ipynb) for a complete walkthrough
([open in Colab](https://colab.research.google.com/github/mansoor181/chimera-bench/blob/main/notebooks/demo.ipynb)).

## Installation

### Minimal (data loading + evaluation)

The `chimera_bench` package only needs numpy, pandas, scipy, and PyTorch:

```bash
pip install -e .          # editable install of the chimera_bench package
pip install -e ".[test]"  # also installs pytest for the test suite
```

This is enough to load the dataset, run evaluation, and reproduce the
leaderboard. The smoke tests run against the committed `sample_data/`, so no
dataset download is required:

```bash
pytest tests/ -q
```

### Full (rebuilding the dataset or training baselines)

Rebuilding the dataset (`pipeline.py`) or training baselines needs conda tools
(ANARCI) and ML packages, installed in a specific order to avoid conflicts.

### Setup script (recommended)

```bash
# Default: CUDA 12.1, env name "chimera-bench"
bash setup_env.sh

# Custom CUDA version
CUDA_VERSION=11.8 bash setup_env.sh

# Custom env name
bash setup_env.sh my-env-name

# Activate
conda activate chimera-bench
```

### Manual installation

```bash
# 1. Create conda env (base packages + bioconda tools)
conda env create -f environment.yml
conda activate chimera-bench

# 2. Install PyTorch (must be before PyG)
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Install PyTorch Geometric
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# 4. Install bioconda tools (ANARCI for antibody numbering)
conda install -y -c bioconda "muscle<5" anarci

# 5. Install protein tools and ML utilities
pip install "fair-esm>=2.0.0" antiberty DockQ
pip install wandb hydra-core easydict lmdb loguru einops
```

### Baseline-specific dependencies

Each baseline may have additional dependencies (e.g., OpenMM, ESM-2 weights, MSA Transformer). See the individual baseline directories and their original READMEs for details.

## Quick Start

Point the loaders at your dataset once, then use the `chimera_bench` API:

```bash
export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0
# Or, to try the API with no download, point at the committed sample:
# export CHIMERA_DATA_ROOT=$(pwd)/sample_data
```

```python
import chimera_bench as cb

# Splits and complexes
split = cb.load_split("epitope_group")            # {"train", "val", "test"}
feat = cb.load_complex(split["test"][0])          # per-complex feature dict
print(feat["complex_id"], feat["heavy_sequence"][:20], "...")
print("Epitope:", len(feat["epitope_residues"]), "residues")

# CDR-H3 residue indices (IMGT) in the concatenated heavy+light chain
h3_idx = cb.cdr_indices(feat, "H3", scheme="imgt")

# A torch Dataset for training
from torch.utils.data import DataLoader
ds = cb.ChimeraDataset("epitope_group", "train")
loader = DataLoader(ds, batch_size=1, collate_fn=lambda b: b)
```

Evaluate predictions (one `.pt` per complex, see [CONTRIBUTING.md](CONTRIBUTING.md)):

```bash
chimera-eval --predictions preds/ --split epitope_group --cdr-type H3
# equivalently: python -m chimera_bench.evaluate --predictions preds/ ...
```

## Baselines

11 methods retrained on CHIMERA-Bench across 6 paradigms:

| Method | Paradigm | Epi-Cond? | Multi-CDR? | Directory | Upstream |
|--------|----------|:---------:|:----------:|-----------|----------|
| DiffAb | Diffusion | Yes | Yes | `baselines/diffab/` | [luost26/diffab](https://github.com/luost26/diffab) |
| AbFlowNet | Flow matching | Yes | Yes | `baselines/abflownet/` | [Patchwork53/abflownet](https://github.com/Patchwork53/abflownet) |
| AbMEGD | Diffusion | Yes | Yes | `baselines/abmedg/` | [Patrick221215/AbMEGD](https://github.com/Patrick221215/AbMEGD) |
| RADAb | Retrieval + diffusion | Yes | Yes | `baselines/radab/` | [GENTEL-lab/RADAb](https://github.com/GENTEL-lab/RADAb) |
| dyAb | Flow matching | Yes | Yes | `baselines/dyab/` | [A4Bio/dyAb](https://github.com/A4Bio/dyAb) |
| MEAN | Equivariant GNN | Yes | No (H3) | `baselines/mean/` | [THUNLP-MT/MEAN](https://github.com/THUNLP-MT/MEAN) |
| dyMEAN | Equivariant GNN | Yes | Yes | `baselines/dymean/` | [THUNLP-MT/dyMEAN](https://github.com/THUNLP-MT/dyMEAN) |
| RAAD | Equivariant GNN | Yes | Yes | `baselines/raad/` | [LirongWu/RAAD](https://github.com/LirongWu/RAAD) |
| RefineGNN | Autoregressive GNN | No | Yes | `baselines/refinegnn/` | [wengong-jin/RefineGNN](https://github.com/wengong-jin/RefineGNN) |
| AbODE | Conjoined ODE | No | Yes | `baselines/abode/` | [Aalto-QuML/AbODE](https://github.com/Aalto-QuML/AbODE) |
| AbDockGen | Hierarchical ENN | Yes (H3) | No | `baselines/abdockgen/` | [wengong-jin/abdockgen](https://github.com/wengong-jin/abdockgen) |

Each baseline includes 5 CHIMERA integration files:
- `config.yaml` -- hyperparameters
- `preprocess.py` -- converts CHIMERA data to baseline's native format
- `chimera_trainer.py` -- training and test inference
- `chimera_evaluate.py` -- 12-metric evaluation
- `chimera_train.sh` -- end-to-end orchestration

### Retraining a Baseline

```bash
export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0

# 1. Preprocess data into baseline's native format
cd baselines/diffab
python preprocess.py

# 2. Train and evaluate on all splits
bash chimera_train.sh

# Or train on a single split
python chimera_trainer.py --split epitope_group --gpu 0
python chimera_evaluate.py --aggregate
```

## Evaluation Metrics

| Group | Metrics | Description |
|-------|---------|-------------|
| Sequence quality | AAR, CAAR, PPL | Amino acid recovery, contact AAR, perplexity |
| Structural accuracy | RMSD, TM-score | Kabsch-aligned CA RMSD, TM-score |
| Binding interface | Fnat, iRMSD, DockQ | Fraction native contacts, interface RMSD, DockQ |
| Epitope specificity | EpiF1 | Precision, recall, F1 for epitope contacts |
| Designability | n_liabilities | Count of NG, DG, DS, DD, NS, NT, M motifs |

Two composite scores summarize structure (CHIMERA-S) and binding (CHIMERA-B).
Run the official evaluator with:

```bash
chimera-eval --predictions preds/ --split epitope_group --cdr-type H3 \
    --output results/mymethod/epitope_group/H3/results.json
```

Notes:
- Interface and epitope metrics are CDR-specific when `--cdr-type` is set: only
  contacts where the antibody partner is a CDR residue are counted, since
  framework contacts are trivially preserved and would dominate the scores.
- TM-score uses a fast Kabsch-based approximation. For camera-ready numbers,
  rescore with the official `TMscore` binary.
- Prediction files load with `weights_only=False` (needed for numpy arrays);
  only evaluate prediction files you trust. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Repository Structure

```
chimera-bench/
  README.md                     # This file
  CONTRIBUTING.md               # How to submit a method / add a baseline
  LICENSE / LICENSE-DATA        # MIT (code) / CC-BY 4.0 (data)
  DATASHEET.md                  # Datasheet for Datasets (Gebru et al.)
  pyproject.toml                # Installable chimera_bench package
  environment.yml / setup_env.sh / requirements.txt   # Full env setup
  leaderboard.csv               # Published results (all methods/splits/CDRs)
  sample_data/                  # 12-complex sample (demo + tests, no download)

  chimera_bench/                # >>> Installable consumer package <<<
    data.py                     # load_split, load_complex, ChimeraDataset, ...
    metrics.py                  # All 12 metrics
    evaluate.py                 # Evaluation entry point (chimera-eval)

  tests/                        # Smoke tests (run against sample_data)
  scripts/
    build_leaderboard.py        # Aggregate results/ into leaderboard.csv
  notebooks/
    demo.ipynb                  # Dataset exploration (Colab-ready)
    analysis.ipynb              # Visualization and analysis
    eda.ipynb                   # Exploratory data analysis

  config.py                     # Construction config (dataset rebuild only)
  pipeline.py                   # Dataset construction pipeline
  data/                         # Construction modules (collect/filter/dedup/...)
  converters/                   # Format converters (5 categories)
  evaluation/
    contamination.py            # PLM training data overlap audit
  baselines/                    # 11 retrained baseline methods
    chimera_utils.py            # Shared utilities
    shared_config.yaml          # Shared paths and settings
    diffab/ ...                 # Each baseline has its own directory
```

> The `chimera_bench/` package is the only thing installed by `pip install -e .`.
> The construction modules (`config.py`, `pipeline.py`, `data/`, `converters/`)
> use generic top-level names and are run from the repo root, kept separate so
> they do not collide with the baselines' own `data/` and `evaluation/` packages.

## Splits

| Split | Train | Val | Test | Generalization axis |
|-------|------:|----:|-----:|---------------------|
| epitope_group | 2,338 | 292 | 292 | Unseen epitope patterns (primary) |
| antigen_fold | 2,338 | 292 | 292 | Unseen antigen folds |
| temporal | 2,337 | 292 | 293 | Prospective (by deposition date) |

Clusters are assigned whole to a single partition, so there is no train/val/test
leakage within a split. See the [paper](https://openreview.net/forum?id=PyZvVIJbSy)
for the clustering methodology and split rationale.


## Configuration

All scripts read the data path from the `CHIMERA_DATA_ROOT` environment variable:

```bash
export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0
```

Alternatively, pass `--data-root` on the CLI:

```bash
python -m pipeline --data-root /path/to/chimera-bench-v1.0 --steps annotate features splits
```

## Citation

```bibtex
@inproceedings{
ahmed2026chimerabench,
title={{CHIMERA}-Bench: A Benchmark Dataset for Epitope-Specific Antibody Design},
author={Mansoor Ahmed and Nadeem Taj and Imdad Ullah Khan and Hemanth Venkateswara and Murray Patterson},
booktitle={ICLR 2026 Workshop on Generative and Experimental Perspectives for Biomolecular Design},
year={2026},
url={https://openreview.net/forum?id=PyZvVIJbSy}
}
```

## License

- **Code**: MIT License (see [LICENSE](LICENSE))
- **Data**: CC-BY 4.0 (see [LICENSE-DATA](LICENSE-DATA))