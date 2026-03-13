# CHIMERA-Bench

A benchmark dataset for **epitope-specific antibody design**.

**Paper**: [CHIMERA-Bench: A Gold-Standard Benchmark for Epitope-Specific Antibody Design](https://openreview.net/forum?id=PyZvVIJbSy/) (ICLR 2026 GEM Workshop)

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

## Dataset

The dataset is hosted on HuggingFace Hub and Zenodo:

- **HuggingFace**: [`chimera-bench/chimera-bench-v1.0`](https://huggingface.co/datasets/mansoorbaloch/chimera-bench)
- **Zenodo**: [DOI: TBD](https://zenodo.org/)
- **Pre-computed residue graphs** (optional, 5.8 GB): available as a separate download

### Download

```bash
# Set the data root (used by all scripts)
export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0

# Option 1: HuggingFace CLI
hf download chimera-bench/chimera-bench-v1.0 --local-dir $CHIMERA_DATA_ROOT

# Option 2: Direct download from Zenodo
# wget <zenodo-url> -O chimera-bench-v1.0.zip && unzip chimera-bench-v1.0.zip
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

See [`demo.ipynb`](demo.ipynb) for a complete walkthrough.

## Installation

The setup requires both conda (for bioinformatics tools) and pip (for PyTorch and ML packages), installed in a specific order to avoid conflicts.

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

```python
import torch, json

# Load a split
with open(f"{data_root}/splits/epitope_group.json") as f:
    split = json.load(f)
print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")

# Load a complex
feat = torch.load(f"{data_root}/complex_features/{split['test'][0]}.pt", weights_only=False)
print(feat['complex_id'], feat['heavy_sequence'][:20], "...")
print(f"Epitope: {len(feat['epitope_residues'])} residues")
print(f"CDR-H3 (IMGT): positions where cdr_masks['imgt']['heavy'] == 2")
```

## Baselines

11 methods retrained on CHIMERA-Bench across 6 paradigms:

| Method | Paradigm | Epi-Cond? | Multi-CDR? | Directory |
|--------|----------|:---------:|:----------:|-----------|
| DiffAb | Diffusion | Yes | Yes | `baselines/diffab/` |
| AbFlowNet | Flow matching | Yes | Yes | `baselines/abflownet/` |
| AbMEGD | Diffusion | Yes | Yes | `baselines/abmedg/` |
| RADAb | Retrieval + diffusion | Yes | Yes | `baselines/radab/` |
| dyAb | Flow matching | Yes | Yes | `baselines/dyab/` |
| MEAN | Equivariant GNN | Yes | No (H3) | `baselines/mean/` |
| dyMEAN | Equivariant GNN | Yes | Yes | `baselines/dymean/` |
| RAAD | Equivariant GNN | Yes | Yes | `baselines/raad/` |
| RefineGNN | Autoregressive GNN | No | Yes | `baselines/refinegnn/` |
| AbODE | Conjoined ODE | No | Yes | `baselines/abode/` |
| AbDockGen | Hierarchical ENN | Yes (H3) | No | `baselines/abdockgen/` |

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

## Repository Structure

```
chimera-bench/
  README.md                     # This file
  LICENSE                       # MIT (code)
  LICENSE-DATA                  # CC-BY 4.0 (data)
  DATASHEET.md                  # Datasheet for Datasets (Gebru et al.)
  environment.yml               # Conda environment (base packages)
  setup_env.sh                  # Full setup script (conda + pip)
  requirements.txt              # Pip-only dependencies (for reference)
  config.py                     # Central configuration
  pipeline.py                   # Dataset construction pipeline
  demo.ipynb                    # Dataset exploration notebook
  analysis.ipynb                # Visualization and analysis
  data/                         # Dataset construction modules
    collect.py                  # SAbDab download
    filter.py                   # Quality filtering
    dedup.py                    # MMseqs2 deduplication
    annotate.py                 # Numbering, CDR masks, contacts
    features.py                 # Coordinate extraction, surface features
    splits.py                   # Split generation
    validate.py                 # Data validation
    graphs.py                   # Residue graph construction
  converters/                   # Format converters (5 categories)
    sabdab_style.py             # Category A: DiffAb-family (11 methods)
    refinegnn_jsonl.py          # Category B: RefineGNN
    iggm_pdb_fasta.py           # Category C: IgGM, ProteinMPNN
    absgm_6d.py                 # Category D: AbSGM
    cdr_only.py                 # Category E: AbODE, AbDockGen
  evaluation/                   # Evaluation framework
    metrics.py                  # All 12 metrics
    evaluate.py                 # Unified evaluation entry point
    contamination.py            # PLM training data overlap audit
  baselines/                    # 11 retrained baseline methods
    chimera_utils.py            # Shared utilities
    shared_config.yaml          # Shared paths and settings
    diffab/                     # Each baseline has its own directory
    ...
```

## Splits

| Split | Train | Val | Test | Generalization Axis |
|-------|------:|----:|-----:|---------------------|
| epitope_group | 2,338 | 292 | 292 | Unseen epitope patterns |
| antigen_fold | 2,338 | 292 | 292 | Unseen antigen folds |
| temporal | 2,337 | 292 | 293 | Prospective (by deposition date) |

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
