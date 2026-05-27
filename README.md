# CHIMERA-Bench

A benchmark dataset for **epitope-specific antibody design**.

**Paper**: [CHIMERA-Bench: A Benchmark Dataset for Epitope-Specific Antibody Design](https://openreview.net/forum?id=PyZvVIJbSy) (ICLR 2026 GEM Workshop)

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


### Split 1: Epitope-Group (primary -- generalize to unseen epitopes)

- Cluster epitopes by structural similarity (TM-align on epitope patches)
- Test set contains epitope clusters never seen during training
- **Tests**: Can the method generalize to novel binding sites?
- This is the gold-standard split for epitope-specific design

### Split 2: Antigen-Fold (structural generalization)

- Cluster antigens by structural similarity using Foldseek (TM-score ≥ 0.5)
- Test set contains antigen folds absent from training
- **Tests**: Can the method handle novel antigen topologies?

### Split 3: Temporal (realistic deployment scenario)

- Train: structures deposited before 2022-01-01
- Val: 2022-01-01 to 2023-06-01
- Test: after 2023-06-01
- **Tests**: Does the method work on genuinely new targets?

### Split Statistics

**Dataset Size**: 2922 antibody-antigen complexes (after filtering and deduplication)

| Split | Train | Val | Test | Ratio |
|-------|-------|-----|------|-------|
| epitope_group | 2338 | 292 | 292 | 80/10/10 |
| antigen_fold | 2338 | 292 | 292 | 80/10/10 |
| temporal | 2337 | 292 | 293 | 80/10/10 |

**Integrity Guarantees**:
- No train/val/test overlap within any split (verified)
- Clusters are never split across train and test (whole-cluster assignment)
- Test sets differ across split types (~10-13% overlap between any two test sets)

**Epitope-Group Clustering**:
- Method: Kabsch RMSD on epitope CA coordinates with hierarchical clustering (average linkage)
- Threshold: 3.0 A RMSD cutoff for cluster membership
- Epitopes with different sizes use geometric hash (center of mass + spread) for distance
- Complexes without valid epitope coordinates (< 3 residues) assigned to singleton clusters

**Antigen-Fold Clustering (Foldseek)**:
- Method: Foldseek easy-cluster with TM-score ≥ 0.5 threshold (same fold)
- Coverage: 99.9% (2918/2922 complexes mapped to 810 structural fold clusters)
- Parameters: bidirectional coverage ≥ 50%, greedy set cover clustering
- Replaces CATH superfamily grouping which only covered ~30% of chains

**Temporal Split Cutoffs**:
- Train: 1990-08-27 to 2024-05-08
- Val: 2024-05-08 to 2025-05-14
- Test: 2025-05-14 to 2026-01-28 (most recent structures)
- Complexes without parseable dates assigned to train

### Detailed Per-Split Statistics

#### Epitope-Group Split

| Subset | n | CDR-H3 Length | Epitope Size | Antigen Size |
|--------|---|---------------|--------------|--------------|
| Train | 2338 | 14.4 ± 4.2 [3-63] | 16.4 ± 6.2 [0-64] | 265.5 ± 273.1 [3-2363] |
| Val | 292 | 15.4 ± 4.3 [6-26] | 24.0 ± 6.6 [0-46] | 381.1 ± 278.4 [1-1853] |
| Test | 292 | 15.5 ± 4.2 [4-28] | 24.0 ± 6.6 [1-62] | 379.2 ± 282.8 [1-1349] |

CDR-H3 length distribution (Train/Val/Test):
- Short (<10): 9.2% / 7.5% / 5.1%
- Medium (10-15): 57.4% / 45.5% / 47.9%
- Long (16-20): 24.1% / 33.6% / 33.9%
- Very Long (>20): 9.2% / 13.4% / 13.0%

Epitope size distribution (Train/Val/Test):
- Small (<15): 38.8% / 5.1% / 5.1%
- Medium (15-25): 54.8% / 59.6% / 53.4%
- Large (>25): 6.4% / 35.3% / 41.4%

Antigen size distribution (Train/Val/Test):
- Small (<200): 57.2% / 34.2% / 33.6%
- Medium (200-500): 30.9% / 46.9% / 48.3%
- Large (>500): 11.8% / 18.8% / 18.2%

#### Antigen-Fold Split

| Subset | n | CDR-H3 Length | Epitope Size | Antigen Size |
|--------|---|---------------|--------------|--------------|
| Train | 2338 | 14.5 ± 4.2 [3-29] | 18.0 ± 6.9 [0-64] | 298.5 ± 284.1 [1-1853] |
| Val | 292 | 14.9 ± 5.1 [5-63] | 17.4 ± 7.1 [1-46] | 262.6 ± 270.1 [1-2363] |
| Test | 292 | 14.6 ± 4.1 [5-28] | 17.7 ± 7.1 [1-52] | 233.6 ± 228.2 [2-1265] |

CDR-H3 length distribution (Train/Val/Test):
- Short (<10): 8.6% / 9.2% / 8.6%
- Medium (10-15): 56.1% / 51.4% / 52.7%
- Long (16-20): 25.2% / 29.8% / 29.1%
- Very Long (>20): 10.1% / 9.6% / 9.6%

Epitope size distribution (Train/Val/Test):
- Small (<15): 31.3% / 33.2% / 37.0%
- Medium (15-25): 55.8% / 55.5% / 49.7%
- Large (>25): 12.9% / 11.3% / 13.4%

Antigen size distribution (Train/Val/Test):
- Small (<200): 50.7% / 57.2% / 62.7%
- Medium (200-500): 35.1% / 32.9% / 29.1%
- Large (>500): 14.2% / 9.9% / 8.2%

#### Temporal Split

| Subset | n | CDR-H3 Length | Epitope Size | Antigen Size | Date Range |
|--------|---|---------------|--------------|--------------|------------|
| Train | 2337 | 14.5 ± 4.3 [3-63] | 17.7 ± 7.0 [0-64] | 283.2 ± 281.8 [1-1853] | 1990-08-27 to 2024-05-08 |
| Val | 292 | 14.9 ± 4.1 [5-33] | 18.9 ± 6.9 [5-52] | 320.0 ± 276.7 [5-1349] | 2024-05-08 to 2025-05-14 |
| Test | 293 | 15.1 ± 3.9 [6-26] | 19.1 ± 6.2 [1-34] | 299.0 ± 249.3 [2-2363] | 2025-05-14 to 2026-01-28 |

CDR-H3 length distribution (Train/Val/Test):
- Short (<10): 9.2% / 7.2% / 5.8%
- Medium (10-15): 55.3% / 55.1% / 55.6%
- Long (16-20): 25.8% / 25.7% / 28.3%
- Very Long (>20): 9.7% / 12.0% / 10.2%

Epitope size distribution (Train/Val/Test):
- Small (<15): 33.8% / 27.1% / 23.5%
- Medium (15-25): 54.2% / 58.6% / 59.0%
- Large (>25): 12.0% / 14.4% / 17.4%

Antigen size distribution (Train/Val/Test):
- Small (<200): 54.3% / 45.5% / 46.1%
- Medium (200-500): 32.6% / 39.0% / 42.3%
- Large (>500): 13.1% / 15.4% / 11.6%

### Stratified Analyses (within each split)

These are not separate splits but diagnostic breakdowns reported for each split:

- **By CDR-H3 length**: Short (<10), Medium (10-15), Long (16-20), Very Long (>20)
- **By epitope size**: Small (<15 residues), Medium (15-25), Large (>25)
- **By antigen size**: Small (<200 residues), Medium (200-500), Large (>500)
- **Per-CDR**: H1, H2, H3, L1, L2, L3 reported separately


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