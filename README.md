---
task_categories:
- other
license: cc-by-4.0
tags:
- biology
- antibody-design
- protein-structure
- epitope-specific
---

# CHIMERA-Bench v1.0

A unified benchmark for epitope-specific antibody CDR sequence-structure co-design.

**Paper**: [CHIMERA-Bench: A Benchmark Dataset for Epitope-Specific Antibody Design](https://huggingface.co/papers/2603.13431) (ICLR GEM Workshop 2026)  
**Code**: [GitHub - mansoor181/chimera-bench](https://github.com/mansoor181/chimera-bench)


## Dataset Summary

CHIMERA-Bench (CDR Modeling with Epitope-guided Redesign) is a curated, deduplicated dataset of **2,922** antibody-antigen complexes with epitope and paratope annotations. It provides a standardized evaluation protocol for antibody design tasks.

| Property | Value |
|----------|-------|
| Complexes | 2,922 |
| PDB structures | 2,721 |
| Pre-computed features | 2,941 .pt files |
| Splits | 3 (epitope-group, antigen-fold, temporal) |
| Numbering schemes | IMGT, Chothia |
| Evaluation contact cutoff | 8.0 A (CA-CA) |
| Resolution cutoff | 4.0 A |
| Baselines evaluated | 11 methods, 8 paradigms |

## Directory Structure

```
chimera-bench-v1.0/
  README.md
  metadata/
    final_summary.csv           # 2,922 complexes with 32 columns
    excluded_complexes.csv      # 59 excluded complexes with reasons
    antibody_sequences.fasta    # VH+VL sequences for all complexes
    contamination_audit.json    # PLM training data overlap analysis
  splits/
    epitope_group.json          # Primary split (2338/292/292)
    antigen_fold.json           # Fold-based generalization (2338/292/292)
    temporal.json               # Prospective evaluation (2337/292/293)
  complex_features/             # Per-complex PyTorch tensors (2,941 files)
    {complex_id}.pt
  structures/                   # PDB structure files (2,721 files)
    {pdb}.pdb
```

## Complex Features Format

Each `.pt` file is a Python dict containing:

**Sequences**
- `complex_id`: str -- unique identifier ({pdb}_{Hchain}_{Lchain}_{Agchain})
- `heavy_sequence`, `light_sequence`, `antigen_sequence`: str -- one-letter AA

**Coordinates**
- `heavy_atom14_coords`: float32 (N_h, 14, 3) -- 14-atom representation
- `heavy_atom14_mask`: bool (N_h, 14) -- valid atom flags
- `heavy_ca_coords`: float32 (N_h, 3) -- CA-only coordinates
- Same for `light_*` and `antigen_*`

**Annotations**
- `epitope_residues`: list of (chain, resid, resname) tuples
- `paratope_residues`: list of (chain, resid, resname) tuples
- `contact_pairs`: list of (ab_chain, ab_resid, ab_resname, ag_chain, ag_resid, ag_resname, distance)

**Numbering**
- `numbering`: dict with `imgt` and `chothia` sub-dicts
- `cdr_masks`: dict with `imgt` and `chothia` sub-dicts (-1=framework; heavy: 0=H1, 1=H2, 2=H3; light: 3=L1, 4=L2, 5=L3)

**Surface Features**
- `ag_surface_points`, `ag_surface_normals`, `ag_surface_curvatures`, `ag_surface_chemical_feats`.

## Splits

- **epitope_group**: clusters by epitope residue fingerprint; test set has epitope patterns unseen during training.
- **antigen_fold**: clusters by antigen identity; test set has entirely unseen antigens.
- **temporal**: splits by PDB deposition date; simulates prospective deployment.

## Sample Usage

```python
import torch, json, pandas as pd

# Load metadata
summary = pd.read_csv("metadata/final_summary.csv")

# Load a split
with open("splits/epitope_group.json") as f:
    split = json.load(f)
print(f"Train: {len(split['train'])}, Val: {len(split['val'])}, Test: {len(split['test'])}")

# Load a complex
feat = torch.load(f"complex_features/{split['test'][0]}.pt", weights_only=False)
print(feat['complex_id'], feat['heavy_sequence'][:20], "...")
print(f"Epitope residues: {len(feat['epitope_residues'])}")
print(f"CDR-H3 (IMGT): positions where cdr_masks['imgt']['heavy'] == 2")
```

## Evaluation Metrics

| Group | Metrics |
|-------|---------|
| Sequence quality | AAR, CAAR, PPL |
| Structural accuracy | RMSD (Kabsch-aligned CA), TM-score |
| Binding interface | Fnat, iRMSD, DockQ |
| Epitope specificity | EpiF1 (precision, recall, F1) |
| Designability | n_liabilities (NG, DG, DS, DD, NS, NT, M motifs) |

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

- **Data**: CC-BY 4.0
- **Code**: MIT