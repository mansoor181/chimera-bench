# AbODE: Ab initio Antibody Design using Conjoined ODEs
The repository for AbODE: Ab initio Antibody Design using Conjoined ODEs ICML 2023 paper. For more information visit the [website](https://yogeshverma1998.github.io/AbODE/).

Evaluation and pre-trained models.

## Prerequisites

- torchdiffeq : https://github.com/rtqichen/torchdiffeq.
- pytorch >= 1.10.0
- torch-scatter 
- torch-sparse 
- torch-cluster 
- torch-spline-conv 
- torch-geometric == 2.0.4
- astropy
- networkx
- tqdm
- Biopython

Make sure to have the correct version of each library as some errors might arise due to version-mismatch. The libraries-version of the local conda env is in `env_list.txt` 
## Datasets
Datasets are placed in the "data/" folder. Note that [Structural Antibody Dataset](https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabdab) is an evolutionary dataset in which new structures are added each week. The model has been evaluated on the same specification as other competing methods in benchmarks.

## Evaluation:

Unconditional Antibody Sequence and structure generation:
```
python evaluation_uncond.py --cdr ${CDR-region-antibody}(1/2/3)
```

Conditional Antibody Sequence and structure generation:
```
python evaluation_cond.py --cdr ${CDR-region-antibody}(1/2/3)
```

## CHIMERA-Bench Integration

CHIMERA integration files for training and evaluating AbODE on the CHIMERA-Bench dataset. Uses the **conditional variant** (`Adobe_cond`) which takes antigen context as input for epitope-conditioned CDR generation.

### Key Design Decisions

- **Conditional model** (`Adobe_cond`): Uses antigen sequence + coords for epitope-conditioned generation
- **Single CDR per run**: Trains/evaluates one CDR type at a time (H1, H2, H3), matching the original evaluation protocol
- **Chothia numbering**: CDR boundaries from CHIMERA's `cdr_masks.chothia.heavy`
- **Training written from scratch**: The original repo only provides evaluation scripts; `chimera_trainer.py` implements training using the native loss function (`loss_function_vm_with_side_chains_angle`) and ODE solver
- **Batch size = 1**: Variable-size fully-connected graphs with ODE solver require per-sample processing
- **Coordinate reconstruction**: Quaternion polar representation -> Cartesian via `_get_cartesian()` + cumulative sum from first residue

### CHIMERA Files

| File | Purpose |
|------|---------|
| `config.yaml` | Hyperparameters (model, training, ODE solver, callbacks) |
| `preprocess.py` | Convert `complex_features/*.pt` to AbODE conditional JSON |
| `chimera_trainer.py` | Train `Adobe_cond` per CDR type per split |
| `chimera_evaluate.py` | Run full CHIMERA metric suite (14 metrics + bootstrap CIs) |
| `chimera_train.sh` | Train all 9 combinations (3 CDRs x 3 splits) |

### Audit Fixes [05-28-2026]

1. **Missing `numbering_scheme: imgt`** in `config.yaml`. AbODE uses IMGT-numbered data from
   CHIMERA complex_features, but the config didn't declare this. Evaluation metrics could use
   wrong CDR boundaries without it.

2. **Missing `--test_only` and `--checkpoint` flags** in `chimera_trainer.py`. All other baselines
   support these for re-running evaluation without retraining. Added to `parse_args()`,
   `build_config()`, and `main()`.

3. **CRITICAL: `_get_cartesian()` return value swap.** The function returns `(Cart_truth, Cart_pred)`
   but `reconstruct_ca_coords()` unpacked as `Cart_pred, Cart_truth = ...`, swapping predicted and
   ground truth coordinates.

4. **CRITICAL: Native coord alignment never executed.** `load_native_cdr_coords()` returned
   CDR-only CA coords (N residues) but AbODE predicts CDR + 1 flanking residue on each side
   (N+2 residues). The length check `len(native) == len(pred)` always failed, falling back to
   AbODE's broken local-frame reconstruction (~14A from PDB coords). Fixed by including flanking
   residues in native coord extraction.

5. **`--test_only` checkpoint key mismatch.** Used `state["model"]` but `ModelCheckpoint` saves
   as `state["model_state_dict"]`. Fixed.

These 3 coordinate bugs explain the previously reported RMSD=14.64A and DockQ=0.37.

### Usage

```bash
# 1. Preprocess (once)
cd baselines/abode
python preprocess.py

# 2. Train a single CDR/split
python chimera_trainer.py --split epitope_group --cdr_type 3 --wandb

# 3. Train all combinations
GPU=0 bash chimera_train.sh

# 4. Evaluate
python chimera_evaluate.py --aggregate --split epitope_group
```

### Data Format

AbODE conditional JSON stores per-complex entries with CDR + flanking residues separated from framework:

```json
{
  "complex_id": "1a2y_B_A_C",
  "ab_seq": "GFTFSTYY",
  "before_seq": "QQQLVESGGR...",
  "after_seq": "MSWVRQAPGK...",
  "ag_seq": "MPHEHSFNEG...",
  "coords_ab": [[[N],[CA],[C]], ...],
  "coords_before": [[[N],[CA],[C]], ...],
  "coords_after": [[[N],[CA],[C]], ...],
  "coords_ag": [[[N],[CA],[C]], ...]
}
```

Preprocessed data is stored in `trans_baselines/abode/{split}/cdrh{1,2,3}/{train,val,test}.json`.
