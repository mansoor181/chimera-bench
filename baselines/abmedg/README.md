
AbMEGD
![overview](./assets/overview.jpg)
Antibody Design and Optimization with Multi-scale Equivariant Graph Diffusion Models for Accurate Complex Antigen Binding

## Install

### Environment

```bash
conda env create -f env.yaml -n AbMEGD
conda activate AbMEGD
```

The default `cudatoolkit` version is 11.3. You may change it in [`env.yaml`](./env.yaml).

### Datasets and Trained Weight

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). Extract `all_structures.zip` into the `data` folder.

The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/).

## Design and Optimize Antibodies

3 design modes are available. Each mode corresponds to a config file in the `configs/test` folder:
| Config File | Description |
| ------------------------ | ------------------------------------------------------------ |
| `codesign_single.yml` | Sample both the **sequence** and **structure** of **one** CDR. |
| `codesign_multicdrs.yml` | Sample both the **sequence** and **structure** of **all** the CDRs simultaneously. |
| `abopt_singlecdr.yml` | Optimize the **sequence** and **structure** of **one** CDR. |

## Train

```bash
python train.py ./configs/train/<config-file-name>
```


## CHIMERA-Bench Integration

Five files added for CHIMERA-Bench retraining and evaluation:

| File | Description |
|------|-------------|
| `config.yaml` | Based on `codesign_multicdrs2024.yml`: model type `AbMEGD`, 400K iters, batch_size 8, seed 2024. |
| `preprocess.py` | Chothia renumbering -> LMDB via `AbMEGD.datasets.sabdab.preprocess_sabdab_structure()`. Category A format. |
| `chimera_trainer.py` | Iteration-based trainer. `model(batch)` returns `loss_dict` directly (no TB loss args). Multi-CDR mode with `MaskMultipleCDRs`. |
| `chimera_evaluate.py` | `--predictions` (single CDR) or `--aggregate` (all CDRs per split). 12-metric CSV output. |
| `chimera_train.sh` | Trains once per split, runs `eval_and_append` after each with `--aggregate`. |

**Key differences from DiffAb/AbFlowNet:**
- Imports from `AbMEGD.*` package
- Model type `AbMEGD`
- No TB loss -- `model.forward(batch)` takes only batch (simpler than AbFlowNet)
- 400K iters (vs 200K), batch_size 8 (vs 16), seed 2024

**Usage:**
```bash
cd baselines/abmedg
python preprocess.py                                    # one-time preprocessing
GPU=0 bash chimera_train.sh                             # train all splits
GPU=0 SPLITS="epitope_group" bash chimera_train.sh      # single split
```