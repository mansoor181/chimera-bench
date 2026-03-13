# MEAN: Conditional Antibody Design as 3D Equivariant Graph Translation

This repo contains the codes for our paper [Conditional Antibody Design as 3D Equivariant Graph Translation](https://arxiv.org/abs/2208.06073). MEAN is the abbreviation for the **M**ulti-channel **E**quivariant **A**ttention **N**etwork proposed in our paper.

## Quick Links

- [Setup](#setup)
    - [Dependencies](#dependencies)
    - [Get Data](#get-data)
- [Experiments](#experiments)
    - [K-fold Evaluation on SAbDab](#k-fold-evaluation-on-sabdab)
    - [Antigen-binding CDR-H3 Redesign](#antigen-binding-cdr-h3-redesign)
    - [Affinity Optimization](#affinity-optimization)
- [Inference API](#inference-api)
- [Contact](#contact)
- [Others](#others)

## Setup

### Dependencies

we have prepared the script for environment setup in scripts/setup.sh, please install the dependencies in it with `bash scripts/setup.sh` before running our code.

### Get Data
We have provided the summary data used in our paper from SAbDab, RAbD, SKEMPI_V2 in the summaries folder, please download all structure data from the [download page of SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true#downloads).
Since the SAbDab is updating on a weekly basis, you may also download the newest summary file from its [official website](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/about/). 
The following instructions will suppose the structure data renumbered by imgt is located at the folder *all_structures/imgt*.

## Experiments
We have provided the shell scripts for different procedures of the experiments, which are located either in the folder *scripts* or in the root folder of our repo. For scripts in *scripts*, you can run them without arguments to see their hints of usage, which will also be illustrated in README later. One thing that might need explanation is the mode parameter which takes the value of either 100 or 111. **100** means only heavy chain is used in the context, and **111** means heavy chain, light chain as well as the epitope are considered in the context. The later one is the complete setting of antigen-binding CDR design, whereas the first one is only for comparison with RefineGNN. \<model type\> in the arguments only specifies the prefix of the directory to save/load the checkpoints, we use **mean** in the following sections. **Please try using absolute directories when passing arguments to the scripts if you encounter problems concerning paths.**

### K-fold evaluation on SAbDab
We have provided the scripts for data preparation, k-fold training and evaluation
- data preparation: bash scripts/prepare_data_kfold.sh \<summary file\> \<pdb folder\>
- training: GPU=\<gpu id\> bash scripts/k_fold_train.sh \<summary folder\> \<mode\> \<model type\> \<port for multi-GPU training\>
- evaluation: GPU=\<gpu id\> bash scripts/k_fold_eval.sh \<mode\> \<model type\> \<version id\>

here is an example for evaluating our MEAN:
```bash
bash scripts/prepare_data_kfold.sh summaries/sabdab_summary.tsv all_structures/imgt
GPU=0 bash scripts/k_fold_train.sh summaries 111 mean 9901
GPU=0 bash scripts/k_fold_eval.sh summaries 111 mean 0
```

By running `bash scripts/prepare_data_kfold.sh summaries/sabdab_summary.tsv all_structures/imgt`, the script will copy the pdbs in the summary to *summaries/pdb*, transform the summary to json format, and generate 10-fold data splits for each cdr, which requires ~5G space. If you want to do data preparation in another directory, just copy the summary file there and replace the *summaries/sabdab_summary.tsv* with the new path.
Also, for each parallel run of trainining, the checkpoints will be saved in version 0, 1, ... So you need to specify the *version id* as the last argument of *k_fold_eval.sh*.


### Antigen-binding CDR-H3 Redesign
before running this task, please at least run the commands of downloading json summary of SAbDab in scripts/prepare_data_kfold.sh (line 23-31). We will suppose the json file is located at summaries/sabdab_all.json.
- data preparation: bash scripts/prepare_data_rabd.sh \<rabd summary file\> \<pdb folder\> \<sabdab summary file in json format\>
- training: GPU=\<gpu id\> MODE=\<mode\> DATA_DIR=\<data directory with train, valid and test json summary\> bash train.sh \<model type\> \<cdr type\>
- evaluation: GPU=\<gpu id\> MODE=\<mode\> DATA_DIR=\<data directory with train, valid and test json summary\> bash rabd_test.sh \<version id\> \[checkpoint path\]

Example:
```bash
bash scripts/prepare_data_rabd.sh summaries/rabd_summary.jsonl all_structures/imgt summaries/sabdab_all.json
GPU=0 MODE=111 DATA_DIR=summaries/cdrh3 bash train.sh mean 3
GPU=0 MODE=111 DATA_DIR=summaries/cdrh3 bash rabd_test.sh 0
```

We have also provided the trained checkpoint used in our paper at checkpoints/ckpt/rabd_cdrh3_mean.ckpt. You can use it for test by running `GPU=0 MODE=111 DATA_DIR=summaries/cdrh3 bash rabd_test.sh 0 checkpoints/ckpt/rabd_cdrh3_mean.ckpt`. The results will be saved to the folder named **results** under the same directory as the checkpoint.

### Affinity Optimization
before running this task, please at least run the commands of downloading json summary of SAbDab in scripts/prepare_data_kfold.sh (line 23-31). We will suppose the json file is located at summaries/sabdab_all.json.
- data preparation: bash scripts/prepare_data_skempi.sh \<skempi summary file\> \<pdb folder\> \<sabdab summary file in json format\>
- training: train.sh, ita_train.sh
- pretraining: GPU=\<gpu id\> MODE=\<mode\> DATA_DIR=\<data directory with train, valid and test json summary\> bash train.sh \<model type\> \<cdr type\>
- ITA training: GPU=\<gpu id\> CKPT_DIR=\<pretrained checkpoint folder\> bash ita_train.sh
- evaluation: GPU=\<gpu id\> DATA_DIR=\<dataset folder \> bash ita_generate.sh \<checkpoint>

Example:
```bash
bash scripts/prepare_data_skempi.sh summaries/skempi_v2_summary.jsonl all_structures/imgt summaries/sabdab_all.json
GPU=0 MODE=111 DATA_DIR=summaries bash train.sh mean 3
GPU=0 CKPT_DIR=summaries/ckpt/mean_CDR3_111/version_0 bash ita_train.sh
GPU=0 DATA_DIR=summaries bash ita_generate.sh summaries/ckpt/mean_CDR3_111/version_0/ita/iter_i.ckpt  # specify the checkpoint from iteration i for testing
```

We have also provided the checkpoint after ITA finetuning at checkpoints/ckpt/opt_cdrh3_mean.ckpt. You can directly use it for inference by running `GPU=0 DATA_DIR=summaries bash ita_generate.sh checkpoints/ckpt/opt_cdrh3_mean.ckpt`. This script will generate 100 optimized candidates for each antibody in summaries/skempi_all.json and report the top1 in terms of predicted ddg. The pdbs of optimized candidates will be located at the same directory of the checkpoint.


## Inference API
We also provide the script for design / optimization of single CDR at scritps/design.py. The script requires an input pdb containing the heavy chain, the light chain and the antigen. The pdb should be renumbered using the IMGT system in advance, which can be achieved by the script at data/ImmunoPDB.py fron ANARCI. Here is an example of CDR-H3 design for the 1ic7 pdb:

```bash
python ./data/ImmunoPDB.py -i data/1ic7.pdb -o 1ic7.pdb -s imgt  # renumber the pdb
python ./scripts/design.py --pdb 1ic7.pdb --heavy_chain H --light_chain L
```

The generated pdb as well as a summary of the CDR-H3 sequence will be saved to ./results.
The default checkpoint used in the script is checkpoints/ckpt/rabd_cdrh3_mean.ckpt. You can pass your own checkpoint by the argument `--ckpt path/to/your/checkpoint` (e.g. use the opt_cdrh3_mean.ckpt for CDR optimization)

Further, the script is able to accommodate multiple pdbs as inputs, for example:
```bash
python ./scripts/design.py
    --pdb pdb1 pdb2 ...
    --heavy_chain Hchain1 Hchain2 ...
    --light_chain Lchain1 Lchain2 ...
```

## CHIMERA-Bench Integration

The following files were added for CHIMERA-Bench evaluation. Original MEAN source code is unmodified.

### Added Files

| File | Purpose |
|------|---------|
| `config.yaml` | MEAN hyperparameters for CHIMERA training (model, training, callbacks, wandb) |
| `preprocess.py` | Converts CHIMERA dataset to MEAN's native format (PDB -> AAComplex -> pkl) |
| `chimera_trainer.py` | Training/validation/test with CHIMERA metrics (AAR, RMSD, TM-score, PPL, liabilities) |
| `chimera_evaluate.py` | Full CHIMERA evaluation (14 metrics) with bootstrap CIs, CSV output |
| `chimera_train.sh` | Automates training across CDR types and splits, then evaluates |

### Key Design Decisions

- **One CDR per run**: MEAN generates one CDR at a time (H1, H2, or H3). Run the trainer 3 times with `--cdr_type 1/2/3`.
- **Index-based complex_id mapping**: MEAN's `AAComplex.get_id()` returns PDB IDs (e.g., `3bsz`), but CHIMERA uses complex_ids (e.g., `3bsz_N_M_E`). Multiple complexes can share a PDB ID, so `preprocess.py` generates `idx_to_cid.json` (ordered list mapping dataset index to complex_id) and `complex_ids.json` (inverse lookup).
- **Paths from shared config**: All paths (data_root, results, trans_baselines) come from `baselines/shared_config.yaml`. No hardcoded paths in scripts.
- **Robust preprocessing**: PDBs that fail parsing (missing CDRs, non-standard residues) are skipped with a log message rather than crashing.

### Usage

```bash
cd baselines/mean

# 1. Preprocess (once)
python preprocess.py

# 2. Train (one CDR at a time)
python chimera_trainer.py                          # CDR-H3, epitope_group split (default)
python chimera_trainer.py --cdr_type 1             # CDR-H1
python chimera_trainer.py --cdr_type 2             # CDR-H2
python chimera_trainer.py --split temporal --wandb  # different split, with wandb

# 3. Evaluate
python chimera_evaluate.py --predictions /path/to/predictions/  # single CDR
python chimera_evaluate.py --aggregate --split epitope_group    # all CDRs
```

### Metrics Computed

**During training** (val/test, Tier 1): PPL, AAR, RMSD, TM-score, liability count.

**`chimera_evaluate.py`** (Tier 2, post-training): Full 14-metric suite -- AAR, CAAR, PPL, RMSD, TM-score, Fnat, iRMSD, DockQ, epitope precision/recall/F1, liability count, CHIMERA-S, CHIMERA-B. Uses saved `.pt` predictions + native `complex_features/*.pt` data.

### Changelog

- 2026-02-05: Added chimera_train.sh; chimera_evaluate.py now computes full 14-metric suite
- 2026-02-04: Initial CHIMERA integration (preprocess, trainer, evaluator, config)

---

## Contact

Thank you for your interest in our work!

Please feel free to ask about any questions about the algorithms, codes, as well as problems encountered in running them so that we can make it clearer and better. You can either create an issue in the github repo or contact us at jackie_kxz@outlook.com.

## Others

Some codes are borrowed from existing repos:

- evaluation/ddg: https://github.com/HeliXonProtein/binding-ddg-predictor
- evaluation/TMscore.cpp: https://zhanggroup.org/TM-score/
- data/anarci: https://github.com/oxpig/ANARCI
- models/MCAttGNN/mc_egnn: https://github.com/vgsatorras/egnn
