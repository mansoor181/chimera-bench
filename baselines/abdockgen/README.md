# Antibody-Antigen Docking and Design via Hierarchical Equivariant Refinement

This is the implementation of our ICML 2022 paper: https://arxiv.org/pdf/2207.06616.pdf

## Dependencies
Our model is tested in Linux with the following packages:
* CUDA >= 11.1
* PyTorch == 1.8.2 (LTS Version)
* Numpy >= 1.18.1
* tqdm

You need additional packages for `process_data.py`, which convert PDB into JSON format. This is required only if you want to process your own data.
* sidechainnet
* prody

To calculate DockQ score (docking evaluation), you need to `git clone https://github.com/bjornwallner/DockQ`

## Data
Our data are downloaded from the Structural Antibody Database (SAbDab). All the PDBs must be renumbered under IMGT scheme. 
The processed JSON files are saved in `data/` folder, along with train/val/test splits.
If you want to process your own PDB file, please run
```
python process_data.py ${pdb_file} ${heavy_chain_id} ${antigen_chain_id}
```
It will process the PDB file into JSON format used by this codebase.

## CDR-H3 Local Docking
The training script can be launched by
```
python dock_train.py --hierarchical --L_target 20 --save_dir ckpts/HERN-dock
```
where `L_target` is the size of the epitope and `--hierarchical` means using hierarchical encoder.

At test time, we can dock CDR-H3 paratopes onto their corresponding epitopes:
```
mkdir outputs
python predict.py ckpts/HERN_dock.ckpt data/rabd/test_data.jsonl
```
It will produce a PDB file for each epitope in the test set with docked CDR-H3 structure. You can evaluate those docked structures using DockQ.py

## Epitope-specific CDR-H3 Design
The training script can be launched by
```
python lm_train.py --hierarchical --L_target 20 --save_dir ckpts/HERN-gen
```

At test time, we can generate new CDR-H3 paratopes specific to a given epitope:
```
python generate.py ckpts/HERN_gen.ckpt data/rabd/test_data.jsonl 1 > results/HERN.txt
```
The above script will generate one CDR-H3 sequence per epitope. You can sample more candidates by changing this parameter.

## CHIMERA-Bench Integration

AbDockGen is integrated with the CHIMERA-Bench evaluation pipeline for standardized benchmarking across 18 antibody design methods. AbDockGen is **H3-only**: it generates CDR-H3 sequence + structure conditioned on the epitope surface.

### Preprocessing

Convert CHIMERA complex features to AbDockGen JSONL format (no renumbering needed -- both use IMGT):

```bash
cd baselines/abdockgen
python preprocess.py            # all splits
python preprocess.py --split epitope_group  # single split
```

Output: `trans_baselines/abdockgen/{split_name}/{train,val,test}.jsonl`

### Training

```bash
python chimera_trainer.py --split epitope_group --wandb
python chimera_trainer.py --split epitope_group --max_epoch 1  # smoke test
```

Results: `results/abdockgen/abdockgen_H3_{split}/`

### Evaluation

```bash
# Full evaluation on saved predictions:
python chimera_evaluate.py --aggregate --split epitope_group

# Or point to a specific predictions directory:
python chimera_evaluate.py --predictions /path/to/predictions/H3/
```

### Full pipeline (all splits)

```bash
GPU=0 bash chimera_train.sh
```

### CHIMERA-specific files

| File | Purpose |
|------|---------|
| `config.yaml` | Hyperparameters (model, training, wandb) |
| `preprocess.py` | Convert complex_features to JSONL |
| `chimera_trainer.py` | Train + inference + per-sample metrics |
| `chimera_evaluate.py` | Full 14-metric evaluation with bootstrap CIs |
| `chimera_train.sh` | Run all splits end-to-end |
