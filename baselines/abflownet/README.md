# AbFlowNet

This codebase was adapted from [[DiffAb]](https://github.com/luost26/diffab)

## Install

### Environment

```bash
conda env create -f env.yaml -n abflownet
conda activate abflownet
```

The default `cudatoolkit` version is 11.3. You may change it in [`env.yaml`](./env.yaml).

### Datasets and Trained Weights

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). Extract `all_structures.zip` into the `data` folder. 

The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/).

Trained model weights are available [**here** (Hugging Face)](https://huggingface.co/luost26/DiffAb/tree/main) or [**here** (Google Drive)](https://drive.google.com/drive/folders/15ANqouWRTG2UmQS_p0ErSsrKsU4HmNQc?usp=sharing).


### [Optional] PyRosetta

PyRosetta is required to relax the generated structures and compute binding energy. Please follow the instruction [**here**](https://www.pyrosetta.org/downloads) to install.


## Train AbFlowNet (For RAbD Benchmarking)

```
python train.py configs/train/codesign_single.yml
```

Default training parameters `max_iters=200_000`, `start_tb_after=195_000` and `train.loss_weights.tb=0.000005` are set in `configs/train/codesign_single.yml` file. We precomputed the binding energies with PyRosetta and saved the results in `precomputed_energies.pkl`, used in `train.py`.

## Test On RAbD
The trained model weights are in `trained_models`

Run `./generate_test.sh` to generate CDRs on the 58 RAbD test complexes using `trained_models\abflownet.pt`

### For energy calculation
`python ./abflownet/tools/eval/run.py --pfx="" --root <path to generated pdbs>`

A `summary.csv` will be created in the same directory where generated pdbs are.

run `energy_eval.py` and give the path of the generated `summary.csv` as `--csv_path` as input argument.



## Switching to DiffAb test complexes
The code for loading the DiffAb test complexes are in `diffab/datasets/sabdab_diffab.py`


## Visualization of Proteins

For the visualization of antibody–antigen interactions, we used the 5MES protein complex. This structure features a broadly neutralizing antibody bound to the HIV-1 envelope glycoprotein (gp120), offering a clear view of the paratope-epitope interface. We used [**PyMOL**](https://pymol.org/). The following script can be used to generate a consistent view highlighting specific complementarity-determining regions (CDRs).

Save the following as a `.pml` script and run it in PyMOL after loading the desired PDB:

```pml
load 5mes_H3/0047.pdb, complex
hide all
bg_color white

# --- Chain assignments (based on your file: H=D, L=C, Ag=B) ---
select heavy, chain H
select light, chain L
select antigen, chain A
select antibody, heavy or light

# --- Base Visualization ---
show cartoon, antibody
show surface, antigen

# Set high transparency for the *entire* antigen first
# (The contacting parts will be made opaque later)
set transparency, 0.7, antigen 

#   Color Reset & CDR Definitions
color gray80, antibody

# Heavy chain (Chain D) CDRs 
select CDR_H1, heavy and resi 26-32
select CDR_H2, heavy and resi 52-56
select CDR_H3, heavy and resi 95-102

# Light chain (Chain C) CDRs 
select CDR_L1, light and resi 24-34
select CDR_L2, light and resi 50-56
select CDR_L3, light and resi 89-97

# --- Group CDRs for easy selection ---
select CDRs_all, CDR_H1 or CDR_H2 or CDR_H3 or CDR_L1 or CDR_L2 or CDR_L3
select CDRs_focus, CDR_H3
select CDRs_other, CDRs_all and not CDRs_focus
select Framework, antibody and not CDRs_all

# --- CDR COLORS ---
color gray50, CDRs_other

# Color CDR H3 by hydrophobicity to match antigen surface
select CDR_H3_hydrophobic, CDR_H3 and (resn ALA,VAL,LEU,ILE,PHE,TRP,MET,PRO,GLY,CYS)
select CDR_H3_hydrophilic, CDR_H3 and not CDR_H3_hydrophobic

color green, CDR_H3_hydrophobic   
color red, CDR_H3_hydrophilic   

# color marine, CDR_L3

# Make non-key antibody parts transparent to focus on CDR3
set cartoon_transparency, 0.7, Framework
set cartoon_transparency, 0.7, CDRs_other

# Define the Interface based ONLY on CDR3s
select ab_interface, (CDRs_focus within 4.5 of antigen)
select ag_interface, (antigen within 4.5 of CDRs_focus)
select interface, ab_interface or ag_interface

# Show Interface Sidechains (Opaque)
show sticks, interface
set stick_radius, 0.15
# util.cnc("interface") # Color sticks by element

#   Find and Color Interactions

# --- SALT BRIDGES (Magenta) ---
select ab_pos, (ab_interface and (resn LYS,ARG))
select ab_neg, (ab_interface and (resn ASP,GLU))
select ag_pos, (ag_interface and (resn LYS,ARG))
select ag_neg, (ag_interface and (resn ASP,GLU))

dist salt_bridges, (ab_pos and elem N), (ag_neg and elem O), 4.0
dist salt_bridges, (ab_neg and elem O), (ag_pos and elem N), 4.0
color magenta, salt_bridges

# --- HYDROGEN BONDS (Green) ---
select H_bond_donors, (ab_interface and (elem N,O) and not (ab_pos or ab_neg))
select H_bond_acceptors, (ag_interface and (elem N,O) and not (ag_pos or ag_neg))
select H_bond_donors_ag, (ag_interface and (elem N,O) and not (ag_pos or ag_neg))
select H_bond_acceptors_ab, (ab_interface and (elem N,O) and not (ab_pos or ab_neg))

dist H_bonds, H_bond_donors, H_bond_acceptors, 3.2
dist H_bonds, H_bond_donors_ag, H_bond_acceptors_ab, 3.2
color green, H_bonds

# --- HYDROPHOBIC POCKETS (Orange) ---
select ag_hydrophobic, (ag_interface and (resn ALA,VAL,LEU,ILE,PHE,TRP,MET,PRO,GLY,CYS))
select ag_polar, (ag_interface and not ag_hydrophobic)

# --- Color the antigen surface patches ---
# Define custom lighter colors
set_color light_yellow, [1.0, 1.0, 0.7]
set_color light_orange, [1.0, 0.8, 0.6]

color gray80, antigen         
color light_orange, ag_polar    
color light_yellow, ag_hydrophobic 

set transparency, 0.0, ag_interface

#   Polish & Output
set dash_gap, 0.15
set dash_width, 2
set surface_quality, 1
set cartoon_fancy_helices, 1
set cartoon_fancy_sheets, 1
set two_sided_lighting, on
set spec_power, 70
set antialias, 2

zoom (ag_interface or CDRs_focus), 1.2

# ray 1200, 1200
# png final_figure.png, dpi=300
```



## CHIMERA-Bench Integration

Five files added for CHIMERA-Bench retraining and evaluation:

| File | Description |
|------|-------------|
| `config.yaml` | Hyperparameters from `codesign_multicdrs.yml`. No TB loss weights (multi-CDR mode). Paths from `shared_config.yaml`. |
| `preprocess.py` | Chothia renumbering -> LMDB via `abflownet.datasets.sabdab_diffab.preprocess_sabdab_structure()`. Category A format. |
| `chimera_trainer.py` | Iteration-based trainer (200K iters). TB loss disabled via `start_tb_after=999999999` + dummy energies. Multi-CDR mode with `MaskMultipleCDRs`. |
| `chimera_evaluate.py` | `--predictions` (single CDR) or `--aggregate` (all CDRs per split). 12-metric CSV output. |
| `chimera_train.sh` | Trains once per split, runs `eval_and_append` after each with `--aggregate`. |

**Key differences from DiffAb integration:**
- All imports use `abflownet.*` package
- `model.forward()` requires extra args (`energies`, `it`, `start_tb_after`) for TB loss -- disabled via dummy values
- Run name: `abflownet_multicdrs_{split}`
- Data stored in `trans_baselines/abflownet/`

**Usage:**
```bash
cd baselines/abflownet
python preprocess.py                                    # one-time preprocessing
GPU=0 bash chimera_train.sh                             # train all splits
GPU=0 SPLITS="epitope_group" bash chimera_train.sh      # single split
```