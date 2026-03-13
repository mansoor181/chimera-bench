# dyAb: Flow Matching for Flexible Antibody Design with AlphaFold-driven Pre-binding Antigen

The official implementation of our AAAI 25' paper [dyAb: Flow Matching for Flexible Antibody Design with AlphaFold-driven Pre-binding Antigen](https://ojs.aaai.org/index.php/AAAI/article/view/32061).


## Quick Links

- [Setup](#setup)
- [Experiments](#experiments)
    - [Data Preprocessing](#data-preprocessing)
    - [CDR-H3 Design](#cdr-h3-design)
    - [Complex Structure Prediction](#complex-structure-prediction)
    - [Affinity Optimization](#affinity-optimization)
- [Contact](#contact)
- [Acknowledgements](#acknowledgements)
- [Others](#others)


## Setup

The environment and prerequisites 

**1. Environment**

We have provided the `env.yml` for creating the runtime conda environment just by running:

```bash
conda env create -f env.yml
```

**2. Scorers**

Please first prepare the scorers for TMscore and DockQ as follows:

The source code for assessing TMscore is at `evaluation/TMscore.cpp`. Please compile it by:
```bash
g++ -static -O3 -ffast-math -lm -o evaluation/TMscore evaluation/TMscore.cpp
```

To prepare the DockQ scorer, please clone its [official github](https://github.com/bjornwallner/DockQ) and compile the prerequisites according to its instructions. After that, please revise the `DOCKQ_DIR` variable in the `configs.py` to point to the directory containing the DockQ project (e.g. ./DockQ).

The lDDT scorer is in the conda environment, and the $\Delta\Delta G$ scorer is integrated into our codes, therefore they don't need additional preparations.

**3. PDB data**

Please download all the structure data of antibodies from the [download page of SAbDab](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/search/?all=true). Please enter the *Downloads* tab on the left of the web page and download the archived zip file for the structures, then decompress it:

```bash
wget https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/ -O all_structures.zip
unzip all_structures.zip
```

You should get a folder named *all_structures* with the following hierarchy:

```
├── all_structures
│   ├── chothia
│   ├── imgt
│   ├── raw
```

Each subfolder contains the pdb files renumbered with the corresponding scheme. We use IMGT in the paper, so the imgt subfolder is what we care about.

Since pdb files are heavy to process, usually people will generate a summary file for the structural database which records the basic information about each structure for fast access. We have provided the summary of the dataset retrieved at November 12, 2022 (`summaries/sabdab_summary.tsv`). Since the dataset is updated on a weekly basis, if you want to use the latest version, please download it from the [official website](http://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/about/).

For the AFDB part, we extracted the antigen sequence from the original antigen-antibody complex and used AlphaFold2 to predict its structure. Then, we aligned them with the original antigen structures using Kabsch algorithm and replaced the original epitopes with the AlphaFold-predicted ones. We have provided the processed dataset we curated for the experiments, which covers all the required PDB files among various datasets. The datasets and checkpoints could be downloaded [here](https://zenodo.org/records/15226646). After unzipping the folder, the `AF2_antigen_aligned_pdbs folder` should be placed under `all_data` folder, and the corresponding json files should also be placed there. All other checkpoints should be placed under `checkpoints` folder.

### Data Preprocessing

**Data**

We follow the data preprocessing techiniques proposed by DyMEAN, who have provided the script for all these procedures in `scripts/data_preprocess.sh`. Suppose the IMGT-renumbered pdb data are located at `all_structures/imgt/`, and that you want to store the processed data (~5G) at `all_data`, you can simply run:

```bash
bash scripts/data_preprocess.sh all_structures/imgt all_data
```
Then, we have provided the AF2-predicted antigens needed in the training/inferencing process. Follow the link to download the `AF2_antigen_aligned_pdbs` folder and `af2_monomer_pdbid.json`, place them under `all_data` folder. 

## Experiments

The trained checkpoints for each task are provided here [github release page](https://github.com/THUNLP-MT/dyMEAN/releases/tag/v1.0.0). To use them, please download the ones you are interested in and save them into the folder `./checkpoints`. For the affinity optimization task, we used the same predictor as dyMEAN's.

### CDR-H3 Design
We use SAbDab for training and RAbD for testing. Please first revise the settings in `scripts/train/configs/single_cdr_design.json` (path to datasets and other hyperparameters) and then run the below command for training:
```bash
python train.py --task single_cdr_design --gpus 1 --wandb_offline 1 --flexible 1 --model_type dyAb --ex_name dyAb_single_cdrh3_m1  --module_type 1

```
We have also provided the trained checkpoint at `checkpoints/best_cdr.ckpt`. Then please revise the path to the test set in `scripts/test/test.sh` and run the following command for testing:

```
bash test_cdr.sh <ckpt_path> <test_set_path> <save_dir> <use_af2ag>
```

For example, you can run 

```bash
bash test_cdr.sh "./checkpoints/best_cdrh3.ckpt" \
                     "./all_data/RAbD/test.json" \
                     "./results/single_cdr_design/" \
                     "True"
```

which will save the generated results (PDB files and a `summary.json`) to `./results/single_cdr_design/`.


### Affinity Optimization
We use SAbDab for training and the antibodies in SKEMPI V2.0 for testing. Similarly, please first revise the settings in `scripts/train/configs/single_cdr_opt.json`. 

For training, you can run 
```python

python train.py --task single_cdr_opt --gpus 1 --wandb_offline 1 --flexible 1 --model_type dyAbOpt --train_set ./all_data/SKEMPI/train.json --valid_set ./all_data/SKEMPI/valid.json --test_set all_data/SKEMPI/test.json --save_dir ./all_data/RAbD/single_cdr_opt/ --ex_name dyAbOpt_v1 --module_type 1
```

We have provided the trained checkpoints at `checkpoints/best_affi_opt.ckpt` and `checkpoints/cdrh3_ddg_predictor.ckpt`. The optimization test can be conducted through:

```python
python opt_gen_new.py --ckpt ./checkpoints/best_opt.ckpt \
 --predictor_ckpt ./checkpoints/cdrh3_ddg_predictor.ckpt \
 --summary_json ./all_data/SKEMPI/test.json \
 --save_dir ./results/affi_8_50/ \
  --num_residue_changes 8 \
  --num_optimize_steps 50
```
which will do 50 steps of gradient search without restrictions on the maximum number of changed residues (change 8 to any number to restrict the upperbound of $\Delta L$).




### Complex Structure Prediction
Same as dyMEAN, we also use SAbDab for training and IgFold for testing.
To train the model, you can run:
```
python train.py --task struct_prediction --gpus 1 --wandb_offline 1 --flexible 1 --model_type dyAb --ex_name dyAb_struct_pred_2 --module_type 1 --batch_size 64&

```
We have also provided the trained checkpoint at `checkpoints/best_struc_pred.ckpt`. Then please run the following command for testing:
Otherwise, you can also run the following command to test the model.

```bash
bash test_pred.sh "./checkpoints/best_pred.ckpt" \
                     "./all_data/IgFold/test.json" \
                     "./results/struct_pred/" \
                     "True"
```

### Evaluation
To obtain the metrics for CDR-H3 design and Structure prediction task, you can run
```
python cal_metrics.py --test_set "path to your summery.json"
```
As you can find a `summary.json` under each folder you choose to store the results, you may also change the config in `./scripts/configs`.

*Note: you may change the path for the referenced PDB file(i.e. before/after renumbering), which may introduce slight variance to the results.*

For the metrics in the affinity optimization task, you can easily find it in the `log.txt` file under the output folder.

## Contact

Feel free to contact us at tancheng@westlake.edu.cn or yj.zhang@mail.mcgill.ca 

## Acknowledgements

We highly thank "dyMEAN: End-to-End Full-Atom Antibody Design". [[paper](https://arxiv.org/pdf/2302.00203), [code](https://github.com/THUNLP-MT/dyMEAN)]

## Debugging Issues

Also, if you encounter any issues related to Rosetta, you can simply go into the `Rosetta/rosetta.source.release-337/main/source` folder and recomplie it using the following command:

```
./scons.py mode=release bin 
```

## CHIMERA-Bench Integration

The following files adapt dyAb for the CHIMERA-Bench retraining pipeline:

| File | Description |
|------|-------------|
| `config.yaml` | Hyperparameters (embed_dim, hidden_size, n_layers, etc.) |
| `preprocess.py` | Converts CHIMERA PDBs to dyAb's pkl cache format via `RobustE2EDataset` |
| `chimera_trainer.py` | Per-CDR training loop (raw PyTorch, replaces Lightning) |
| `chimera_evaluate.py` | 12-metric evaluation (`--predictions` or `--aggregate` mode) |
| `chimera_train.sh` | End-to-end script: train + evaluate across splits and CDR types |

**Key differences from original dyAb**:
- Uses CHIMERA's 2981-complex dataset instead of SAbDab/RAbD
- Raw PyTorch training loop instead of PyTorch Lightning
- 3 CHIMERA splits (epitope_group, antigen_fold, temporal) instead of random split
- Per-CDR pattern: trains H1, H2, H3 separately (like dyMEAN)
- AlphaFold2 pre-binding antigen feature disabled (uses native antigen structures)
- Full 12-metric CHIMERA evaluation suite

**Usage**:
```bash
# Preprocess (once)
python preprocess.py

# Train all splits and CDR types
GPU=0 bash chimera_train.sh

# Train specific split/CDR
GPU=0 SPLITS="epitope_group" CDR_ARGS="H3" bash chimera_train.sh

# Evaluate only
python chimera_evaluate.py --predictions /path/to/predictions/ --split epitope_group
python chimera_evaluate.py --aggregate --split epitope_group
```

## Citation

```text
@inproceedings{tan2025dyab,
  title={dyAb: Flow Matching for Flexible Antibody Design with AlphaFold-driven Pre-binding Antigen},
  author={Tan, Cheng and Zhang, Yijie and Gao, Zhangyang and Huang, Yufei and Lin, Haitao and Wu, Lirong and Wu, Fandi and Blanchette, Mathieu and Li, Stan Z},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={39},
  number={1},
  pages={782--790},
  year={2025}
}
```