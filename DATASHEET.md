# Datasheet for CHIMERA-Bench

Following the framework of Gebru et al. (2021), "Datasheets for Datasets."

---

## Motivation

**For what purpose was the dataset created?**
CHIMERA-Bench was created to provide a standardized, gold-standard benchmark for evaluating computational methods for epitope-specific antibody design. Existing benchmarks (RAbD-60, SKEMPI-53) are small, use inconsistent evaluation protocols, and lack epitope-specific annotations, making fair comparison across methods difficult.

**Who created the dataset and on behalf of which entity?**
[Authors], [Institution]. Created as part of the CHIMERA-Bench project for submission to NeurIPS Datasets & Benchmarks track.

**Who funded the creation of the dataset?**
[Funding information to be added.]

---

## Composition

**What do the instances that comprise the dataset represent?**
Each instance is an antibody-antigen complex structure from the Structural Antibody Database (SAbDab), comprising a paired heavy-light chain antibody bound to a protein or peptide antigen, with associated annotations.

**How many instances are there in total?**
2,981 deduplicated antibody-antigen complexes after quality filtering and deduplication at 95% sequence identity.

**Does the dataset contain all possible instances or is it a sample?**
It contains all qualifying antibody-antigen complexes from SAbDab as of the download date that pass quality filters: resolution <= 4.0 A, paired VH+VL chains, protein/peptide antigen only.

**What data does each instance consist of?**
Per complex:
- PDB structure file (heavy chain, light chain, antigen)
- ANARCI numbering in IMGT and Chothia schemes
- CDR annotations (H1-H3, L1-L3) under both schemes
- Epitope and paratope residue annotations (4.5 A cutoff)
- Residue-level contact pairs with distances
- Amino acid sequences (heavy, light, antigen)
- PyG HeteroData residue graph with multi-relational edges
- Per-chain atom14 and CA coordinates
- SKEMPI v2 linkage (where available, 29 complexes)

**Is there a label or target associated with each instance?**
The primary design targets are CDR sequences and structures (especially CDR-H3). Epitope residues serve as conditioning input. Ground truth includes native CDR sequences, backbone coordinates, and binding contacts.

**Is any information missing from individual instances?**
Some structures have incomplete backbone atoms (missing N, C, or O). These are marked with NaN in backbone coordinate tensors. 2 complexes have zero contacts at the 4.5 A threshold (edge cases with very loose binding).

**Are there recommended data splits?**
Yes, three split types are provided, all at 80/10/10 train/val/test ratio:
1. **Epitope-group**: Clusters by epitope similarity to prevent epitope leakage
2. **Antigen-fold**: Clusters by antigen fold to test generalization to new folds
3. **Temporal**: Date-based split for prospective evaluation

**Are there any errors, sources of noise, or redundancies?**
- 3 known PDB entries (3h3b, 2ghw, 3uzq) from RAbD have incorrect chain annotations in SAbDab
- Sequence deduplication at 95% identity means near-identical structures may remain
- Resolution varies from sub-angstrom to 4.0 A; lower-resolution structures have less reliable coordinates

**Is the dataset self-contained?**
Yes. All structures and annotations are included. The original PDB files can be independently verified from the RCSB PDB using the PDB codes.

---

## Collection Process

**How was the data associated with each instance acquired?**
Structures are from X-ray crystallography and cryo-EM experiments deposited in the PDB and curated by SAbDab (Oxford Protein Informatics Group). Affinity data is from the SKEMPI v2 database (experimentally measured binding free energies).

**What mechanisms or procedures were used to collect the data?**
1. Download SAbDab summary and all antibody-antigen PDB structures
2. Apply quality filters (resolution, chain pairing, antigen type)
3. Deduplicate using MMseqs2 at 95% CDR-H3 sequence identity, 80% coverage
4. Run ANARCI for antibody numbering in IMGT and Chothia schemes
5. Compute epitope/paratope/contact annotations at 4.5 A atom-atom cutoff
6. Link to SKEMPI v2 for binding affinity data

**If the dataset is a sample, what is the sampling strategy?**
Not applicable -- all qualifying structures are included (no sampling).

**Over what timeframe was the data collected?**
PDB structures span from the earliest antibody structures (~1990s) through the SAbDab snapshot download date. The temporal split separates by deposition date.

**Were any ethical review processes conducted?**
Not applicable -- the dataset consists entirely of publicly available structural data from the PDB, which are open-access scientific records.

---

## Preprocessing / Cleaning / Labeling

**Was any preprocessing/cleaning/labeling of the data done?**
Yes:
- Quality filtering removed structures with resolution > 4.0 A, non-protein/peptide antigens, and unpaired (VH-only or VL-only) antibodies
- MMseqs2 deduplication at 95% sequence identity on CDR-H3 to reduce redundancy
- ANARCI numbering standardizes residue positions across different PDB numbering schemes
- CDR boundaries defined by IMGT (primary) and Chothia (secondary) conventions
- Contact computation uses BioPython NeighborSearch for efficient atom-atom distance queries, deduplicated to residue-level pairs

**Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data?**
Yes. Raw PDB files are stored separately from processed annotations and features.

**Is the software used to preprocess/clean/label the instances available?**
Yes. The full pipeline is provided as `benchmark/pipeline.py` with all processing code in `benchmark/data/`.

---

## Uses

**Has the dataset been used for any tasks already?**
The dataset is designed for evaluation of 18 antibody design baseline methods spanning 5 format categories (SAbDab-style, RefineGNN JSONL, PDB+FASTA, 6D coordinates, CDR-only).

**Is there a repository that links to any or all papers or systems that use this dataset?**
[Repository URL to be added.]

**What (other) tasks could the dataset be used for?**
- Antibody-antigen binding prediction
- Epitope prediction
- Paratope prediction
- Antibody structure prediction
- Antibody humanization
- Affinity maturation (via SKEMPI-linked subset)

**Is there anything about the composition or collection that might impact future uses?**
The dataset is biased toward experimentally tractable complexes (those amenable to crystallography/cryo-EM). Membrane-bound antigens and intrinsically disordered regions are underrepresented.

**Are there tasks for which the dataset should not be used?**
The dataset should not be used for clinical antibody design without additional experimental validation. Computational predictions are not substitutes for experimental characterization.

---

## Distribution

**Will the dataset be distributed to third parties outside of the entity on behalf of which the dataset was created?**
Yes. The dataset will be publicly released.

**How will the dataset be distributed?**
- Zenodo (DOI-assigned archive)
- HuggingFace Hub (with data loaders)
- PyPI package (`chimera-bench`) for programmatic access
- Croissant metadata for ML dataset discovery

**When will the dataset be released?**
Upon acceptance of the accompanying paper.

**Will the dataset be distributed under a copyright or other IP license?**
CC-BY-4.0. The underlying PDB structures are public domain; annotations and derived features are released under Creative Commons Attribution.

**Have any third parties imposed IP-based or other restrictions on the data?**
No. All source data (PDB, SAbDab, SKEMPI v2) is publicly available for research use.

---

## Maintenance

**Who is supporting/hosting/maintaining the dataset?**
[Authors/Institution]. The dataset will be hosted on Zenodo (persistent DOI) and HuggingFace Hub.

**How can the owner/curator/manager of the dataset be contacted?**
[Contact information to be added.]

**Will the dataset be updated?**
Annual updates are planned to incorporate new structures deposited in the PDB. The temporal split ensures new structures are automatically assigned to test sets.

**Will older versions be supported/hosted/maintained?**
Yes. Each version will have a unique DOI on Zenodo. Previous versions will remain accessible.

**If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?**
Yes. A contribution guide and standardized format specification will be provided. The pipeline code is open-source and can be re-run with updated SAbDab snapshots.
