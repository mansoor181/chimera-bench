"""Preprocess CHIMERA dataset into RADAb's native format.

RADAb uses LMDB + Chothia format (same as DiffAb) plus per-CDR reference
FASTA files for its retrieval-augmented generation. This script:
1. Renumbers PDBs to Chothia numbering
2. Builds LMDB cache via RADAb's preprocess_sabdab_structure()
3. Builds per-split reference FASTA files from training CDR sequences

Output structure (in trans_baselines/radab/):
    chothia/               -- Chothia-renumbered PDBs
    processed/
        structures.lmdb    -- LMDB cache
        structures.lmdb-ids
    ref_seqs/{split}/      -- Per-split reference FASTA files
        H_CDR1.fasta       -- Heavy chain CDR1 sequences
        H_CDR2.fasta
        H_CDR3.fasta
        ref_sequences_chothia_CDR4.fasta  -- Light chain L1
        ref_sequences_chothia_CDR5.fasta  -- L2
        ref_sequences_chothia_CDR6.fasta  -- L3
    idx_to_cid.json
    complex_ids.json
    renumber_log.json

Usage:
    cd baselines/radab
    python preprocess.py
"""

import csv
import json
import logging
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# RADAb's code is in src/diffab/
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "src"))
from diffab.tools.renumber.run import renumber as renumber_pdb
from diffab.datasets.sabdab import preprocess_sabdab_structure
from diffab.utils.protein.constants import CDR, AA

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import load_shared_config, load_split_ids

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB

# RADAb AA ordering: alphabetical by 1-letter code
AA_INDEX_TO_ONE = {int(aa): aa.name[0] if len(aa.name) == 3 else 'X'
                   for aa in AA if aa != AA.UNK}
# Build explicitly for clarity
AA_INDEX_TO_ONE = {
    0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I',
    8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S',
    16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X',
}

# CDR enum -> FASTA filename (matches RADAb's hardcoded names)
CDR_TO_FASTA = {
    CDR.H1: "H_CDR1.fasta",
    CDR.H2: "H_CDR2.fasta",
    CDR.H3: "H_CDR3.fasta",
    CDR.L1: "ref_sequences_chothia_CDR4.fasta",
    CDR.L2: "ref_sequences_chothia_CDR5.fasta",
    CDR.L3: "ref_sequences_chothia_CDR6.fasta",
}


def load_chimera_entries(data_root):
    """Load all CHIMERA complex entries from final_summary.csv."""
    summary_path = Path(data_root) / "processed" / "final_summary.csv"
    structures_dir = Path(data_root) / "raw" / "structures"
    entries = []
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            pdb = row["pdb"]
            pdb_path = structures_dir / f"{pdb}.pdb"
            if not pdb_path.exists():
                continue
            ag_chains = [c.strip() for c in row["antigen_chain"].split("|")]
            entries.append({
                "complex_id": row["complex_id"],
                "pdb": pdb,
                "pdb_path": str(pdb_path),
                "H_chain": row["Hchain"] if row["Hchain"] else None,
                "L_chain": row["Lchain"] if row["Lchain"] else None,
                "ag_chains": ag_chains,
            })
    return entries


def renumber_all(entries, chothia_dir):
    """Renumber PDBs to Chothia and return renumbering log."""
    os.makedirs(chothia_dir, exist_ok=True)
    renumber_log = {}

    pdb_to_entries = {}
    for entry in entries:
        pdb_to_entries.setdefault(entry["pdb"], []).append(entry)

    for pdb, pdb_entries in tqdm(pdb_to_entries.items(), desc="Renumber"):
        out_path = os.path.join(chothia_dir, f"{pdb}.pdb")
        if os.path.exists(out_path):
            renumber_log[pdb] = {"status": "cached"}
            continue
        in_path = pdb_entries[0]["pdb_path"]
        try:
            heavy_chains, light_chains, other_chains = renumber_pdb(
                in_path, out_path, return_other_chains=True)
            renumber_log[pdb] = {
                "status": "ok",
                "heavy_chains": heavy_chains,
                "light_chains": light_chains,
                "other_chains": other_chains,
            }
        except Exception as e:
            log.warning("Renumber failed for %s: %s", pdb, e)
            renumber_log[pdb] = {"status": "failed", "error": str(e)}
    return renumber_log


def build_lmdb(entries, chothia_dir, lmdb_path):
    """Build LMDB cache using RADAb's preprocess_sabdab_structure()."""
    tasks = []
    for entry in entries:
        pdb_path = os.path.join(chothia_dir, f"{entry['pdb']}.pdb")
        if not os.path.exists(pdb_path):
            log.warning("Chothia PDB missing for %s, skipping", entry["pdb"])
            continue
        entry_id = entry["complex_id"]
        task = {
            "id": entry_id,
            "entry": {
                "id": entry_id,
                "pdbcode": entry["pdb"],
                "H_chain": entry["H_chain"],
                "L_chain": entry["L_chain"],
                "ag_chains": entry["ag_chains"],
            },
            "pdb_path": pdb_path,
        }
        tasks.append(task)

    log.info("Preprocessing %d structures...", len(tasks))
    succeeded_ids = []

    db_conn = lmdb.open(
        lmdb_path, map_size=MAP_SIZE, create=True,
        subdir=False, readonly=False,
    )

    with db_conn.begin(write=True, buffers=True) as txn:
        for task in tqdm(tasks, desc="Preprocess"):
            try:
                data = preprocess_sabdab_structure(task)
            except Exception as e:
                log.warning("Preprocess failed for %s: %s", task["id"], e)
                continue
            if data is None:
                log.warning("Preprocess returned None for %s", task["id"])
                continue
            if data.get("heavy") is None and data.get("light") is None:
                log.warning("No valid chains for %s, skipping", task["id"])
                continue
            succeeded_ids.append(task["id"])
            txn.put(task["id"].encode("utf-8"), pickle.dumps(data))

    with open(lmdb_path + "-ids", "wb") as f:
        pickle.dump(succeeded_ids, f)

    db_conn.close()
    return succeeded_ids


def extract_cdr_data(lmdb_path, complex_ids):
    """Extract per-CDR sequences AND backbone coords from LMDB structures.

    Returns: dict {cdr_enum: {pdb_code: [(seq_str, ca_coords), ...]}}
        where ca_coords is a numpy array of shape (L, 3)
    """
    cdr_data = {cdr: defaultdict(list) for cdr in CDR_TO_FASTA}

    db_conn = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=False,
                        subdir=False, readonly=True, lock=False)

    with db_conn.begin() as txn:
        for cid in tqdm(complex_ids, desc="Extract CDR data"):
            raw = txn.get(cid.encode("utf-8"))
            if raw is None:
                continue
            structure = pickle.loads(raw)
            pdb_code = cid[:4].lower()

            for chain_key, cdr_enums in [
                ("heavy", [CDR.H1, CDR.H2, CDR.H3]),
                ("light", [CDR.L1, CDR.L2, CDR.L3]),
            ]:
                chain_data = structure.get(chain_key)
                if chain_data is None:
                    continue
                cdr_flag = chain_data.get("cdr_flag")
                aa = chain_data.get("aa")
                pos = chain_data.get("pos_heavyatom")
                if cdr_flag is None or aa is None or pos is None:
                    continue

                for cdr_enum in cdr_enums:
                    mask = (cdr_flag == int(cdr_enum))
                    if mask.sum() == 0:
                        continue
                    seq = "".join(AA_INDEX_TO_ONE.get(a.item(), "X")
                                 for a in aa[mask])
                    if 5 <= len(seq) <= 30:
                        ca_coords = pos[mask, 1].numpy()  # CA at atom index 1
                        cdr_data[cdr_enum][pdb_code].append((seq, ca_coords))

    db_conn.close()
    return cdr_data


def _cdr_backbone_rmsd(coords_a, coords_b):
    """Compute CA backbone RMSD between two CDR fragments after Kabsch alignment.

    Returns inf if lengths differ.
    """
    if len(coords_a) != len(coords_b):
        return float("inf")
    if len(coords_a) == 0:
        return float("inf")

    # Center
    ca = coords_a - coords_a.mean(axis=0)
    cb = coords_b - coords_b.mean(axis=0)

    # Kabsch: find optimal rotation
    H = ca.T @ cb
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    ca_aligned = ca @ R.T
    rmsd = np.sqrt(np.mean(np.sum((ca_aligned - cb) ** 2, axis=1)))
    return rmsd


def build_ref_fasta(cdr_data, output_dir, train_pdbs=None):
    """Build per-CDR reference FASTA files ranked by backbone RMSD similarity.

    For each PDB (train or test), writes the top-30 most structurally similar
    CDR sequences from train PDBs only. This mimics RADAb's original MASTER
    structural search but uses pairwise CA RMSD within the CHIMERA dataset.

    Args:
        cdr_data: dict {cdr_enum: {pdb_code: [(seq, ca_coords), ...]}}
        output_dir: directory to write FASTA files
        train_pdbs: set of PDB codes allowed as retrieval sources (train only).
                    If None, all PDBs are used.
    """
    os.makedirs(output_dir, exist_ok=True)

    for cdr_enum, fasta_name in CDR_TO_FASTA.items():
        fasta_path = os.path.join(output_dir, fasta_name)
        pdb_data = cdr_data[cdr_enum]
        all_pdbs = sorted(pdb_data.keys())

        # Build flat list of (pdb_code, seq, coords) for train PDBs only
        train_entries = []
        for pdb_code in all_pdbs:
            if train_pdbs is not None and pdb_code not in train_pdbs:
                continue
            for seq, coords in pdb_data[pdb_code]:
                train_entries.append((pdb_code, seq, coords))

        log.info("  %s: %d PDBs, %d train entries for retrieval",
                 fasta_name, len(all_pdbs), len(train_entries))

        with open(fasta_path, "w") as f:
            for query_pdb in tqdm(all_pdbs, desc=fasta_name, leave=False):
                f.write(f">{query_pdb}\n")

                # Get query CDR coords (use first complex's CDR for this PDB)
                query_entries = pdb_data[query_pdb]
                query_seq, query_coords = query_entries[0]

                # Write native sequence first (expected by retrieval function)
                f.write(f"{query_seq}\n")

                # Score all train entries by RMSD to query
                scored = []
                for t_pdb, t_seq, t_coords in train_entries:
                    if t_pdb == query_pdb:
                        continue  # exclude self
                    rmsd = _cdr_backbone_rmsd(query_coords, t_coords)
                    scored.append((rmsd, t_seq))

                # Sort by RMSD (most similar first), take top 29 (native already written)
                scored.sort(key=lambda x: x[0])
                for _, seq in scored[:29]:
                    f.write(f"{seq}\n")

        log.info("  %s: wrote FASTA for %d PDBs", fasta_name, len(all_pdbs))


def main():
    shared = load_shared_config()
    data_root = shared["paths"]["data_root"]
    output_dir = os.path.join(shared["paths"]["trans_baselines"], "radab")
    os.makedirs(output_dir, exist_ok=True)

    chothia_dir = os.path.join(output_dir, "chothia")
    processed_dir = os.path.join(output_dir, "processed")
    os.makedirs(processed_dir, exist_ok=True)

    # Step 1: Load entries
    log.info("Loading CHIMERA entries...")
    entries = load_chimera_entries(data_root)
    log.info("Found %d CHIMERA complexes", len(entries))

    # Step 2: Renumber to Chothia
    log.info("Renumbering PDBs to Chothia...")
    renumber_log = renumber_all(entries, chothia_dir)
    renumber_log_path = os.path.join(output_dir, "renumber_log.json")
    with open(renumber_log_path, "w") as f:
        json.dump(renumber_log, f, indent=2)

    n_failed = sum(1 for v in renumber_log.values() if v["status"] == "failed")
    log.info("Renumber: %d ok, %d failed out of %d PDBs",
             len(renumber_log) - n_failed, n_failed, len(renumber_log))

    # Step 3: Build LMDB
    lmdb_path = os.path.join(processed_dir, "structures.lmdb")
    if os.path.exists(lmdb_path):
        log.info("LMDB already exists at %s, skipping", lmdb_path)
        with open(lmdb_path + "-ids", "rb") as f:
            succeeded_ids = pickle.load(f)
    else:
        succeeded_ids = build_lmdb(entries, chothia_dir, lmdb_path)
    log.info("LMDB: %d structures", len(succeeded_ids))

    # Step 4: Save index mappings
    idx_to_cid = succeeded_ids
    cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}

    idx_to_cid_path = os.path.join(output_dir, "idx_to_cid.json")
    with open(idx_to_cid_path, "w") as f:
        json.dump(idx_to_cid, f, indent=2)
    log.info("Saved idx_to_cid (%d entries)", len(idx_to_cid))

    complex_ids_path = os.path.join(output_dir, "complex_ids.json")
    with open(complex_ids_path, "w") as f:
        json.dump(cid_to_idx, f, indent=2)

    # Step 5: Build per-split reference FASTA files for retrieval.
    # Extract CDR data (sequences + backbone coords) from ALL complexes.
    # Then for each split, write FASTA with sequences ranked by backbone RMSD
    # similarity, using ONLY train complexes as retrieval sources.
    available_ids = set(idx_to_cid)
    all_available = [cid for cid in idx_to_cid if cid in available_ids]
    log.info("Extracting CDR data from %d complexes...", len(all_available))
    cdr_data = extract_cdr_data(lmdb_path, all_available)

    for split_name in ["epitope_group", "antigen_fold", "temporal"]:
        log.info("Building RMSD-ranked FASTA for split=%s...", split_name)
        split_ids = load_split_ids(split_name, data_root)
        train_ids = [cid for cid in split_ids.get("train", [])
                     if cid in available_ids]
        train_pdbs = set(cid[:4].lower() for cid in train_ids)
        log.info("  Train PDBs for retrieval: %d", len(train_pdbs))

        fasta_dir = os.path.join(output_dir, "ref_seqs", split_name)
        build_ref_fasta(cdr_data, fasta_dir, train_pdbs=train_pdbs)

    log.info("RADAb preprocessing complete.")


if __name__ == "__main__":
    main()
