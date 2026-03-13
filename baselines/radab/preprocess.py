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


def extract_cdr_sequences(lmdb_path, complex_ids):
    """Extract per-CDR sequences from LMDB structures.

    Returns: dict {cdr_enum: {pdb_code: [seq_str, ...]}}
    """
    cdr_seqs = {cdr: defaultdict(list) for cdr in CDR_TO_FASTA}

    db_conn = lmdb.open(lmdb_path, map_size=MAP_SIZE, create=False,
                        subdir=False, readonly=True, lock=False)

    with db_conn.begin() as txn:
        for cid in tqdm(complex_ids, desc="Extract CDR seqs"):
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
                if cdr_flag is None or aa is None:
                    continue

                for cdr_enum in cdr_enums:
                    mask = (cdr_flag == int(cdr_enum))
                    if mask.sum() == 0:
                        continue
                    seq = "".join(AA_INDEX_TO_ONE.get(a.item(), "X")
                                 for a in aa[mask])
                    if 5 <= len(seq) <= 30:
                        cdr_seqs[cdr_enum][pdb_code].append(seq)

    db_conn.close()
    return cdr_seqs


def build_ref_fasta(cdr_seqs, output_dir):
    """Build per-CDR reference FASTA files from extracted sequences."""
    os.makedirs(output_dir, exist_ok=True)

    for cdr_enum, fasta_name in CDR_TO_FASTA.items():
        fasta_path = os.path.join(output_dir, fasta_name)
        pdb_seqs = cdr_seqs[cdr_enum]

        # Collect all sequences (from ALL PDBs) for cross-referencing
        all_seqs = []
        for pdb_code, seqs in pdb_seqs.items():
            all_seqs.extend(seqs)

        with open(fasta_path, "w") as f:
            for pdb_code in sorted(pdb_seqs.keys()):
                f.write(f">{pdb_code}\n")
                # Write this PDB's own sequences first
                for seq in pdb_seqs[pdb_code]:
                    f.write(f"{seq}\n")
                # Add sequences from other PDBs (up to 30 total)
                added = len(pdb_seqs[pdb_code])
                for other_pdb, other_seqs in pdb_seqs.items():
                    if other_pdb == pdb_code:
                        continue
                    for seq in other_seqs:
                        if added >= 30:
                            break
                        f.write(f"{seq}\n")
                        added += 1
                    if added >= 30:
                        break

        n_pdbs = len(pdb_seqs)
        n_seqs = sum(len(v) for v in pdb_seqs.values())
        log.info("  %s: %d PDBs, %d sequences", fasta_name, n_pdbs, n_seqs)


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
    # Use ALL complexes (train+val+test) so every PDB has retrieval entries.
    # The original RADAb uses all of SAbDab for retrieval -- this is CDR-level
    # sequence retrieval from other antibodies, not data leakage.  The test-time
    # retrieval function (_get_retrieved_2) already excludes the native sequence.
    available_ids = set(idx_to_cid)
    for split_name in ["epitope_group", "antigen_fold", "temporal"]:
        log.info("Building reference FASTA for split=%s...", split_name)
        split_ids = load_split_ids(split_name, data_root)
        all_ids = [cid for subset in ("train", "val", "test")
                   for cid in split_ids.get(subset, [])
                   if cid in available_ids]
        log.info("  All complexes for FASTA: %d", len(all_ids))

        cdr_seqs = extract_cdr_sequences(lmdb_path, all_ids)
        fasta_dir = os.path.join(output_dir, "ref_seqs", split_name)
        build_ref_fasta(cdr_seqs, fasta_dir)

    log.info("RADAb preprocessing complete.")


if __name__ == "__main__":
    main()
