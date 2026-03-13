"""Preprocess CHIMERA dataset into AbFlowNet's native format.

AbFlowNet uses the same LMDB + Chothia format as DiffAb. This script
renumbers PDBs to Chothia numbering via abnumber, then runs AbFlowNet's
preprocess_sabdab_structure() to build LMDB cache.

Output structure (in trans_baselines/abflownet/):
    chothia/               -- Chothia-renumbered PDBs
    processed/
        structures.lmdb    -- LMDB cache (same format as DiffAb)
        structures.lmdb-ids
    idx_to_cid.json        -- [complex_id, ...] ordered by LMDB key order
    complex_ids.json       -- {chimera_complex_id: lmdb_key}
    renumber_log.json      -- renumbering results per complex

Usage:
    cd baselines/abflownet
    python preprocess.py
"""

import csv
import json
import logging
import os
import pickle
import sys
from pathlib import Path

import lmdb
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, _SCRIPT_DIR)
from abflownet.tools.renumber.run import renumber as renumber_pdb
from abflownet.datasets.sabdab_diffab import preprocess_sabdab_structure

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import load_shared_config

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

MAP_SIZE = 32 * (1024 * 1024 * 1024)  # 32GB


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

    # Group by PDB to avoid re-renumbering the same PDB
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
    """Build LMDB cache using AbFlowNet's preprocess_sabdab_structure()."""
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

    # Save IDs list
    with open(lmdb_path + "-ids", "wb") as f:
        pickle.dump(succeeded_ids, f)

    db_conn.close()
    return succeeded_ids


def main():
    shared = load_shared_config()
    data_root = shared["paths"]["data_root"]
    output_dir = os.path.join(shared["paths"]["trans_baselines"], "abflownet")
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
    log.info("Saved renumber log to %s", renumber_log_path)

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
    log.info("Saved idx_to_cid (%d entries) to %s", len(idx_to_cid), idx_to_cid_path)

    complex_ids_path = os.path.join(output_dir, "complex_ids.json")
    with open(complex_ids_path, "w") as f:
        json.dump(cid_to_idx, f, indent=2)
    log.info("Saved complex_ids (%d entries) to %s", len(cid_to_idx), complex_ids_path)

    log.info("AbFlowNet preprocessing complete.")


if __name__ == "__main__":
    main()
