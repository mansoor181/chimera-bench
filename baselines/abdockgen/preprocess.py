"""Preprocess CHIMERA dataset into AbDockGen's JSONL format.

Converts complex_features/*.pt directly to AbDockGen JSONL. No renumbering
needed -- both CHIMERA and AbDockGen use IMGT numbering.

For each complex:
  1. Load .pt from processed/complex_features/
  2. Concatenate heavy + light chains as antibody
  3. Build antibody_cdr string from cdr_masks.imgt.heavy (H1='1', H2='2', H3='3', fw='0')
     Light chain residues are all '0' (framework)
  4. Extract antigen sequence + atom14 coords
  5. Write JSONL line

Output structure (in trans_baselines/abdockgen/):
    {split_name}/train.jsonl
    {split_name}/val.jsonl
    {split_name}/test.jsonl
    idx_to_cid.json       -- [complex_id, ...] ordered by processing index
    complex_ids.json      -- {complex_id: index}

Usage:
    cd baselines/abdockgen
    python preprocess.py
    python preprocess.py --split temporal
"""

import argparse
import json
import os
import sys

import numpy as np
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import load_shared_config, load_split_ids

# CDR code in CHIMERA's cdr_masks.imgt.heavy -> AbDockGen label
_CDR_CODE_MAP = {0: "1", 1: "2", 2: "3"}  # H1, H2, H3


def convert_complex(feat):
    """Convert a CHIMERA complex_features dict to AbDockGen JSONL entry.

    Returns dict or None if the complex cannot be converted.
    """
    cid = feat["complex_id"]

    # Heavy chain
    h_seq = feat["heavy_sequence"]
    h_coords = np.array(feat["heavy_atom14_coords"], dtype=np.float64)  # (N_h, 14, 3)

    # Light chain
    l_seq = feat["light_sequence"]
    l_coords = np.array(feat["light_atom14_coords"], dtype=np.float64)  # (N_l, 14, 3)

    # Antibody = heavy + light
    ab_seq = h_seq + l_seq
    ab_coords = np.concatenate([h_coords, l_coords], axis=0)  # (N_h+N_l, 14, 3)

    # Build CDR label string
    h_mask = feat["cdr_masks"]["imgt"]["heavy"]  # list of ints: -1=fw, 0=H1, 1=H2, 2=H3
    cdr_str = ""
    for code in h_mask:
        cdr_str += _CDR_CODE_MAP.get(code, "0")
    # Light chain: all framework
    cdr_str += "0" * len(l_seq)

    if len(cdr_str) != len(ab_seq):
        return None

    # Antigen
    ag_seq = feat["antigen_sequence"]
    ag_coords = np.array(feat["antigen_atom14_coords"], dtype=np.float64)  # (M, 14, 3)

    if len(ag_seq) == 0:
        return None

    return {
        "pdb": cid.split("_")[0],
        "complex_id": cid,
        "antibody_seq": ab_seq,
        "antibody_cdr": cdr_str,
        "antibody_coords": ab_coords.tolist(),
        "antigen_seq": ag_seq,
        "antigen_coords": ag_coords.tolist(),
    }


def write_split_jsonl(entries, cid_set, output_path):
    """Write JSONL for a subset of complexes defined by cid_set."""
    written = 0
    with open(output_path, "w") as f:
        for entry in entries:
            if entry["complex_id"] in cid_set:
                f.write(json.dumps(entry) + "\n")
                written += 1
    return written


def main():
    parser = argparse.ArgumentParser(description="Preprocess CHIMERA for AbDockGen")
    parser.add_argument("--split", default=None,
                        help="Single split to process (default: all splits)")
    args = parser.parse_args()

    shared = load_shared_config()
    data_root = shared["paths"]["data_root"]
    output_dir = os.path.join(shared["paths"]["trans_baselines"], "abdockgen")
    feat_dir = os.path.join(data_root, "processed", "complex_features")
    os.makedirs(output_dir, exist_ok=True)

    # Convert all complexes
    import torch
    feat_files = sorted(os.listdir(feat_dir))
    feat_files = [f for f in feat_files if f.endswith(".pt")]
    print(f"Found {len(feat_files)} complex feature files")

    all_entries = []
    idx_to_cid = []
    skipped = 0
    for fname in tqdm(feat_files, desc="Converting"):
        feat = torch.load(os.path.join(feat_dir, fname), map_location="cpu",
                          weights_only=False)
        entry = convert_complex(feat)
        if entry is None:
            skipped += 1
            continue
        all_entries.append(entry)
        idx_to_cid.append(entry["complex_id"])

    print(f"Converted {len(all_entries)} complexes ({skipped} skipped)")

    # Save mappings
    cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}
    with open(os.path.join(output_dir, "idx_to_cid.json"), "w") as f:
        json.dump(idx_to_cid, f, indent=2)
    with open(os.path.join(output_dir, "complex_ids.json"), "w") as f:
        json.dump(cid_to_idx, f, indent=2)
    print(f"Saved mappings: {len(idx_to_cid)} entries")

    # Generate per-split JSONL files
    splits_to_process = [args.split] if args.split else ["epitope_group", "antigen_fold", "temporal"]
    for split_name in splits_to_process:
        split_ids = load_split_ids(split_name, data_root)
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)

        for subset in ("train", "val", "test"):
            cid_set = set(split_ids[subset])
            out_path = os.path.join(split_dir, f"{subset}.jsonl")
            n = write_split_jsonl(all_entries, cid_set, out_path)
            print(f"  {split_name}/{subset}: {n} / {len(cid_set)} complexes")

    print("AbDockGen preprocessing complete.")


if __name__ == "__main__":
    main()
