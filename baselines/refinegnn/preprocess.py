"""Preprocess CHIMERA dataset into RefineGNN's native JSONL format.

Converts complex_features/*.pt to JSONL with structure:
    {pdb, complex_id, seq, cdr, coords: {N, CA, C, O}}

RefineGNN uses heavy+light concatenated sequences with Chothia CDR labels
on the heavy chain only. Backbone coords (N, CA, C, O) are extracted from
atom14_coords[:, 0:4, :].

Output structure (in trans_baselines/refinegnn/):
    {split_name}/train.jsonl
    {split_name}/val.jsonl
    {split_name}/test.jsonl
    idx_to_cid.json       -- [complex_id, ...] ordered
    complex_ids.json       -- {complex_id: index}

Usage:
    cd baselines/refinegnn
    python preprocess.py
    python preprocess.py --splits epitope_group antigen_fold temporal
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import load_shared_config, load_split_ids


# CDR mask codes (from CHIMERA complex_features):
#   -1 = framework, 0 = H1, 1 = H2, 2 = H3
# RefineGNN CDR label encoding:
#   '0' = framework, '1' = H1, '2' = H2, '3' = H3
_CDR_CODE_TO_LABEL = {-1: "0", 0: "1", 1: "2", 2: "3"}


def ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def convert_complex(feat):
    """Convert a CHIMERA complex_features dict to RefineGNN JSONL entry.

    Returns dict or None if conversion fails.
    """
    cid = feat["complex_id"]

    # Heavy chain
    h_seq = feat["heavy_sequence"]
    h_a14 = ensure_numpy(feat["heavy_atom14_coords"])  # (N_h, 14, 3)
    h_backbone = h_a14[:, :4, :]  # (N_h, 4, 3) -- N, CA, C, O

    # Light chain
    l_seq = feat["light_sequence"]
    l_a14 = ensure_numpy(feat["light_atom14_coords"])  # (N_l, 14, 3)
    l_backbone = l_a14[:, :4, :]  # (N_l, 4, 3)

    # Concatenate heavy + light
    seq = h_seq + l_seq
    backbone = np.concatenate([h_backbone, l_backbone], axis=0)  # (L, 4, 3)

    # Build CDR label string from Chothia heavy chain masks
    chothia_heavy = feat["cdr_masks"]["chothia"]["heavy"]
    h_labels = [_CDR_CODE_TO_LABEL.get(c, "0") for c in chothia_heavy]
    l_labels = ["0"] * len(l_seq)

    # Pad or trim if numbering length != sequence length
    if len(h_labels) < len(h_seq):
        h_labels.extend(["0"] * (len(h_seq) - len(h_labels)))
    elif len(h_labels) > len(h_seq):
        h_labels = h_labels[:len(h_seq)]

    cdr = "".join(h_labels + l_labels)

    # Coords dict: separate arrays per atom type
    coords = {
        "N": backbone[:, 0, :].tolist(),
        "CA": backbone[:, 1, :].tolist(),
        "C": backbone[:, 2, :].tolist(),
        "O": backbone[:, 3, :].tolist(),
    }

    pdb = cid.rsplit("_", 3)[0] if "_" in cid else cid

    return {
        "pdb": pdb,
        "complex_id": cid,
        "seq": seq,
        "cdr": cdr,
        "coords": coords,
    }


def write_jsonl(entries, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    print(f"  Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CHIMERA for RefineGNN")
    parser.add_argument("--splits", nargs="+",
                        default=["epitope_group", "antigen_fold", "temporal"])
    args = parser.parse_args()

    shared = load_shared_config()
    data_root = shared["paths"]["data_root"]
    output_dir = os.path.join(shared["paths"]["trans_baselines"], "refinegnn")
    feat_dir = os.path.join(data_root, "processed", "complex_features")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert all complexes
    print("Converting all CHIMERA complexes to RefineGNN format...")
    all_entries = {}
    feat_files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".pt")])
    skipped = 0

    for fname in tqdm(feat_files, desc="Converting"):
        feat = torch.load(os.path.join(feat_dir, fname),
                          map_location="cpu", weights_only=False)
        cid = feat["complex_id"]

        # Skip if no Chothia heavy CDR masks
        if "cdr_masks" not in feat or "chothia" not in feat["cdr_masks"]:
            skipped += 1
            continue

        entry = convert_complex(feat)
        if entry is not None:
            all_entries[cid] = entry

    print(f"Converted {len(all_entries)} complexes ({skipped} skipped)")

    # Step 2: Save idx_to_cid and complex_ids mappings
    idx_to_cid = sorted(all_entries.keys())
    cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}

    idx_to_cid_path = os.path.join(output_dir, "idx_to_cid.json")
    with open(idx_to_cid_path, "w") as f:
        json.dump(idx_to_cid, f, indent=2)
    print(f"Saved idx_to_cid ({len(idx_to_cid)} entries) to {idx_to_cid_path}")

    complex_ids_path = os.path.join(output_dir, "complex_ids.json")
    with open(complex_ids_path, "w") as f:
        json.dump(cid_to_idx, f, indent=2)
    print(f"Saved complex_ids ({len(cid_to_idx)} entries) to {complex_ids_path}")

    # Step 3: Generate split-specific JSONL files
    for split_name in args.splits:
        print(f"\nGenerating {split_name} split JSONL files...")
        split_ids = load_split_ids(split_name, data_root)
        split_dir = os.path.join(output_dir, split_name)

        for subset in ("train", "val", "test"):
            entries = [all_entries[cid] for cid in split_ids[subset]
                       if cid in all_entries]
            write_jsonl(entries, os.path.join(split_dir, f"{subset}.jsonl"))

    print("\nRefineGNN preprocessing complete.")


if __name__ == "__main__":
    main()
