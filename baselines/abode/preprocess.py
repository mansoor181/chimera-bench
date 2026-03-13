"""Preprocess CHIMERA dataset into AbODE conditional JSON format.

Converts complex_features/*.pt to AbODE's conditional format where each
CDR is extracted with +-1 flanking residues, split into ab_seq/coords_ab
(CDR+flanking), before_seq/coords_before (framework before CDR),
after_seq/coords_after (framework after CDR), and ag_seq/coords_ag (antigen).

Uses IMGT CDR definitions on heavy chain only (H1, H2, H3) so that
predictions align with CHIMERA's evaluation pipeline.
Backbone atoms: N (atom14 idx 0), CA (idx 1), C (idx 2) -> (L, 3, 3).

Output structure (in trans_baselines/abode/):
    {split_name}/cdrh{1,2,3}/{train,val,test}.json
    idx_to_cid.json
    complex_ids.json

Usage:
    cd baselines/abode
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


# IMGT CDR mask codes in complex_features:
#   -1 = framework, 0 = H1, 1 = H2, 2 = H3
CDR_TYPE_TO_CODE = {"1": 0, "2": 1, "3": 2}


def ensure_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def extract_backbone_ncc(atom14_coords, atom14_mask):
    """Extract N, CA, C backbone coords from atom14 format.

    Args:
        atom14_coords: (L, 14, 3) all-atom coordinates
        atom14_mask: (L, 14) bool mask

    Returns:
        (L, 3, 3) array -- 3 atoms (N, CA, C) x 3 coordinates
    """
    coords = ensure_numpy(atom14_coords)
    return coords[:, :3, :]  # (L, 3, 3) -- N=0, CA=1, C=2


def convert_complex_for_cdr(feat, cdr_type):
    """Convert a CHIMERA complex_features dict to AbODE conditional format for one CDR.

    AbODE conditional format extracts:
    - ab_seq: CDR + 1 flanking residue each side
    - coords_ab: backbone (N,CA,C) for ab_seq
    - before_seq / coords_before: heavy chain residues before the CDR region
    - after_seq / coords_after: heavy chain residues after the CDR region
    - ag_seq / coords_ag: antigen backbone

    Returns dict or None if conversion fails.
    """
    cid = feat["complex_id"]
    cdr_code = CDR_TYPE_TO_CODE[cdr_type]

    # Get IMGT heavy chain CDR mask
    if "cdr_masks" not in feat or "imgt" not in feat["cdr_masks"]:
        return None
    imgt_heavy = feat["cdr_masks"]["imgt"]["heavy"]

    # Find CDR residue indices
    cdr_indices = [i for i, c in enumerate(imgt_heavy) if c == cdr_code]
    if len(cdr_indices) == 0:
        return None

    cdr_start = cdr_indices[0]
    cdr_end = cdr_indices[-1]

    h_seq = feat["heavy_sequence"]
    h_len = len(h_seq)

    # Add +-1 flanking residues
    flank_start = max(0, cdr_start - 1)
    flank_end = min(h_len - 1, cdr_end + 1)

    # Extract heavy chain backbone (N, CA, C)
    h_backbone = extract_backbone_ncc(feat["heavy_atom14_coords"],
                                      feat["heavy_atom14_mask"])  # (H_len, 3, 3)

    # ab_seq: CDR + flanking
    ab_seq = h_seq[flank_start:flank_end + 1]
    coords_ab = h_backbone[flank_start:flank_end + 1]  # (L_cdr+flanking, 3, 3)

    if len(ab_seq) < 3:
        return None

    # before_seq: heavy chain residues before the flanked region
    before_seq = h_seq[:flank_start]
    coords_before = h_backbone[:flank_start] if flank_start > 0 else np.zeros((0, 3, 3))

    # after_seq: heavy chain residues after the flanked region
    after_seq = h_seq[flank_end + 1:]
    coords_after = h_backbone[flank_end + 1:] if flank_end + 1 < h_len else np.zeros((0, 3, 3))

    # Antigen
    ag_seq = feat["antigen_sequence"]
    ag_backbone = extract_backbone_ncc(feat["antigen_atom14_coords"],
                                       feat["antigen_atom14_mask"])

    if len(ag_seq) == 0:
        return None
    if "*" in ag_seq:
        return None

    return {
        "pdb": cid.rsplit("_", 3)[0] if "_" in cid else cid,
        "complex_id": cid,
        "ab_seq": ab_seq,
        "before_seq": before_seq,
        "after_seq": after_seq,
        "ag_seq": ag_seq,
        "coords_ab": coords_ab.tolist(),
        "coords_before": coords_before.tolist(),
        "coords_after": coords_after.tolist(),
        "coords_ag": ag_backbone.tolist(),
    }


def write_json(entries, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(entries, f)
    print(f"  Wrote {len(entries)} entries to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess CHIMERA for AbODE")
    parser.add_argument("--splits", nargs="+",
                        default=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--cdr_types", nargs="+", default=["1", "2", "3"],
                        choices=["1", "2", "3"])
    args = parser.parse_args()

    shared = load_shared_config()
    data_root = shared["paths"]["data_root"]
    output_dir = os.path.join(shared["paths"]["trans_baselines"], "abode")
    feat_dir = os.path.join(data_root, "processed", "complex_features")
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Convert all complexes for each CDR type
    print("Converting all CHIMERA complexes to AbODE conditional format...")
    feat_files = sorted([f for f in os.listdir(feat_dir) if f.endswith(".pt")])

    # {cdr_type: {complex_id: entry_dict}}
    all_entries = {ct: {} for ct in args.cdr_types}
    skipped = {ct: 0 for ct in args.cdr_types}

    for fname in tqdm(feat_files, desc="Converting"):
        feat = torch.load(os.path.join(feat_dir, fname),
                          map_location="cpu", weights_only=False)
        cid = feat["complex_id"]

        for ct in args.cdr_types:
            entry = convert_complex_for_cdr(feat, ct)
            if entry is not None:
                all_entries[ct][cid] = entry
            else:
                skipped[ct] += 1

    for ct in args.cdr_types:
        print(f"CDR H{ct}: converted {len(all_entries[ct])} complexes "
              f"({skipped[ct]} skipped)")

    # Step 2: Save idx_to_cid and complex_ids mappings (union of all CDR types)
    all_cids = set()
    for ct in args.cdr_types:
        all_cids.update(all_entries[ct].keys())
    idx_to_cid = sorted(all_cids)
    cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}

    idx_to_cid_path = os.path.join(output_dir, "idx_to_cid.json")
    with open(idx_to_cid_path, "w") as f:
        json.dump(idx_to_cid, f, indent=2)
    print(f"Saved idx_to_cid ({len(idx_to_cid)} entries) to {idx_to_cid_path}")

    complex_ids_path = os.path.join(output_dir, "complex_ids.json")
    with open(complex_ids_path, "w") as f:
        json.dump(cid_to_idx, f, indent=2)
    print(f"Saved complex_ids ({len(cid_to_idx)} entries) to {complex_ids_path}")

    # Step 3: Generate split-specific JSON files per CDR type
    for split_name in args.splits:
        split_ids = load_split_ids(split_name, data_root)

        for ct in args.cdr_types:
            print(f"\nGenerating {split_name}/cdrh{ct} JSON files...")
            split_dir = os.path.join(output_dir, split_name, f"cdrh{ct}")

            for subset in ("train", "val", "test"):
                entries = [all_entries[ct][cid] for cid in split_ids[subset]
                           if cid in all_entries[ct]]
                write_json(entries, os.path.join(split_dir, f"{subset}.json"))

    print("\nAbODE preprocessing complete.")


if __name__ == "__main__":
    main()
