"""Data contamination audit for CHIMERA-Bench.

Checks overlap between CHIMERA-Bench test sets and training data of:
  - ESM-2 (UniRef50/UniRef90, up to ~2021)
  - AntiBERTy (OAS, up to ~2021)
  - AlphaFold2 (PDB up to 2018-04-30)

Reports contamination rates per split type.

Usage:
    python -m evaluation.contamination \
        --split epitope_group --data-root /path/to/chimera-bench-v1.0
"""

import argparse
import json
import logging
import pickle
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import Config

log = logging.getLogger(__name__)

# Training data cutoff dates (approximate)
CUTOFFS = {
    "esm2": {
        "description": "ESM-2 trained on UniRef50/90 (Jan 2022 release)",
        "date_cutoff": "2021-09-01",
        "source": "UniRef50/90",
    },
    "antiberty": {
        "description": "AntiBERTy trained on OAS (unpaired, ~558M sequences)",
        "date_cutoff": "2021-06-01",
        "source": "OAS (Observed Antibody Space)",
    },
    "alphafold2": {
        "description": "AlphaFold2 trained on PDB (up to 2018-04-30)",
        "date_cutoff": "2018-04-30",
        "source": "PDB structures",
    },
}


def audit_temporal_overlap(df: pd.DataFrame, split: dict,
                           date_col: str = "date") -> dict:
    """Check temporal overlap between test set and model training data.

    For each model, counts how many test set structures were deposited
    before the model's training data cutoff date.

    Args:
        df: summary DataFrame with PDB deposition dates
        split: dict with train/val/test complex_id lists
        date_col: column name for deposition date

    Returns:
        dict with contamination stats per model
    """
    test_ids = set(split.get("test", []))
    train_ids = set(split.get("train", []))

    # Map complex_id -> PDB code -> date
    id_to_pdb = {}
    for _, row in df.iterrows():
        cid = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
        id_to_pdb[cid] = row["pdb"]

    # Parse dates
    pdb_dates = {}
    if date_col in df.columns:
        for _, row in df.iterrows():
            try:
                d = pd.to_datetime(row[date_col])
                pdb_dates[row["pdb"]] = d
            except (ValueError, TypeError):
                pass

    results = {}
    for model_name, info in CUTOFFS.items():
        cutoff = pd.to_datetime(info["date_cutoff"])

        test_before = 0
        test_total = 0
        train_before = 0
        train_total = 0

        for cid in test_ids:
            pdb = id_to_pdb.get(cid, cid.split("_")[0])
            if pdb in pdb_dates:
                test_total += 1
                if pdb_dates[pdb] <= cutoff:
                    test_before += 1

        for cid in train_ids:
            pdb = id_to_pdb.get(cid, cid.split("_")[0])
            if pdb in pdb_dates:
                train_total += 1
                if pdb_dates[pdb] <= cutoff:
                    train_before += 1

        results[model_name] = {
            "description": info["description"],
            "cutoff_date": info["date_cutoff"],
            "test_before_cutoff": test_before,
            "test_total_with_date": test_total,
            "test_contamination_rate": test_before / test_total if test_total > 0 else 0.0,
            "train_before_cutoff": train_before,
            "train_total_with_date": train_total,
        }

    return results


def audit_pdb_overlap(split: dict) -> dict:
    """Check if any PDB IDs appear in both train and test sets.

    This should never happen with proper splits but is a sanity check.
    """
    train_pdbs = {cid.split("_")[0] for cid in split.get("train", [])}
    val_pdbs = {cid.split("_")[0] for cid in split.get("val", [])}
    test_pdbs = {cid.split("_")[0] for cid in split.get("test", [])}

    train_test_overlap = train_pdbs & test_pdbs
    train_val_overlap = train_pdbs & val_pdbs
    val_test_overlap = val_pdbs & test_pdbs

    return {
        "train_test_pdb_overlap": len(train_test_overlap),
        "train_val_pdb_overlap": len(train_val_overlap),
        "val_test_pdb_overlap": len(val_test_overlap),
        "train_test_overlap_pdbs": sorted(train_test_overlap)[:20],
        "train_val_overlap_pdbs": sorted(train_val_overlap)[:20],
        "val_test_overlap_pdbs": sorted(val_test_overlap)[:20],
    }


def audit_sequence_overlap(annotations: list, split: dict,
                           identity_threshold: float = 0.9) -> dict:
    """Check CDR-H3 sequence overlap between train and test sets.

    Counts test sequences that have a near-identical match (>= threshold)
    in the training set using simple identity computation.
    """
    ann_map = {a.complex_id: a for a in annotations}
    train_ids = set(split.get("train", []))
    test_ids = set(split.get("test", []))

    # Extract CDR-H3 sequences from annotations
    def _get_h3_seq(cid):
        ann = ann_map.get(cid)
        if ann is None:
            return None
        cdr_masks = ann.cdr_masks.get("imgt", {})
        heavy_mask = cdr_masks.get("heavy", [])
        heavy_seq = ann.sequences.get("heavy", "")
        if not heavy_seq or not heavy_mask:
            return None
        # CDR-H3 = label 2 in annotation
        h3_residues = [heavy_seq[i] for i, m in enumerate(heavy_mask) if m == 2
                       and i < len(heavy_seq)]
        return "".join(h3_residues) if h3_residues else None

    train_h3 = []
    for cid in train_ids:
        seq = _get_h3_seq(cid)
        if seq:
            train_h3.append(seq)

    test_h3_seqs = []
    for cid in test_ids:
        seq = _get_h3_seq(cid)
        if seq:
            test_h3_seqs.append((cid, seq))

    # Check each test H3 against all training H3s
    n_contaminated = 0
    contaminated_ids = []

    for cid, test_seq in test_h3_seqs:
        for train_seq in train_h3:
            if len(test_seq) != len(train_seq):
                continue
            matches = sum(a == b for a, b in zip(test_seq, train_seq))
            identity = matches / len(test_seq)
            if identity >= identity_threshold:
                n_contaminated += 1
                contaminated_ids.append(cid)
                break

    return {
        "test_h3_total": len(test_h3_seqs),
        "train_h3_total": len(train_h3),
        "test_h3_contaminated": n_contaminated,
        "contamination_rate": n_contaminated / len(test_h3_seqs) if test_h3_seqs else 0.0,
        "identity_threshold": identity_threshold,
        "contaminated_ids": contaminated_ids[:20],
    }


def run_audit(cfg: Config, split_name: str) -> dict:
    """Run full contamination audit for a given split."""
    # Load data
    df = pd.read_csv(cfg.processed_dir / "final_summary.csv")

    split_path = cfg.splits_dir / f"{split_name}.json"
    with open(split_path) as f:
        split = json.load(f)

    ann_path = cfg.processed_dir / "annotations.pkl"
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)

    log.info("Running contamination audit for split=%s", split_name)
    log.info("  Train: %d, Val: %d, Test: %d",
             len(split.get("train", [])), len(split.get("val", [])),
             len(split.get("test", [])))

    results = {
        "split": split_name,
        "temporal_overlap": audit_temporal_overlap(df, split),
        "pdb_overlap": audit_pdb_overlap(split),
        "sequence_overlap": audit_sequence_overlap(annotations, split),
    }

    # Print summary
    log.info("=== Temporal Overlap ===")
    for model, info in results["temporal_overlap"].items():
        log.info("  %s: %d/%d test structures before cutoff (%.1f%%)",
                 model, info["test_before_cutoff"],
                 info["test_total_with_date"],
                 info["test_contamination_rate"] * 100)

    log.info("=== PDB Overlap ===")
    pdb = results["pdb_overlap"]
    log.info("  Train-Test PDB overlap: %d", pdb["train_test_pdb_overlap"])
    log.info("  Train-Val PDB overlap: %d", pdb["train_val_pdb_overlap"])

    log.info("=== CDR-H3 Sequence Overlap ===")
    seq = results["sequence_overlap"]
    log.info("  %d/%d test H3 sequences have >= %.0f%% identity match in train (%.1f%%)",
             seq["test_h3_contaminated"], seq["test_h3_total"],
             seq["identity_threshold"] * 100,
             seq["contamination_rate"] * 100)

    return results


def main():
    parser = argparse.ArgumentParser(description="CHIMERA-Bench contamination audit")
    parser.add_argument("--split", default="all",
                        help="Split name or 'all' for all splits")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root

    splits = ["epitope_group", "antigen_fold", "temporal"] if args.split == "all" else [args.split]

    all_results = {}
    for split_name in splits:
        all_results[split_name] = run_audit(cfg, split_name)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(all_results, f, indent=2, default=str)
        log.info("Saved audit results to %s", args.output)


if __name__ == "__main__":
    main()
