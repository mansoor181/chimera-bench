"""Step 8: Generate train/val/test splits along three axes."""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from config import Config

log = logging.getLogger(__name__)


def _random_split(ids: list[str], cfg: Config,
                  seed: int = 42) -> dict[str, list[str]]:
    """Helper: randomly split a list of IDs into train/val/test."""
    rng = np.random.RandomState(seed)
    ids = list(ids)
    rng.shuffle(ids)
    n = len(ids)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    return {
        "train": ids[:n_train],
        "val": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def _cluster_split(ids: list[str], cluster_labels: dict[str, int],
                   cfg: Config, seed: int = 42) -> dict[str, list[str]]:
    """Split by clusters: no cluster spans train and test.

    Assigns whole clusters greedily to the split furthest below its
    target count, so large clusters don't cause one bucket to overshoot.
    """
    rng = np.random.RandomState(seed)

    # Group IDs by cluster
    clusters = {}
    for cid in ids:
        label = cluster_labels.get(cid, -1)
        clusters.setdefault(label, []).append(cid)

    # Sort clusters largest-first for better packing, then shuffle ties
    cluster_keys = list(clusters.keys())
    rng.shuffle(cluster_keys)
    cluster_keys.sort(key=lambda k: len(clusters[k]), reverse=True)

    n_total = len(ids)
    targets = {
        "train": n_total * cfg.train_ratio,
        "val": n_total * cfg.val_ratio,
        "test": n_total * cfg.test_ratio,
    }
    split = {"train": [], "val": [], "test": []}
    counts = {"train": 0, "val": 0, "test": 0}

    for key in cluster_keys:
        members = clusters[key]
        # Pick the bucket with the largest remaining deficit
        best = max(targets, key=lambda b: targets[b] - counts[b])
        split[best].extend(members)
        counts[best] += len(members)

    return split


def epitope_group_split(df: pd.DataFrame, annotations: list,
                        cfg: Config) -> dict[str, list[str]]:
    """Split by epitope structural similarity clusters.

    Clusters epitopes so test set contains entirely unseen epitope patterns.
    Uses a hash of sorted epitope residue IDs as a proxy for structural grouping.
    For full structural clustering, use TM-align on epitope patches (expensive).
    """
    # Build epitope fingerprint per complex
    cluster_labels = {}
    for ann in annotations:
        # Use sorted epitope residue positions as fingerprint
        epi_key = tuple(sorted((c, r) for c, r, _ in ann.epitope_residues))
        cluster_labels[ann.complex_id] = hash(epi_key) % 10000

    ids = [ann.complex_id for ann in annotations]
    split = _cluster_split(ids, cluster_labels, cfg, seed=42)
    log.info("Epitope-group split: train=%d, val=%d, test=%d",
             len(split["train"]), len(split["val"]), len(split["test"]))
    return split


def antigen_fold_split(df: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Split by antigen identity (proxy for fold).

    Groups complexes by antigen sequence, so test has unseen antigens.
    Full CATH classification requires external lookup.
    """
    # Group by antigen -- use antigen_name or antigen_chain + pdb combo
    cluster_labels = {}
    ag_groups = df.groupby("antigen_name").ngroups if "antigen_name" in df.columns else 0

    if "antigen_name" in df.columns:
        name_to_id = {}
        for _, row in df.iterrows():
            cid = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
            name = row["antigen_name"]
            if name not in name_to_id:
                name_to_id[name] = len(name_to_id)
            cluster_labels[cid] = name_to_id[name]
    else:
        # Fallback: cluster by PDB antigen chain
        for _, row in df.iterrows():
            cid = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
            cluster_labels[cid] = hash(row.get("compound", row["pdb"])) % 5000

    ids = list(cluster_labels.keys())
    split = _cluster_split(ids, cluster_labels, cfg, seed=43)
    log.info("Antigen-fold split: train=%d, val=%d, test=%d",
             len(split["train"]), len(split["val"]), len(split["test"]))
    return split


def temporal_split(df: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Split by deposition date, using quantile cutoffs to hit target ratios.

    Sorts all complexes by date, then picks the date thresholds that give
    approximately train_ratio / val_ratio / test_ratio while keeping
    temporal ordering (train = oldest, test = newest).
    """
    df = df.copy()
    df["date_parsed"] = pd.to_datetime(df["date"], format="%m/%d/%y", errors="coerce")
    mask = df["date_parsed"].isna()
    df.loc[mask, "date_parsed"] = pd.to_datetime(
        df.loc[mask, "date"], format="mixed", errors="coerce"
    )
    df["complex_id"] = (
        df["pdb"] + "_" + df["Hchain"] + "_" + df["Lchain"] + "_" + df["antigen_chain"]
    )

    # Rows with no parseable date go to train
    has_date = df["date_parsed"].notna()
    no_date_ids = df.loc[~has_date, "complex_id"].tolist()

    dated = df.loc[has_date].sort_values("date_parsed").reset_index(drop=True)
    n = len(dated)

    # Compute split points on the dated subset (adjust for no-date rows in train)
    n_total = len(df)
    n_train_target = int(n_total * cfg.train_ratio) - len(no_date_ids)
    n_train_target = max(n_train_target, 0)
    n_val_target = int(n_total * cfg.val_ratio)

    train_cutoff = dated.iloc[min(n_train_target, n - 1)]["date_parsed"]
    val_end = min(n_train_target + n_val_target, n - 1)
    val_cutoff = dated.iloc[val_end]["date_parsed"]

    split = {"train": list(no_date_ids), "val": [], "test": []}
    for _, row in dated.iterrows():
        cid = row["complex_id"]
        dt = row["date_parsed"]
        if dt <= train_cutoff and len(split["train"]) < int(n_total * cfg.train_ratio):
            split["train"].append(cid)
        elif dt <= val_cutoff and len(split["val"]) < int(n_total * cfg.val_ratio):
            split["val"].append(cid)
        else:
            split["test"].append(cid)

    log.info("Temporal split: train=%d, val=%d, test=%d "
             "(cutoffs: train<=%s, val<=%s)",
             len(split["train"]), len(split["val"]), len(split["test"]),
             train_cutoff.strftime("%Y-%m-%d"), val_cutoff.strftime("%Y-%m-%d"))
    return split


def save_splits(splits: dict[str, dict], cfg: Config):
    """Save all split definitions to JSON files."""
    cfg.splits_dir.mkdir(parents=True, exist_ok=True)
    for name, split in splits.items():
        path = cfg.splits_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(split, f, indent=2)
        log.info("Saved %s split to %s", name, path)


def generate_splits(df: pd.DataFrame, annotations: list,
                    cfg: Config) -> dict[str, dict]:
    """Generate all three split types and save them."""
    # Filter annotations to only include complexes present in df
    valid_cids = set(
        f"{r['pdb']}_{r['Hchain']}_{r['Lchain']}_{r['antigen_chain']}"
        for _, r in df.iterrows()
    )
    filtered_anns = [a for a in annotations if a.complex_id in valid_cids]

    splits = {
        "epitope_group": epitope_group_split(df, filtered_anns, cfg),
        "antigen_fold": antigen_fold_split(df, cfg),
        "temporal": temporal_split(df, cfg),
    }
    save_splits(splits, cfg)
    return splits
