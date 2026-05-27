"""Step 8: Generate train/val/test splits along three axes."""

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from benchmark.config import Config

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


def _kabsch_rmsd(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Compute RMSD after optimal superposition (Kabsch algorithm).

    Both inputs should be (N, 3) arrays centered at origin.
    Returns inf if alignment fails (e.g., degenerate coordinates).
    """
    if len(coords1) != len(coords2) or len(coords1) < 3:
        return float("inf")

    # Center coordinates
    c1 = coords1 - coords1.mean(axis=0)
    c2 = coords2 - coords2.mean(axis=0)

    # Compute covariance matrix
    H = c1.T @ c2
    U, S, Vt = np.linalg.svd(H)

    # Handle reflection
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T

    # Rotate and compute RMSD
    c1_rot = c1 @ R
    rmsd = np.sqrt(np.mean(np.sum((c1_rot - c2) ** 2, axis=1)))
    return rmsd


def _build_resid_to_index_map(pdb_path: Path, chain_id: str) -> dict[int, int]:
    """Build mapping from PDB residue ID to CA array index."""
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]

    if chain_id not in model:
        return {}

    resid_to_idx = {}
    idx = 0
    for res in model[chain_id].get_residues():
        if res.id[0] != " ":  # Skip hetero residues
            continue
        if "CA" in res:
            resid_to_idx[res.id[1]] = idx
            idx += 1

    return resid_to_idx


def _extract_epitope_coords(complex_id: str, cfg: Config) -> np.ndarray | None:
    """Load epitope CA coordinates from complex_features."""
    import torch

    feat_path = cfg.processed_dir / "complex_features" / f"{complex_id}.pt"
    if not feat_path.exists():
        return None

    data = torch.load(feat_path, weights_only=False)
    epitope_residues = data.get("epitope_residues", [])
    ag_ca_coords = data.get("antigen_ca_coords")

    if not epitope_residues or ag_ca_coords is None:
        return None

    # Convert to numpy if tensor
    if hasattr(ag_ca_coords, "numpy"):
        ag_ca_coords = ag_ca_coords.numpy()

    # Build PDB resid -> array index mapping
    # Extract PDB ID and antigen chain from complex_id
    parts = complex_id.split("_")
    pdb_id = parts[0]
    ag_chain = parts[3] if len(parts) > 3 else parts[-1]

    pdb_path = cfg.structures_dir / f"{pdb_id}.pdb"
    if not pdb_path.exists():
        return None

    resid_to_idx = _build_resid_to_index_map(pdb_path, ag_chain)

    # Map epitope residues to array indices
    epitope_indices = []
    for chain, resid, resname in epitope_residues:
        idx = resid_to_idx.get(resid)
        if idx is not None and idx < len(ag_ca_coords):
            epitope_indices.append(idx)

    if len(epitope_indices) < 3:
        return None

    return ag_ca_coords[epitope_indices]


def _compute_epitope_distance_matrix(
    complex_ids: list[str],
    cfg: Config,
    max_workers: int = 8
) -> tuple[np.ndarray, list[str]]:
    """Compute pairwise RMSD matrix for epitope patches.

    Returns distance matrix and list of valid complex IDs.
    Uses multiprocessing for efficiency.
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed

    # Load all epitope coordinates
    log.info("Loading epitope coordinates for %d complexes...", len(complex_ids))
    epitope_coords = {}
    for cid in complex_ids:
        coords = _extract_epitope_coords(cid, cfg)
        if coords is not None and len(coords) >= 3:
            epitope_coords[cid] = coords

    valid_ids = list(epitope_coords.keys())
    n = len(valid_ids)
    log.info("Loaded %d valid epitope patches (skipped %d)", n, len(complex_ids) - n)

    if n == 0:
        return np.zeros((0, 0)), []

    # Compute pairwise RMSD matrix
    log.info("Computing pairwise RMSD matrix (%d x %d)...", n, n)
    dist_matrix = np.zeros((n, n))

    # Use condensed distance computation
    for i in range(n):
        for j in range(i + 1, n):
            coords_i = epitope_coords[valid_ids[i]]
            coords_j = epitope_coords[valid_ids[j]]

            # For different-sized epitopes, use the shorter one and align
            if len(coords_i) != len(coords_j):
                # Use geometric hash: center of mass + spread
                com_i = coords_i.mean(axis=0)
                com_j = coords_j.mean(axis=0)
                spread_i = np.std(np.linalg.norm(coords_i - com_i, axis=1))
                spread_j = np.std(np.linalg.norm(coords_j - com_j, axis=1))
                # Distance based on size mismatch + spread difference
                dist = abs(len(coords_i) - len(coords_j)) * 0.5 + abs(spread_i - spread_j)
            else:
                dist = _kabsch_rmsd(coords_i, coords_j)

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix, valid_ids


def _cluster_epitopes_by_structure(
    complex_ids: list[str],
    cfg: Config,
    rmsd_threshold: float = 3.0
) -> dict[str, int]:
    """Cluster epitopes by structural similarity using hierarchical clustering.

    Args:
        complex_ids: List of complex IDs to cluster
        cfg: Config with data paths
        rmsd_threshold: RMSD cutoff for clustering (Angstroms)

    Returns:
        dict mapping complex_id -> cluster_label
    """
    dist_matrix, valid_ids = _compute_epitope_distance_matrix(complex_ids, cfg)

    if len(valid_ids) == 0:
        log.warning("No valid epitope patches found, falling back to individual clusters")
        return {cid: i for i, cid in enumerate(complex_ids)}

    # Convert to condensed form for scipy
    condensed = squareform(dist_matrix)

    # Hierarchical clustering
    log.info("Performing hierarchical clustering with threshold %.1f A...", rmsd_threshold)
    linkage_matrix = linkage(condensed, method="average")
    cluster_labels = fcluster(linkage_matrix, t=rmsd_threshold, criterion="distance")

    # Map valid IDs to cluster labels
    cluster_map = {cid: int(label) for cid, label in zip(valid_ids, cluster_labels)}

    # Assign singletons to complexes that couldn't be clustered
    max_label = max(cluster_map.values()) if cluster_map else 0
    for cid in complex_ids:
        if cid not in cluster_map:
            max_label += 1
            cluster_map[cid] = max_label

    n_clusters = len(set(cluster_map.values()))
    log.info("Created %d epitope clusters from %d complexes", n_clusters, len(complex_ids))

    return cluster_map


def epitope_group_split(df: pd.DataFrame, annotations: list,
                        cfg: Config) -> dict[str, list[str]]:
    """Split by epitope structural similarity clusters.

    Clusters epitopes using Kabsch RMSD so test set contains structurally
    distinct epitope patterns.
    """
    ids = [ann.complex_id for ann in annotations]

    # Perform structural clustering
    cluster_labels = _cluster_epitopes_by_structure(ids, cfg, rmsd_threshold=3.0)

    split = _cluster_split(ids, cluster_labels, cfg, seed=42)
    log.info("Epitope-group split: train=%d, val=%d, test=%d",
             len(split["train"]), len(split["val"]), len(split["test"]))
    return split


def _extract_antigen_chains(df: pd.DataFrame, cfg: Config,
                            out_dir: Path) -> dict[str, str]:
    """Extract antigen chains from PDB files into individual PDB files.

    Returns dict mapping output filename (stem) -> complex_id.
    Deduplicates by PDB+chain so Foldseek only clusters unique chains.
    """
    from Bio.PDB import PDBParser, PDBIO, Select

    class ChainSelect(Select):
        def __init__(self, chain_id):
            self.chain_id = chain_id

        def accept_chain(self, chain):
            return chain.id == self.chain_id

    parser = PDBParser(QUIET=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map unique PDB+chain to complex IDs that share it
    pdb_chain_to_cids = {}
    for _, row in df.iterrows():
        cid = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
        key = f"{row['pdb']}_{row['antigen_chain']}"
        pdb_chain_to_cids.setdefault(key, []).append(cid)

    # Extract one PDB file per unique antigen chain
    filename_to_cids = {}
    extracted = 0
    for key, cids in pdb_chain_to_cids.items():
        pdb_id, ag_chain = key.rsplit("_", 1)
        pdb_path = cfg.structures_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue

        out_path = out_dir / f"{key}.pdb"
        if not out_path.exists():
            structure = parser.get_structure(pdb_id, pdb_path)
            model = structure[0]
            if ag_chain not in model:
                continue
            io = PDBIO()
            io.set_structure(model)
            io.save(str(out_path), ChainSelect(ag_chain))

        filename_to_cids[key] = cids
        extracted += 1

    log.info("Extracted %d unique antigen chains from %d complexes",
             extracted, len(df))
    return filename_to_cids


def _run_foldseek_cluster(pdb_dir: Path, cfg: Config) -> dict[str, int]:
    """Run Foldseek easy-cluster on antigen chain PDB files.

    Returns dict mapping PDB_CHAIN key -> cluster_id.
    """
    cache_path = cfg.raw_dir / "foldseek_clusters.tsv"
    if cache_path.exists():
        log.info("Using cached Foldseek clusters: %s", cache_path)
        cluster_map = {}
        cluster_name_to_id = {}
        with open(cache_path) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                if rep not in cluster_name_to_id:
                    cluster_name_to_id[rep] = len(cluster_name_to_id)
                cluster_map[member] = cluster_name_to_id[rep]
        log.info("Loaded %d chains in %d Foldseek clusters",
                 len(cluster_map), len(cluster_name_to_id))
        return cluster_map

    with tempfile.TemporaryDirectory() as tmpdir:
        out_prefix = Path(tmpdir) / "foldseek_out"
        tmp_dir = Path(tmpdir) / "tmp"
        tmp_dir.mkdir()

        cmd = [
            "foldseek", "easy-cluster",
            str(pdb_dir),
            str(out_prefix),
            str(tmp_dir),
            "--min-seq-id", "0.0",       # cluster by structure, not sequence
            "-c", "0.5",                  # 50% coverage threshold
            "--cov-mode", "0",            # bidirectional coverage
            "--cluster-mode", "0",        # greedy set cover
            "--tmscore-threshold", "0.5", # TM-score ≥ 0.5 = same fold
        ]

        log.info("Running Foldseek clustering on %s ...", pdb_dir)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise RuntimeError(f"Foldseek failed:\n{result.stderr}")

        # Parse cluster TSV (rep_name \t member_name)
        cluster_tsv = Path(f"{out_prefix}_cluster.tsv")
        cluster_map = {}
        cluster_name_to_id = {}

        with open(cluster_tsv) as f:
            for line in f:
                rep, member = line.strip().split("\t")
                if rep not in cluster_name_to_id:
                    cluster_name_to_id[rep] = len(cluster_name_to_id)
                cluster_map[member] = cluster_name_to_id[rep]

        # Cache results
        cfg.raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(cluster_tsv, cache_path)

    log.info("Foldseek: %d chains clustered into %d fold groups",
             len(cluster_map), len(cluster_name_to_id))
    return cluster_map


def antigen_fold_split(df: pd.DataFrame, cfg: Config) -> dict[str, list[str]]:
    """Split by structural fold of the antigen chain (Foldseek clustering).

    Extracts antigen chains, clusters them by structural similarity using
    Foldseek (TM-score ≥ 0.5), then splits so test set contains antigens
    from different structural folds than training. Achieves ~100% coverage
    unlike CATH which only covers ~30% of PDB chains.
    """
    # Extract antigen chains to temporary directory
    ag_chain_dir = cfg.raw_dir / "antigen_chains"
    filename_to_cids = _extract_antigen_chains(df, cfg, ag_chain_dir)

    # Run Foldseek clustering
    foldseek_clusters = _run_foldseek_cluster(ag_chain_dir, cfg)

    # Map complex IDs to cluster labels
    cluster_labels = {}
    mapped = 0
    unmapped = 0
    max_cluster_id = max(foldseek_clusters.values()) if foldseek_clusters else 0

    for pdb_chain_key, cids in filename_to_cids.items():
        cluster_id = foldseek_clusters.get(pdb_chain_key)
        if cluster_id is not None:
            for cid in cids:
                cluster_labels[cid] = cluster_id
                mapped += 1
        else:
            for cid in cids:
                max_cluster_id += 1
                cluster_labels[cid] = max_cluster_id
                unmapped += 1

    # Handle complexes not in filename_to_cids (missing PDB files)
    all_cids = set(
        f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
        for _, row in df.iterrows()
    )
    for cid in all_cids:
        if cid not in cluster_labels:
            max_cluster_id += 1
            cluster_labels[cid] = max_cluster_id
            unmapped += 1

    n_clusters = len(set(cluster_labels.values()))
    log.info("Mapped %d/%d complexes to %d Foldseek fold clusters (%d unmapped)",
             mapped, mapped + unmapped, n_clusters, unmapped)

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
