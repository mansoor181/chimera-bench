"""Shared utilities for CHIMERA baseline trainers and evaluators."""

import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.spatial import cKDTree

log = logging.getLogger(__name__)

_BASELINES_DIR = Path(__file__).resolve().parent
_SHARED_CONFIG_PATH = _BASELINES_DIR / "shared_config.yaml"


def _resolve_env_vars(value):
    """Resolve ${ENV_VAR} placeholders in a string."""
    if isinstance(value, str) and "${" in value:
        return os.path.expandvars(value)
    return value


def load_shared_config():
    """Load the shared CHIMERA config from baselines/shared_config.yaml.

    Path values containing ${CHIMERA_DATA_ROOT} are resolved from the
    environment variable. Set it before calling any baseline code:
        export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0
    """
    with open(_SHARED_CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    # Resolve environment variables in paths
    if "paths" in cfg:
        for key, val in cfg["paths"].items():
            cfg["paths"][key] = _resolve_env_vars(val)
    return cfg


def get_data_root():
    """Get the CHIMERA data root path from shared config or CHIMERA_DATA_ROOT env var."""
    env = os.environ.get("CHIMERA_DATA_ROOT")
    if env:
        return Path(env)
    cfg = load_shared_config()
    return Path(cfg["paths"]["data_root"])


def get_results_root():
    """Get the results root path from shared config."""
    cfg = load_shared_config()
    return Path(cfg["paths"]["results_root"])


def get_trans_baselines_root():
    """Get the trans_baselines root path from shared config."""
    cfg = load_shared_config()
    return Path(cfg["paths"]["trans_baselines"])


def generate_master_jsonl(output_path, data_root=None):
    """Generate a single JSONL with all CHIMERA complexes.

    Returns:
        dict mapping complex_id -> row dict from summary CSV.
    """
    root = Path(data_root) if data_root else get_data_root()
    summary_path = root / "processed" / "final_summary.csv"
    structures_dir = root / "raw" / "structures"

    cid_to_row = {}
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid_to_row[row["complex_id"]] = row

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    written = 0
    with open(output_path, "w") as fout:
        for cid, row in cid_to_row.items():
            pdb = row["pdb"]
            pdb_path = str(structures_dir / f"{pdb}.pdb")
            if not os.path.exists(pdb_path):
                continue
            ag_chains = row["antigen_chain"].split("|")
            entry = {
                "pdb": pdb,
                "heavy_chain": row["Hchain"],
                "light_chain": row["Lchain"],
                "antigen_chains": ag_chains,
                "pdb_data_path": pdb_path,
                "complex_id": cid,
            }
            fout.write(json.dumps(entry) + "\n")
            written += 1
    print(f"Wrote {written} entries to {output_path}")
    return cid_to_row


def generate_split_jsonl(split_name, output_dir, data_root=None):
    """Generate train/val/test JSONL files from CHIMERA split + summary.

    Returns:
        dict with keys 'train', 'val', 'test' (paths) and 'test_ids' (list)
    """
    root = Path(data_root) if data_root else get_data_root()
    summary_path = root / "processed" / "final_summary.csv"
    split_path = root / "splits" / f"{split_name}.json"
    structures_dir = root / "raw" / "structures"

    with open(split_path) as f:
        split = json.load(f)

    cid_to_row = {}
    with open(summary_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid_to_row[row["complex_id"]] = row

    os.makedirs(output_dir, exist_ok=True)
    paths = {}
    for subset in ("train", "val", "test"):
        ids = split[subset]
        out_path = os.path.join(output_dir, f"{subset}.jsonl")
        with open(out_path, "w") as fout:
            for cid in ids:
                row = cid_to_row.get(cid)
                if row is None:
                    continue
                pdb = row["pdb"]
                pdb_path = str(structures_dir / f"{pdb}.pdb")
                if not os.path.exists(pdb_path):
                    continue
                ag_chains = row["antigen_chain"].split("|")
                entry = {
                    "pdb": pdb,
                    "heavy_chain": row["Hchain"],
                    "light_chain": row["Lchain"],
                    "antigen_chains": ag_chains,
                    "pdb_data_path": pdb_path,
                    "complex_id": cid,
                }
                fout.write(json.dumps(entry) + "\n")
        paths[subset] = out_path

    paths["test_ids"] = split["test"]
    return paths


def load_split_ids(split_name, data_root=None):
    """Load train/val/test complex_id lists from a CHIMERA split file.

    Returns:
        dict with keys 'train', 'val', 'test', each a list of complex_ids.
    """
    root = Path(data_root) if data_root else get_data_root()
    split_path = root / "splits" / f"{split_name}.json"
    with open(split_path) as f:
        return json.load(f)


def copy_chimera_pdbs(output_dir, data_root=None):
    """Copy all CHIMERA PDB structures to output_dir."""
    root = Path(data_root) if data_root else get_data_root()
    structures_dir = root / "raw" / "structures"
    os.makedirs(output_dir, exist_ok=True)
    copied = 0
    for pdb_file in sorted(structures_dir.glob("*.pdb")):
        dst = os.path.join(output_dir, pdb_file.name)
        if not os.path.exists(dst):
            shutil.copy2(str(pdb_file), dst)
        copied += 1
    print(f"Copied {copied} PDB files to {output_dir}")
    return copied


class DatasetView(torch.utils.data.Dataset):
    """Wraps a dataset with a subset of indices, sharing underlying data."""

    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        # Copy attributes that trainers/collate_fn may access
        self.data = dataset.data
        self.mode = getattr(dataset, "mode", "111")
        self.cdr = getattr(dataset, "cdr", None)
        self.paratope = getattr(dataset, "paratope", None)
        # Build idx_mapping for inference (maps view idx -> original data idx)
        self.idx_mapping = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        # Temporarily set dataset's idx_mapping to identity so we can index directly
        old_mapping = self.dataset.idx_mapping
        old_num = self.dataset.num_entry
        self.dataset.idx_mapping = list(range(old_num))
        try:
            item = self.dataset[real_idx]
        finally:
            self.dataset.idx_mapping = old_mapping
            self.dataset.num_entry = old_num
        return item


def build_cid_to_idx(dataset):
    """Build complex_id -> dataset index mapping from a loaded dataset."""
    cid_to_idx = {}
    for i in range(dataset.num_entry):
        cplx = dataset.data[i]
        cid = cplx.get_id()
        cid_to_idx[cid] = i
    return cid_to_idx


def split_dataset(dataset, split_ids, cid_to_idx):
    """Create DatasetView instances for train/val/test from split IDs.

    Args:
        dataset: fully loaded dataset (all complexes)
        split_ids: dict with 'train', 'val', 'test' lists of complex_ids
        cid_to_idx: mapping from complex_id to dataset index

    Returns:
        dict with 'train', 'val', 'test' DatasetView instances
    """
    views = {}
    for subset in ("train", "val", "test"):
        indices = [cid_to_idx[cid] for cid in split_ids[subset]
                   if cid in cid_to_idx]
        views[subset] = DatasetView(dataset, indices)
        print(f"  {subset}: {len(indices)} / {len(split_ids[subset])} complexes")
    return views


class EarlyStopping:
    """Monitor a metric and signal when to stop."""

    def __init__(self, patience=10, mode="min"):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False

    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        if self.mode == "min":
            improved = value < self.best_value
        else:
            improved = value > self.best_value
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.patience:
            self.early_stop = True
        return self.early_stop


class ModelCheckpoint:
    """Save best and last checkpoints."""

    def __init__(self, save_dir, mode="min"):
        self.save_dir = Path(save_dir) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_value = None

    def save(self, model, optimizer, scheduler, epoch, val_metric):
        ckpt = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "epoch": epoch,
            "val_metric": val_metric,
        }
        torch.save(ckpt, self.save_dir / "last.pt")
        if self.best_value is None or \
           (self.mode == "min" and val_metric < self.best_value) or \
           (self.mode == "max" and val_metric > self.best_value):
            self.best_value = val_metric
            torch.save(ckpt, self.save_dir / "best.pt")
            return True
        return False

    def load_best(self, model, device):
        path = self.save_dir / "best.pt"
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        return ckpt


def setup_wandb(project, run_name, config, enabled=True):
    """Initialize wandb run, return run object or None."""
    if not enabled:
        return None
    import wandb
    run = wandb.init(project=project, name=run_name, config=config)
    return run


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_device(data, device):
    """Recursively move tensors to device."""
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(to_device(v, device) for v in data)
    if hasattr(data, "to"):
        return data.to(device)
    return data


def save_predictions(predictions, output_dir):
    """Save per-complex prediction dicts as .pt files.

    Compatible with benchmark/evaluation/evaluate.py CLI.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for pred in predictions:
        cid = pred["complex_id"]
        torch.save(pred, out / f"{cid}.pt")


# ---------------------------------------------------------------------------
# Full evaluation utilities (Tier 2)
# ---------------------------------------------------------------------------

# CDR code -> label mapping
_CDR_CODE_TO_LABEL = {0: "H1", 1: "H2", 2: "H3", 3: "L1", 4: "L2", 5: "L3"}
_CDR_LABEL_TO_CODE = {v: k for k, v in _CDR_CODE_TO_LABEL.items()}

FULL_METRIC_KEYS = [
    "aar", "caar", "ppl",
    "rmsd", "tm_score",
    "fnat", "irmsd", "dockq",
    "epitope_f1",
    "n_liabilities",
    "chimera_s", "chimera_b",
]


def _ensure_numpy(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.numpy()
    return np.asarray(x)


def compute_ca_contacts(ab_ca, ag_ca, cutoff=8.0):
    """Compute residue-level contacts from symmetric CA-CA distances.

    Args:
        ab_ca: (N_ab, 3) antibody CA coordinates
        ag_ca: (N_ag, 3) antigen CA coordinates
        cutoff: CA-CA distance threshold in Angstroms (default 8.0)

    Returns:
        epitope_mask: (N_ag,) bool
        paratope_mask: (N_ab,) bool
        contact_pairs: set of (ab_residue_idx, ag_residue_idx)
    """
    n_ab = len(ab_ca)
    n_ag = len(ag_ca)
    epitope_mask = np.zeros(n_ag, dtype=bool)
    paratope_mask = np.zeros(n_ab, dtype=bool)
    contact_pairs = set()

    if n_ab == 0 or n_ag == 0:
        return epitope_mask, paratope_mask, contact_pairs

    tree = cKDTree(ag_ca)
    for i, coord in enumerate(ab_ca):
        for j in tree.query_ball_point(coord, cutoff):
            paratope_mask[i] = True
            epitope_mask[j] = True
            contact_pairs.add((i, j))

    return epitope_mask, paratope_mask, contact_pairs


def load_native_complex(complex_id, feat_dir, cutoff=8.0):
    """Load a native complex from complex_features and compute contacts.

    Contacts are computed using symmetric CA-CA distances at the given cutoff.

    Returns:
        dict with keys: complex_id, true_sequence, ab_ca_coords, ag_ca_coords,
        epitope_mask, paratope_mask, contact_pairs, heavy_len, light_len
    """
    feat_dir = Path(feat_dir)
    feat = torch.load(feat_dir / f"{complex_id}.pt",
                      map_location="cpu", weights_only=False)

    h_ca = _ensure_numpy(feat["heavy_ca_coords"])
    l_ca = _ensure_numpy(feat["light_ca_coords"])
    ag_ca = _ensure_numpy(feat["antigen_ca_coords"])
    ab_ca = np.concatenate([h_ca, l_ca], axis=0)

    h_seq = feat["heavy_sequence"]
    l_seq = feat["light_sequence"]
    full_seq = h_seq + l_seq

    epitope_mask, paratope_mask, contact_pairs = compute_ca_contacts(
        ab_ca, ag_ca, cutoff=cutoff,
    )

    return {
        "complex_id": complex_id,
        "true_sequence": full_seq,
        "ab_ca_coords": ab_ca,
        "ag_ca_coords": ag_ca,
        "epitope_mask": epitope_mask,
        "paratope_mask": paratope_mask,
        "contact_pairs": contact_pairs,
        "heavy_len": len(h_ca),
        "light_len": len(l_ca),
        "feat": feat,
    }


def load_natives(split_name, data_root=None, split_type="test", cutoff=8.0):
    """Load native complexes for a split.

    Returns:
        dict {complex_id: native_dict}
    """
    root = Path(data_root) if data_root else get_data_root()
    feat_dir = root / "processed" / "complex_features"
    split_ids = load_split_ids(split_name, data_root)
    test_ids = split_ids.get(split_type, [])

    natives = {}
    for cid in test_ids:
        feat_path = feat_dir / f"{cid}.pt"
        if not feat_path.exists():
            continue
        try:
            natives[cid] = load_native_complex(cid, feat_dir, cutoff=cutoff)
        except Exception as e:
            log.warning("Failed to load native %s: %s", cid, e)
    log.info("Loaded %d / %d native complexes for %s/%s",
             len(natives), len(test_ids), split_name, split_type)
    return natives


def get_cdr_indices(feat, cdr_type, scheme="imgt"):
    """Get sequential indices of CDR residues in concatenated (heavy+light) CA array.

    Uses stored cdr_masks from complex_features. The mask arrays cover the
    ANARCI-numbered variable domain, which starts at position 0 in each chain.
    Light chain indices are offset by the actual heavy CA coord length.

    Args:
        feat: loaded .pt dict from complex_features
        cdr_type: "H1","H2","H3","L1","L2","L3","all"
        scheme: "imgt" or "chothia"

    Returns:
        numpy array of indices into the concatenated heavy+light CA array
    """
    masks = feat["cdr_masks"][scheme]
    h_mask = np.array(masks["heavy"])
    l_mask = np.array(masks["light"])
    h_ca_len = len(feat["heavy_ca_coords"])

    if cdr_type == "all":
        h_idx = np.where(h_mask >= 0)[0]
        l_idx = np.where(l_mask >= 0)[0] + h_ca_len
        return np.concatenate([h_idx, l_idx])

    code = _CDR_LABEL_TO_CODE.get(cdr_type)
    if code is None:
        raise ValueError(f"Unknown CDR type: {cdr_type}")
    if code <= 2:  # Heavy chain CDR (H1=0, H2=1, H3=2)
        return np.where(h_mask == code)[0]
    else:  # Light chain CDR (L1=3, L2=4, L3=5)
        return np.where(l_mask == code)[0] + h_ca_len


def construct_pred_ab_full_coords(pred_cdr_coords, cdr_indices, native_ab_coords):
    """Replace CDR positions in native ab coords with predicted coords.

    Args:
        pred_cdr_coords: (N_cdr, 3) predicted CDR CA coords
        cdr_indices: (N_cdr,) indices into native_ab_coords
        native_ab_coords: (N_ab, 3) native antibody CA coords

    Returns:
        (N_ab, 3) full antibody coords with CDR positions replaced
    """
    result = native_ab_coords.copy()
    result[cdr_indices] = pred_cdr_coords
    return result


def evaluate_prediction_full(pred_data, native, cdr_type, feat,
                             numbering_scheme=None):
    """Evaluate a single prediction with the full CHIMERA metric suite.

    Args:
        pred_data: dict from saved .pt prediction file
        native: dict from load_native_complex
        cdr_type: CDR type string (e.g., "H3", "all")
        feat: loaded complex_features dict
        numbering_scheme: explicit scheme ("imgt" or "chothia"), or None to auto-detect

    Returns:
        dict with all metric values
    """
    import sys
    baselines_dir = Path(__file__).resolve().parent
    repo_root = baselines_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from evaluation.metrics import (
        aar, caar, kabsch_rmsd, tm_score, fnat, interface_rmsd,
        dockq_score, epitope_metrics, compute_epitope_mask,
        count_liabilities, chimera_s, chimera_b,
    )

    result = {"complex_id": native["complex_id"], "cdr_type": cdr_type}

    ab_ca = native["ab_ca_coords"]
    ag_ca = native["ag_ca_coords"]

    pred_coords = _ensure_numpy(pred_data.get("pred_coords"))
    true_coords = _ensure_numpy(pred_data.get("true_coords"))
    pred_seq = pred_data.get("pred_sequence", "")
    true_seq = pred_data.get("true_sequence", "")

    # Get CDR indices using explicit scheme or auto-detect
    if numbering_scheme:
        cdr_indices = get_cdr_indices(feat, cdr_type, scheme=numbering_scheme)
    else:
        # Auto-detect: try IMGT first, fall back to Chothia if length doesn't match
        cdr_indices = get_cdr_indices(feat, cdr_type, scheme="imgt")
        if pred_coords is not None and len(cdr_indices) != len(pred_coords):
            cdr_indices_chothia = get_cdr_indices(feat, cdr_type, scheme="chothia")
            if len(cdr_indices_chothia) == len(pred_coords):
                cdr_indices = cdr_indices_chothia

    # Group 1: Sequence
    result["aar"] = aar(pred_seq, true_seq) if pred_seq and true_seq else 0.0

    paratope_full = native["paratope_mask"]
    if len(cdr_indices) > 0 and len(paratope_full) > 0:
        cdr_paratope = paratope_full[cdr_indices]
        result["caar"] = caar(pred_seq, true_seq, cdr_paratope) if pred_seq and true_seq else 0.0
    else:
        result["caar"] = 0.0

    result["ppl"] = float(pred_data.get("ppl", 0.0))
    result["n_liabilities"] = count_liabilities(pred_seq) if pred_seq else 0

    # Group 2: Structure (CDR-level)
    result["rmsd"] = 0.0
    result["tm_score"] = 0.0
    if pred_coords is not None and true_coords is not None and len(pred_coords) > 0:
        result["rmsd"] = kabsch_rmsd(pred_coords, true_coords)
        result["tm_score"] = tm_score(pred_coords, true_coords)

    # Full antibody coords: copy native, replace CDR positions with predictions
    # Kabsch-align pred_coords to true_coords to handle coordinate frame mismatch
    pred_ab_full = None
    if pred_coords is not None and len(cdr_indices) == len(pred_coords):
        # Align pred_coords to true_coords (from prediction file) if they're in different frames
        if true_coords is not None and len(true_coords) == len(pred_coords):
            # Check if alignment is needed (large coordinate difference suggests different frames)
            coord_diff = np.abs(pred_coords - true_coords).max()
            if coord_diff > 10.0:  # More than 10A diff suggests different coord frames
                # Kabsch-align pred to true
                centroid_p = pred_coords.mean(axis=0)
                centroid_t = true_coords.mean(axis=0)
                p_centered = pred_coords - centroid_p
                t_centered = true_coords - centroid_t
                H = p_centered.T @ t_centered
                U, S, Vt = np.linalg.svd(H)
                R = Vt.T @ U.T
                if np.linalg.det(R) < 0:
                    Vt[-1, :] *= -1
                    R = Vt.T @ U.T
                pred_coords = (pred_coords - centroid_p) @ R.T + centroid_t
        pred_ab_full = construct_pred_ab_full_coords(pred_coords, cdr_indices, ab_ca)

    # CDR-specific filtering: only count contacts where the antibody partner
    # is a CDR residue. Framework contacts are trivially preserved by
    # construct_pred_ab_full_coords and would dominate the metrics.
    cdr_set = set(cdr_indices.tolist()) if len(cdr_indices) > 0 else set()

    # Group 4: Epitope specificity (CDR-specific)
    result["epitope_precision"] = 0.0
    result["epitope_recall"] = 0.0
    result["epitope_f1"] = 0.0
    if pred_ab_full is not None and len(ag_ca) > 0 and len(cdr_indices) > 0:
        # CDR-specific: only antigen residues contacted by CDR residues
        true_epi_mask = compute_epitope_mask(ab_ca[cdr_indices], ag_ca, cutoff=8.0)
        epi = epitope_metrics(pred_ab_full[cdr_indices], ag_ca, true_epi_mask)
        result.update(epi)

    # Group 3: Interface (CDR-specific contacts)
    result["fnat"] = 0.0
    result["irmsd"] = float("inf")
    result["dockq"] = 0.0
    if pred_ab_full is not None and cdr_set:
        # Full CA-CA 8A contacts, then filter to CDR-specific
        all_true = native["contact_pairs"]
        _, _, all_pred = compute_ca_contacts(pred_ab_full, ag_ca, cutoff=8.0)

        true_cdr_contacts = {(i, j) for i, j in all_true if i in cdr_set}
        pred_cdr_contacts = {(i, j) for i, j in all_pred if i in cdr_set}

        if true_cdr_contacts:
            fnat_val = fnat(pred_cdr_contacts, true_cdr_contacts)
            result["fnat"] = fnat_val

            irmsd_val = interface_rmsd(
                pred_ab_full, ag_ca, ab_ca, ag_ca,
                list(true_cdr_contacts),
            )
            result["irmsd"] = irmsd_val

            lrmsd = kabsch_rmsd(pred_ab_full, ab_ca)
            result["dockq"] = dockq_score(fnat_val, irmsd_val, lrmsd)

    # Composites
    result["chimera_s"] = chimera_s(
        dockq=result["dockq"],
        rmsd=result["rmsd"],
        tmscore=result["tm_score"],
    )
    result["chimera_b"] = chimera_b(
        epitope_f1=result["epitope_f1"],
        fnat_val=result["fnat"],
    )

    return result


def run_full_evaluation(pred_dir, split_name, data_root=None, cdr_type_hint=None,
                        numbering_scheme=None):
    """Run full CHIMERA evaluation on a predictions directory.

    Args:
        pred_dir: path to directory with .pt prediction files
        split_name: split name (e.g., "epitope_group")
        data_root: override data root
        cdr_type_hint: CDR type if not stored in predictions

    Returns:
        (per_complex_list, summary_dict, cdr_type)
    """
    import sys
    baselines_dir = Path(__file__).resolve().parent
    repo_root = baselines_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from evaluation.metrics import mean_std

    pred_dir = Path(pred_dir)
    root = Path(data_root) if data_root else get_data_root()
    feat_dir = root / "processed" / "complex_features"

    # Load natives
    natives = load_natives(split_name, data_root, split_type="test")

    # Load predictions
    per_complex = []
    all_metrics = {k: [] for k in FULL_METRIC_KEYS}
    cdr_type = cdr_type_hint

    for f in sorted(pred_dir.glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=False)
        cid = data.get("complex_id", f.stem)
        if cdr_type is None:
            cdr_type = data.get("cdr_type", "unknown")

        if cid not in natives:
            log.warning("No native found for %s, skipping", cid)
            continue

        native = natives[cid]
        feat = native["feat"]

        result = evaluate_prediction_full(data, native, cdr_type, feat,
                                          numbering_scheme=numbering_scheme)
        per_complex.append(result)
        for k in FULL_METRIC_KEYS:
            if k in result and result[k] != float("inf"):
                all_metrics[k].append(result[k])

    # Summary with bootstrap CIs
    summary = {}
    for k in FULL_METRIC_KEYS:
        if all_metrics[k]:
            summary[k] = mean_std(all_metrics[k])

    cdr_type = cdr_type or "unknown"
    log.info("Evaluated %d complexes for %s", len(per_complex), cdr_type)
    return per_complex, summary, cdr_type
