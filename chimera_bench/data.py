"""Data loading API for CHIMERA-Bench.

Resolve the dataset location from the ``CHIMERA_DATA_ROOT`` environment
variable, or pass ``data_root`` explicitly to any function.

Expected on-disk layout (as distributed on HuggingFace/Zenodo)::

    chimera-bench-v1.0/
        metadata/final_summary.csv
        splits/{epitope_group,antigen_fold,temporal}.json
        complex_features/{complex_id}.pt

Quick start::

    import chimera_bench as cb

    split = cb.load_split("epitope_group")          # {"train", "val", "test"}
    feat = cb.load_complex(split["test"][0])        # dict of tensors
    ds = cb.ChimeraDataset("epitope_group", "test") # torch Dataset
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

SPLITS = ("epitope_group", "antigen_fold", "temporal")
SUBSETS = ("train", "val", "test")

CDR_LABEL_TO_CODE = {"H1": 0, "H2": 1, "H3": 2, "L1": 3, "L2": 4, "L3": 5}


def get_data_root(data_root=None) -> Path:
    """Resolve the dataset root from an argument or CHIMERA_DATA_ROOT.

    Raises a clear error if neither is set, since every loader needs it.
    """
    if data_root is not None:
        return Path(data_root)
    env = os.environ.get("CHIMERA_DATA_ROOT")
    if env:
        return Path(env)
    raise RuntimeError(
        "No data root set. Either pass data_root=... or set the "
        "CHIMERA_DATA_ROOT environment variable to your dataset directory. "
        "Download instructions: https://huggingface.co/datasets/mansoorbaloch/chimera-bench"
    )


def _features_dir(root: Path) -> Path:
    """complex_features lives at the root in the distributed layout."""
    return root / "complex_features"


def _metadata_path(root: Path) -> Path:
    """final_summary.csv lives under metadata/ when distributed, else at root."""
    packaged = root / "metadata" / "final_summary.csv"
    return packaged if packaged.exists() else root / "final_summary.csv"


def load_metadata(data_root=None) -> pd.DataFrame:
    """Load the complex-level metadata table (one row per complex)."""
    root = get_data_root(data_root)
    return pd.read_csv(_metadata_path(root))


def load_split(name: str, data_root=None) -> dict:
    """Load a split file as ``{"train": [...], "val": [...], "test": [...]}``."""
    if name not in SPLITS:
        raise ValueError(f"Unknown split '{name}'. Choose from {SPLITS}.")
    root = get_data_root(data_root)
    with open(root / "splits" / f"{name}.json") as f:
        return json.load(f)


def list_complexes(split=None, subset=None, data_root=None) -> list[str]:
    """List complex IDs, optionally restricted to a split and subset.

    Args:
        split: one of SPLITS, or None for all complexes on disk.
        subset: one of SUBSETS, requires split. None returns the whole split.
    """
    root = get_data_root(data_root)
    if split is None:
        return sorted(p.stem for p in _features_dir(root).glob("*.pt"))
    sp = load_split(split, data_root=root)
    if subset is None:
        return [cid for s in SUBSETS for cid in sp.get(s, [])]
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset '{subset}'. Choose from {SUBSETS}.")
    return list(sp.get(subset, []))


def load_complex(complex_id: str, data_root=None) -> dict:
    """Load the per-complex feature dict for one complex ID."""
    root = get_data_root(data_root)
    path = _features_dir(root) / f"{complex_id}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No feature file for '{complex_id}' at {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


def cdr_indices(feat: dict, cdr_type: str, scheme: str = "imgt"):
    """Return residue indices of one CDR in the concatenated heavy+light chain.

    Heavy CDRs (H1-H3) index into the heavy chain; light CDRs (L1-L3) are
    offset by the heavy-chain length so they index the concatenated tensor.
    """
    import numpy as np
    code = CDR_LABEL_TO_CODE.get(cdr_type)
    if code is None:
        raise ValueError(f"Unknown cdr_type '{cdr_type}'.")
    masks = feat["cdr_masks"][scheme]
    h_len = len(feat["heavy_ca_coords"])
    if code <= 2:
        return np.where(np.array(masks["heavy"]) == code)[0]
    return np.where(np.array(masks["light"]) == code)[0] + h_len


class ChimeraDataset(Dataset):
    """PyTorch Dataset over CHIMERA-Bench complexes.

    Args:
        split: one of SPLITS.
        subset: one of SUBSETS (default "train").
        data_root: dataset root, or None to use CHIMERA_DATA_ROOT.
        transform: optional callable applied to each loaded feature dict.
    """

    def __init__(self, split: str, subset: str = "train",
                 data_root=None, transform=None):
        self.root = get_data_root(data_root)
        self.ids = list_complexes(split, subset, data_root=self.root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        feat = load_complex(self.ids[idx], data_root=self.root)
        return self.transform(feat) if self.transform else feat
