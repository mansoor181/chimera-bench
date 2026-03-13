"""Central configuration for CHIMERA-Bench dataset construction."""

import os
from dataclasses import dataclass, field
from pathlib import Path


def _default_data_root() -> Path:
    """Resolve data root from CHIMERA_DATA_ROOT env var, or fall back to ./data."""
    return Path(os.environ.get("CHIMERA_DATA_ROOT", "./data"))


@dataclass
class Config:
    """All paths, URLs, and thresholds for the curation pipeline.

    Set the CHIMERA_DATA_ROOT environment variable to point to your local copy
    of the CHIMERA-Bench dataset, or pass --data-root on the CLI.
    """

    # -- Directories --
    data_root: Path = field(default_factory=_default_data_root)

    @property
    def logs_dir(self) -> Path:
        return self.data_root / "logs"

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def structures_dir(self) -> Path:
        return self.data_root / "raw" / "structures"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def splits_dir(self) -> Path:
        return self.data_root / "splits"

    # -- SAbDab --
    sabdab_summary_url: str = (
        "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/"
    )
    sabdab_pdb_url: str = (
        "https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/pdb/{}/?scheme=imgt"
    )

    # -- SKEMPI v2 --
    skempi_url: str = (
        "https://life.bsc.es/pid/skempi2/database/download/skempi_v2.csv"
    )

    # -- Quality filters --
    max_resolution: float = 4.0
    antigen_types: tuple = ("protein", "peptide")
    require_paired_chains: bool = True  # both VH + VL

    # -- Contacts --
    contact_cutoff: float = 4.5  # Angstrom

    # -- Deduplication --
    seq_identity_threshold: float = 0.95

    # -- Numbering --
    numbering_schemes: tuple = ("imgt", "chothia")

    # -- Splits --
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    temporal_train_cutoff: str = "2022-01-01"
    temporal_val_cutoff: str = "2023-06-01"

    # -- Residue graph --
    graph_k_nn: int = 10
    graph_spatial_cutoff: float = 8.0  # Angstrom

    # -- PLM embeddings --
    esm2_model: str = "esm2_t33_650M_UR50D"

    # -- Processing --
    n_workers: int = 8

    def ensure_dirs(self):
        """Create all output directories."""
        for d in [self.raw_dir, self.structures_dir, self.processed_dir,
                  self.splits_dir, self.logs_dir]:
            d.mkdir(parents=True, exist_ok=True)
