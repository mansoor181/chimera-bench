"""CHIMERA-Bench: epitope-specific antibody CDR design benchmark.

Public API::

    import chimera_bench as cb

    cb.load_split("epitope_group")        # {"train", "val", "test"}
    cb.load_complex(cid)                   # per-complex feature dict
    cb.list_complexes("epitope_group", "test")
    cb.ChimeraDataset("epitope_group", "test")   # torch Dataset

    # Evaluation
    from chimera_bench.evaluate import evaluate_benchmark, load_natives
    from chimera_bench import metrics
"""

__version__ = "1.0.0"

from .data import (
    SPLITS,
    SUBSETS,
    ChimeraDataset,
    cdr_indices,
    get_data_root,
    list_complexes,
    load_complex,
    load_metadata,
    load_split,
)
from .evaluate import evaluate_benchmark, evaluate_single, load_natives

__all__ = [
    "__version__",
    "SPLITS",
    "SUBSETS",
    "ChimeraDataset",
    "cdr_indices",
    "get_data_root",
    "list_complexes",
    "load_complex",
    "load_metadata",
    "load_split",
    "evaluate_benchmark",
    "evaluate_single",
    "load_natives",
]
