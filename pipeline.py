"""CHIMERA-Bench dataset construction pipeline.

Usage:
    python pipeline.py --steps all
    python pipeline.py --steps collect filter dedup
    python pipeline.py --steps annotate features splits
    python pipeline.py --data-root /path/to/chimera-bench-v1.0
"""

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path

import pandas as pd

from config import Config
from data.annotate import annotate_all
from data.collect import collect
from data.dedup import deduplicate
from data.features import compute_all_features, link_skempi
from data.graphs import build_all_graphs
from data.filter import apply_filters
from data.splits import generate_splits
from data.validate import validate_all
from converters.sabdab_style import convert_dataset as convert_sabdab
from converters.refinegnn_jsonl import convert_dataset as convert_refinegnn
from converters.iggm_pdb_fasta import convert_dataset as convert_iggm_mpnn
from converters.cdr_only import convert_dataset as convert_cdr_only
from converters.absgm_6d import convert_dataset as convert_absgm

log = logging.getLogger(__name__)

LOG_FORMAT = "%(asctime)s %(name)s %(levelname)s %(message)s"


def setup_logging(cfg: Config, level: int, steps: list[str]):
    """Configure logging to both console and a timestamped log file."""
    cfg.ensure_dirs()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    step_tag = "-".join(steps)
    log_file = cfg.logs_dir / f"pipeline_{step_tag}_{timestamp}.log"

    root = logging.getLogger()
    root.setLevel(level)

    # Console handler
    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(console)

    # File handler
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)  # always capture DEBUG in file
    fh.setFormatter(logging.Formatter(LOG_FORMAT))
    root.addHandler(fh)

    log.info("Log file: %s", log_file)
    return log_file

ALL_STEPS = ["collect", "filter", "dedup", "annotate", "validate", "features",
             "graphs", "splits", "convert"]


def step_collect(cfg: Config):
    return collect(cfg)


def step_filter(cfg: Config):
    summary_path = cfg.raw_dir / "sabdab_summary_all.tsv"
    df = pd.read_csv(summary_path, sep="\t")
    return apply_filters(df, cfg)


def step_dedup(cfg: Config):
    df = pd.read_csv(cfg.processed_dir / "filtered_summary.csv")
    return deduplicate(df, cfg)


def step_annotate(cfg: Config):
    df = pd.read_csv(cfg.processed_dir / "deduplicated_summary.csv")
    annotations = annotate_all(df, cfg)

    # Save annotations
    out_path = cfg.processed_dir / "annotations.pkl"
    with open(out_path, "wb") as f:
        pickle.dump(annotations, f)
    log.info("Saved %d annotations to %s", len(annotations), out_path)
    return annotations


def step_features(cfg: Config):
    df = pd.read_csv(cfg.processed_dir / "final_summary.csv")

    # Build coordinate tensors only for validated complexes
    ann_path = cfg.processed_dir / "annotations.pkl"
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)
    compute_all_features(df, annotations, cfg)

    log.info("Feature computation complete for %d validated complexes.", len(df))
    return df


def step_validate(cfg: Config):
    df = pd.read_csv(cfg.processed_dir / "deduplicated_summary.csv")
    ann_path = cfg.processed_dir / "annotations.pkl"
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)

    valid_df, excluded_df = validate_all(df, annotations, cfg)

    # Save validated summary and exclusion list
    valid_df.to_csv(cfg.processed_dir / "validated_summary.csv", index=False)
    excluded_df.to_csv(cfg.processed_dir / "excluded_complexes.csv", index=False)

    # Filter annotations to only keep validated complexes
    valid_cids = set(
        f"{r['pdb']}_{r['Hchain']}_{r['Lchain']}_{r['antigen_chain']}"
        for _, r in valid_df.iterrows()
    )
    filtered_annotations = [a for a in annotations if a.complex_id in valid_cids]
    with open(ann_path, "wb") as f:
        pickle.dump(filtered_annotations, f)
    log.info("Updated annotations.pkl: %d -> %d (removed %d excluded)",
             len(annotations), len(filtered_annotations),
             len(annotations) - len(filtered_annotations))

    # Link SKEMPI and produce final_summary.csv from validated complexes only
    final_df = link_skempi(valid_df, cfg)
    final_df.to_csv(cfg.processed_dir / "final_summary.csv", index=False)

    log.info("Validation complete: %d valid, %d excluded. Saved final_summary.csv.",
             len(valid_df), len(excluded_df))
    return final_df


def step_graphs(cfg: Config):
    df = pd.read_csv(cfg.processed_dir / "final_summary.csv")
    ann_path = cfg.processed_dir / "annotations.pkl"
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)
    build_all_graphs(df, annotations, cfg)
    log.info("Graph construction complete.")


def step_splits(cfg: Config):
    df = pd.read_csv(cfg.processed_dir / "final_summary.csv")
    ann_path = cfg.processed_dir / "annotations.pkl"
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)

    return generate_splits(df, annotations, cfg)


def step_convert(cfg: Config):
    df = pd.read_csv(cfg.processed_dir / "final_summary.csv")
    ann_path = cfg.processed_dir / "annotations.pkl"
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)

    convert_dir = cfg.processed_dir / "converted"
    convert_dir.mkdir(parents=True, exist_ok=True)

    # Category A: SAbDab-style (11 methods)
    log.info("Converting: Category A (SAbDab-style)")
    convert_sabdab(df, annotations, cfg,
                   output_dir=convert_dir / "sabdab_format")

    # Category B: RefineGNN JSONL (1 method)
    log.info("Converting: Category B (RefineGNN JSONL)")
    convert_refinegnn(df, annotations, cfg,
                      output_dir=convert_dir / "refinegnn_format")

    # Category C: IgGM + ProteinMPNN PDB+FASTA (2 methods)
    log.info("Converting: Category C (IgGM)")
    convert_iggm_mpnn(df, annotations, cfg, method="iggm",
                      output_dir=convert_dir / "iggm_format")
    log.info("Converting: Category C (ProteinMPNN)")
    convert_iggm_mpnn(df, annotations, cfg, method="proteinmpnn",
                      output_dir=convert_dir / "proteinmpnn_format")

    # Category D: AbSGM 6D (1 method)
    log.info("Converting: Category D (AbSGM 6D)")
    convert_absgm(df, annotations, cfg,
                  output_dir=convert_dir / "absgm_format")

    # Category E: CDR-only (2 methods)
    log.info("Converting: Category E (CDR-only)")
    convert_cdr_only(df, annotations, cfg,
                     output_dir=convert_dir / "cdr_only_format")

    log.info("All format conversions complete.")


STEP_FUNCS = {
    "collect": step_collect,
    "filter": step_filter,
    "dedup": step_dedup,
    "annotate": step_annotate,
    "validate": step_validate,
    "features": step_features,
    "graphs": step_graphs,
    "splits": step_splits,
    "convert": step_convert,
}


def main():
    parser = argparse.ArgumentParser(description="CHIMERA-Bench pipeline")
    parser.add_argument(
        "--steps", nargs="+", default=["all"],
        help=f"Pipeline steps to run. Options: {ALL_STEPS} or 'all'",
    )
    parser.add_argument(
        "--data-root", type=Path, default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--n-workers", type=int, default=8,
        help="Number of parallel workers for downloads",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root
    cfg.n_workers = args.n_workers

    steps = ALL_STEPS if "all" in args.steps else args.steps
    level = getattr(logging, args.log_level)
    log_file = setup_logging(cfg, level, steps)

    for step_name in steps:
        if step_name not in STEP_FUNCS:
            log.error("Unknown step: %s. Choose from %s", step_name, ALL_STEPS)
            continue
        log.info("=== Running step: %s ===", step_name)
        STEP_FUNCS[step_name](cfg)
        log.info("=== Completed step: %s ===", step_name)


if __name__ == "__main__":
    main()
