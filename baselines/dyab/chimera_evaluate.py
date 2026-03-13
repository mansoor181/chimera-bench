"""Evaluate dyAb predictions with full CHIMERA-Bench metric suite.

Computes all 12 metrics (sequence, structure, interface, epitope, composites)
from saved .pt prediction files + native complex_features data.

Usage:
    cd baselines/dyab

    # Single CDR run:
    python chimera_evaluate.py --predictions /path/to/predictions/

    # Aggregate all CDR runs for a split:
    python chimera_evaluate.py --aggregate --split epitope_group
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

import yaml

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, run_full_evaluation, FULL_METRIC_KEYS,
)

log = logging.getLogger(__name__)


def load_numbering_scheme():
    """Load numbering scheme from config.yaml."""
    config_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("numbering_scheme", "chothia")


CDR_TYPES = ["H1", "H2", "H3", "L1", "L2", "L3"]


def find_cdr_prediction_dirs(results_root, split):
    """Find all CDR prediction directories under a results root for a split.

    dyAb run names follow the pattern: dyab_{cdr_label}_{split}
    where cdr_label is 'all', 'H3', 'H1+H2+H3', etc.

    For multi-CDR runs (cdr_label='all'), predictions are in per-CDR subdirectories:
        dyab_all_{split}/predictions/H1/, dyab_all_{split}/predictions/H2/, etc.
    """
    results_root = Path(results_root)
    dirs = {}
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or split not in d.name:
            continue
        pred_dir = d / "predictions"
        if not pred_dir.exists():
            continue
        name = d.name
        prefix = "dyab_"
        if not name.startswith(prefix):
            continue
        suffix = f"_{split}"
        if not name.endswith(suffix):
            continue
        cdr_label = name[len(prefix):-len(suffix)]

        # Check for per-CDR subdirectories (multi-CDR runs like 'all')
        has_cdr_subdirs = any((pred_dir / cdr).is_dir() for cdr in CDR_TYPES)
        if has_cdr_subdirs:
            # Multi-CDR run with per-CDR subdirectories
            for cdr in CDR_TYPES:
                cdr_subdir = pred_dir / cdr
                if cdr_subdir.is_dir() and any(cdr_subdir.glob("*.pt")):
                    dirs[cdr] = cdr_subdir
        elif any(pred_dir.glob("*.pt")):
            # Single-CDR run (predictions directly in predictions/)
            dirs[cdr_label] = pred_dir
    return dirs


def build_csv_row(cdr_type, summary):
    """Build a CSV row dict with mean+-std values."""
    row = {"cdr_type": cdr_type}
    for k in FULL_METRIC_KEYS:
        if k in summary:
            row[k] = f"{summary[k]['mean']:.2f}\u00b1{summary[k]['std']:.2f}"
        else:
            row[k] = ""
    return row


def save_metrics_csv(rows, output_path):
    """Save per-CDR metrics as CSV."""
    fieldnames = ["cdr_type"] + list(FULL_METRIC_KEYS)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Saved metrics CSV to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate dyAb predictions with full CHIMERA metric suite",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", type=Path,
                       help="Single predictions directory (.pt files)")
    group.add_argument("--aggregate", action="store_true",
                       help="Aggregate all CDR runs from results_root/dyab/")
    parser.add_argument("--split", default="epitope_group",
                        choices=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--output", type=Path, default=None,
                        help="Output CSV path (default: auto-generated)")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    shared = load_shared_config()
    numbering_scheme = load_numbering_scheme()
    log.info("Using numbering scheme: %s", numbering_scheme)
    csv_rows = []

    if args.predictions:
        # Check for per-CDR subdirectories
        has_cdr_subdirs = any((args.predictions / cdr).is_dir() for cdr in CDR_TYPES)

        if has_cdr_subdirs:
            log.info("Found per-CDR subdirectories in %s", args.predictions)
            for cdr in CDR_TYPES:
                cdr_dir = args.predictions / cdr
                if not cdr_dir.is_dir() or not any(cdr_dir.glob("*.pt")):
                    continue
                log.info("Evaluating %s from %s", cdr, cdr_dir)
                per_complex, summary, _ = run_full_evaluation(
                    cdr_dir, args.split, cdr_type_hint=cdr,
                    numbering_scheme=numbering_scheme,
                )
                csv_rows.append(build_csv_row(cdr, summary))

                results_json = cdr_dir / "eval_results.json"
                with open(results_json, "w") as f:
                    json.dump({"per_complex": per_complex, "summary": {
                        k: v for k, v in summary.items()
                    }}, f, indent=2, default=str)
        else:
            log.info("Evaluating predictions from %s", args.predictions)
            per_complex, summary, cdr_type = run_full_evaluation(
                args.predictions, args.split, numbering_scheme=numbering_scheme,
            )
            csv_rows.append(build_csv_row(cdr_type, summary))

            results_json = args.predictions / "eval_results.json"
            with open(results_json, "w") as f:
                json.dump({"per_complex": per_complex, "summary": {
                    k: v for k, v in summary.items()
                }}, f, indent=2, default=str)

        output_path = args.output or args.predictions.parent / "eval_metrics.csv"
    else:
        results_root = Path(shared["paths"]["results_root"]) / "dyab"
        cdr_dirs = find_cdr_prediction_dirs(results_root, args.split)
        if not cdr_dirs:
            log.error("No CDR prediction dirs found under %s for split=%s",
                      results_root, args.split)
            sys.exit(1)

        log.info("Found CDR prediction dirs: %s",
                 {k: str(v) for k, v in cdr_dirs.items()})

        for cdr_type in sorted(cdr_dirs.keys()):
            pred_dir = cdr_dirs[cdr_type]
            log.info("Evaluating %s from %s", cdr_type, pred_dir)
            per_complex, summary, _ = run_full_evaluation(
                pred_dir, args.split, cdr_type_hint=cdr_type,
                numbering_scheme=numbering_scheme,
            )
            csv_rows.append(build_csv_row(cdr_type, summary))

            results_json = pred_dir / "eval_results.json"
            with open(results_json, "w") as f:
                json.dump({"per_complex": per_complex, "summary": {
                    k: v for k, v in summary.items()
                }}, f, indent=2, default=str)

        output_path = args.output or results_root / f"eval_metrics_{args.split}.csv"

    save_metrics_csv(csv_rows, output_path)

    # Print summary table
    print("\nCHIMERA Full Evaluation Results (dyAb)")
    print(f"{'CDR':<12}", end="")
    for k in FULL_METRIC_KEYS:
        print(f" {k:>18}", end="")
    print()
    print("-" * (12 + 19 * len(FULL_METRIC_KEYS)))
    for row in csv_rows:
        print(f"{row['cdr_type']:<12}", end="")
        for k in FULL_METRIC_KEYS:
            print(f" {row.get(k, ''):>18}", end="")
        print()
    print(f"\nFull CSV: {output_path}")


if __name__ == "__main__":
    main()
