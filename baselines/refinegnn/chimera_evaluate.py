"""Evaluate RefineGNN predictions with full CHIMERA-Bench metric suite.

Computes all 12 metrics (sequence, structure, interface, epitope, composites)
from saved .pt prediction files + native complex_features data.

Usage:
    cd baselines/refinegnn

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


def load_numbering_scheme():
    """Load numbering scheme from config.yaml."""
    config_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    return cfg.get("numbering_scheme", "chothia")


log = logging.getLogger(__name__)


def find_cdr_prediction_dirs(results_root, split):
    """Find all CDR prediction directories under results root for a split."""
    results_root = Path(results_root)
    dirs = {}
    for d in sorted(results_root.iterdir()):
        if not d.is_dir() or split not in d.name:
            continue
        pred_dir = d / "predictions"
        if not pred_dir.exists():
            continue
        for cdr_num in ["1", "2", "3"]:
            if f"cdr{cdr_num}" in d.name:
                dirs[f"H{cdr_num}"] = pred_dir
                break
    return dirs


def build_csv_row(cdr_type, summary):
    """Build a CSV row dict with mean\u00b1std values."""
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
        description="Evaluate RefineGNN predictions with full CHIMERA metric suite",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", type=Path,
                       help="Single predictions directory (.pt files)")
    group.add_argument("--aggregate", action="store_true",
                       help="Aggregate all CDR runs from results_root/refinegnn/")
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
        results_root = Path(shared["paths"]["results_root"]) / "refinegnn"
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
    print("\nCHIMERA Full Evaluation Results")
    print(f"{'CDR':<6}", end="")
    for k in FULL_METRIC_KEYS:
        print(f" {k:>18}", end="")
    print()
    print("-" * (6 + 19 * len(FULL_METRIC_KEYS)))
    for row in csv_rows:
        print(f"{row['cdr_type']:<6}", end="")
        for k in FULL_METRIC_KEYS:
            print(f" {row.get(k, ''):>18}", end="")
        print()
    print(f"\nFull CSV: {output_path}")


if __name__ == "__main__":
    main()
