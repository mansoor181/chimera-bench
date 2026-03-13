"""Evaluate AbDockGen predictions with full CHIMERA-Bench metric suite.

Computes all 12 metrics (sequence, structure, interface, epitope, composites)
from saved .pt prediction files + native complex_features data.

AbDockGen is H3-only, so this is simpler than multi-CDR baselines -- there's
only one prediction directory (H3/).

Usage:
    cd baselines/abdockgen

    # Single prediction directory:
    python chimera_evaluate.py --predictions /path/to/predictions/H3/

    # Aggregate (auto-find H3/ under results_root/abdockgen/):
    python chimera_evaluate.py --aggregate --split epitope_group
"""

import argparse
import csv
import json
import logging
import os
import sys
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, run_full_evaluation, FULL_METRIC_KEYS,
)

log = logging.getLogger(__name__)


def find_prediction_dir(results_root, split):
    """Find H3 prediction directory for a given split."""
    results_root = Path(results_root)
    run_dir = results_root / f"abdockgen_H3_{split}"
    pred_dir = run_dir / "predictions" / "H3"
    if pred_dir.exists():
        return pred_dir
    return None


def build_csv_row(cdr_type, summary):
    row = {"cdr_type": cdr_type}
    for k in FULL_METRIC_KEYS:
        if k in summary:
            row[k] = f"{summary[k]['mean']:.2f}\u00b1{summary[k]['std']:.2f}"
        else:
            row[k] = ""
    return row


def save_metrics_csv(rows, output_path):
    fieldnames = ["cdr_type"] + list(FULL_METRIC_KEYS)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    log.info("Saved metrics CSV to %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate AbDockGen predictions with full CHIMERA metric suite",
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--predictions", type=Path,
                       help="Single predictions directory (.pt files)")
    group.add_argument("--aggregate", action="store_true",
                       help="Auto-find predictions from results_root/abdockgen/")
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
    csv_rows = []

    if args.predictions:
        log.info("Evaluating predictions from %s", args.predictions)
        per_complex, summary, cdr_type = run_full_evaluation(
            args.predictions, args.split, cdr_type_hint="H3",
        )
        csv_rows.append(build_csv_row(cdr_type, summary))

        results_json = args.predictions / "eval_results.json"
        with open(results_json, "w") as f:
            json.dump({"per_complex": per_complex, "summary": {
                k: v for k, v in summary.items()
            }}, f, indent=2, default=str)

        output_path = args.output or args.predictions.parent / "eval_metrics.csv"
    else:
        results_root = Path(shared["paths"]["results_root"]) / "abdockgen"
        pred_dir = find_prediction_dir(results_root, args.split)
        if pred_dir is None:
            log.error("No prediction dir found under %s for split=%s",
                      results_root, args.split)
            sys.exit(1)

        log.info("Evaluating H3 from %s", pred_dir)
        per_complex, summary, _ = run_full_evaluation(
            pred_dir, args.split, cdr_type_hint="H3",
        )
        csv_rows.append(build_csv_row("H3", summary))

        results_json = pred_dir / "eval_results.json"
        with open(results_json, "w") as f:
            json.dump({"per_complex": per_complex, "summary": {
                k: v for k, v in summary.items()
            }}, f, indent=2, default=str)

        output_path = args.output or results_root / f"eval_metrics_{args.split}.csv"

    save_metrics_csv(csv_rows, output_path)

    # Print summary table
    print("\nCHIMERA Full Evaluation Results (AbDockGen, H3 only)")
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
