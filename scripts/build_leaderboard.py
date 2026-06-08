"""Aggregate per-method evaluation outputs into leaderboard.csv.

Expects a results directory laid out as::

    results/
        {method}/
            {split}/
                {cdr}/results.json     # output of `python -m chimera_bench.evaluate`

Each results.json is the JSON written by chimera_bench.evaluate (it has a
"summary" block of {metric: {mean, std, ...}}). This script flattens the
"best_*" summary metrics into one row per (method, split, cdr_type).

Usage::

    python scripts/build_leaderboard.py --results results/ --output leaderboard.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd

# summary key in results.json -> leaderboard column
METRIC_MAP = {
    "best_aar": "aar",
    "best_caar": "caar",
    "best_rmsd": "rmsd",
    "best_tm_score": "tm_score",
    "best_fnat": "fnat",
    "best_dockq": "dockq",
    "best_epitope_f1": "epitope_f1",
    "best_chimera_s": "chimera_s",
    "best_chimera_b": "chimera_b",
}

SPLIT_ORDER = {"epitope_group": 0, "antigen_fold": 1, "temporal": 2}
CDR_ORDER = {c: i for i, c in enumerate(["H1", "H2", "H3", "L1", "L2", "L3"])}


def _fmt(stat: dict) -> str:
    """Render a {mean, std} summary stat as 'mean±std'."""
    return f"{stat['mean']:.2f}\u00b1{stat.get('std', 0.0):.2f}"


def collect(results_dir: Path) -> pd.DataFrame:
    rows = []
    for res_file in sorted(results_dir.glob("*/*/*/results.json")):
        cdr = res_file.parent.name
        split = res_file.parent.parent.name
        method = res_file.parent.parent.parent.name
        summary = json.loads(res_file.read_text()).get("summary", {})
        row = {"method": method, "split": split, "cdr_type": cdr}
        for skey, col in METRIC_MAP.items():
            if skey in summary:
                row[col] = _fmt(summary[skey])
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["_s"] = df.split.map(SPLIT_ORDER).fillna(99)
    df["_c"] = df.cdr_type.map(CDR_ORDER).fillna(99)
    return (df.sort_values(["_s", "_c", "method"])
              .drop(columns=["_s", "_c"]).reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser(description="Build leaderboard.csv from results/")
    ap.add_argument("--results", type=Path, required=True,
                    help="Directory of {method}/{split}/{cdr}/results.json")
    ap.add_argument("--output", type=Path, default=Path("leaderboard.csv"))
    args = ap.parse_args()

    df = collect(args.results)
    if df.empty:
        raise SystemExit(f"No results.json found under {args.results}")
    df.to_csv(args.output, index=False)
    print(f"Wrote {args.output}: {len(df)} rows, {len(df.columns)} columns")


if __name__ == "__main__":
    main()
