"""Step: Validate annotated complexes for structural and numbering quality.

Checks each complex for issues that would cause downstream baseline failures:
1. ANARCI numbering success (both heavy and light chains)
2. Conserved residue identity (CYS at IMGT 23/104, TRP at IMGT 41)
3. CDR completeness (all 6 CDRs identifiable via IMGT numbering)
4. Backbone CA atom completeness (no missing CA in any chain)

Runs after annotation, before feature extraction, so only valid complexes
get processed downstream.

Usage (standalone):
    python -m benchmark.data.validate
"""

import logging
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser

from benchmark.config import Config
from benchmark.data.annotate import ComplexAnnotation

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)

# IMGT conserved positions and their expected residues
CONSERVED_POSITIONS = {
    23: "C",   # CYS: canonical disulfide bond
    41: "W",   # TRP: hydrophobic core of VH/VL interface
    104: "C",  # CYS: second disulfide bond partner
}

# IMGT CDR ranges (inclusive)
IMGT_CDR_RANGES = {
    "CDR1": (27, 38),
    "CDR2": (56, 65),
    "CDR3": (105, 117),
}


def validate_numbering(ann: ComplexAnnotation) -> list[str]:
    """Check ANARCI IMGT numbering for both chains.

    Returns list of issue strings (empty = all good).
    """
    issues = []
    imgt = ann.numbering.get("imgt", {})

    for chain_type in ["heavy", "light"]:
        nums = imgt.get(chain_type, [])
        if not nums:
            issues.append(f"no_anarci_{chain_type}")
            continue

        # Build position -> amino acid lookup
        pos_to_aa = {}
        for pos, _icode, aa in nums:
            pos_to_aa[pos] = aa

        # Check conserved residues
        for pos, expected_aa in CONSERVED_POSITIONS.items():
            actual = pos_to_aa.get(pos)
            if actual is None:
                issues.append(f"{chain_type}_missing_pos{pos}")
            elif actual != expected_aa:
                issues.append(f"{chain_type}_non_{expected_aa}_at_{pos}({actual})")

        # Check CDR completeness
        for cdr_name, (start, end) in IMGT_CDR_RANGES.items():
            cdr_positions = [p for p in pos_to_aa if start <= p <= end]
            if not cdr_positions:
                issues.append(f"{chain_type}_missing_{cdr_name}")

    return issues


def validate_backbone(pdb_path: Path, h_chain: str, l_chain: str,
                      ag_chain: str) -> list[str]:
    """Check for missing CA atoms by parsing the PDB file directly.

    Returns list of issue strings (empty = all good).
    """
    if not pdb_path.exists():
        return ["missing_pdb_file"]

    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]
    issues = []

    chain_map = {"heavy": h_chain, "light": l_chain, "antigen": ag_chain}
    for chain_type, chain_id in chain_map.items():
        if chain_id not in model:
            continue
        for res in model[chain_id].get_residues():
            if res.id[0] != " ":
                continue
            if "CA" not in res:
                issues.append(f"{chain_type}_missing_CA({res.get_resname()}{res.id[1]})")
                break  # one missing CA per chain is enough to flag

    return issues


def validate_all(df: pd.DataFrame, annotations: list[ComplexAnnotation],
                 cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate all complexes. Returns (valid_df, excluded_df).

    excluded_df has an additional 'exclusion_reasons' column.
    """
    ann_map = {a.complex_id: a for a in annotations}

    valid_rows = []
    excluded_rows = []

    for _, row in df.iterrows():
        complex_id = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
        ann = ann_map.get(complex_id)

        issues = []

        # Numbering and CDR checks
        if ann is None:
            issues.append("no_annotation")
        else:
            issues.extend(validate_numbering(ann))

        # Backbone completeness (directly from PDB)
        pdb_path = cfg.structures_dir / f"{row['pdb']}.pdb"
        issues.extend(validate_backbone(
            pdb_path, row["Hchain"], row["Lchain"], row["antigen_chain"]))

        if issues:
            row_copy = row.copy()
            row_copy["exclusion_reasons"] = "; ".join(issues)
            excluded_rows.append(row_copy)
        else:
            valid_rows.append(row)

    valid_df = pd.DataFrame(valid_rows)
    excluded_df = pd.DataFrame(excluded_rows)

    log.info("Validation: %d valid, %d excluded out of %d total",
             len(valid_df), len(excluded_df), len(df))

    if len(excluded_df) > 0:
        log.info("Exclusion breakdown:")
        for _, erow in excluded_df.iterrows():
            cid = f"{erow['pdb']}_{erow['Hchain']}_{erow['Lchain']}_{erow['antigen_chain']}"
            log.info("  %s: %s", cid, erow["exclusion_reasons"])

    return valid_df, excluded_df


def run_validation(cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run full validation pipeline. Reads annotations from disk."""
    import pickle

    df = pd.read_csv(cfg.processed_dir / "deduplicated_summary.csv")
    ann_path = cfg.processed_dir / "annotations.pkl"
    with open(ann_path, "rb") as f:
        annotations = pickle.load(f)

    valid_df, excluded_df = validate_all(df, annotations, cfg)

    # Save results
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

    log.info("Saved validated_summary.csv (%d rows) and excluded_complexes.csv (%d rows)",
             len(valid_df), len(excluded_df))

    return valid_df, excluded_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = Config()
    run_validation(cfg)
