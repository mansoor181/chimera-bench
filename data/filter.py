"""Step 2: Quality filtering of SAbDab complexes."""

import logging

import pandas as pd

from config import Config

log = logging.getLogger(__name__)


def filter_resolution(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep structures with resolution <= threshold."""
    mask = pd.to_numeric(df["resolution"], errors="coerce") <= cfg.max_resolution
    filtered = df[mask]
    log.info("Resolution filter (<=%.1fA): %d -> %d",
             cfg.max_resolution, len(df), len(filtered))
    return filtered


def filter_antigen_type(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep entries with protein/peptide antigens only."""
    mask = df["antigen_type"].str.lower().isin(cfg.antigen_types)
    filtered = df[mask]
    log.info("Antigen type filter %s: %d -> %d",
             cfg.antigen_types, len(df), len(filtered))
    return filtered


def filter_paired_chains(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep entries with both heavy and light chains."""
    if not cfg.require_paired_chains:
        return df
    mask = df["Hchain"].notna() & df["Lchain"].notna()
    filtered = df[mask]
    log.info("Paired chains filter: %d -> %d", len(df), len(filtered))
    return filtered


def filter_has_antigen(df: pd.DataFrame) -> pd.DataFrame:
    """Keep entries that have an antigen chain annotated."""
    mask = df["antigen_chain"].notna() & (df["antigen_chain"] != "")
    filtered = df[mask]
    log.info("Antigen chain filter: %d -> %d", len(df), len(filtered))
    return filtered


def filter_has_structure(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Keep entries whose PDB file was actually downloaded."""
    downloaded = {p.stem for p in cfg.structures_dir.glob("*.pdb")}
    mask = df["pdb"].isin(downloaded)
    filtered = df[mask]
    log.info("Structure availability filter: %d -> %d", len(df), len(filtered))
    return filtered


def deduplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows for the same PDB + chain combination.

    SAbDab can have multiple rows per PDB (different Ab-Ag pairings).
    We keep all unique (pdb, Hchain, Lchain, antigen_chain) tuples.
    """
    key_cols = ["pdb", "Hchain", "Lchain", "antigen_chain"]
    before = len(df)
    df = df.drop_duplicates(subset=key_cols)
    log.info("Row deduplication: %d -> %d", before, len(df))
    return df


def apply_filters(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Apply all quality filters in sequence."""
    log.info("Starting filtering: %d entries", len(df))
    df = filter_antigen_type(df, cfg)
    df = filter_paired_chains(df, cfg)
    df = filter_has_antigen(df)
    df = filter_resolution(df, cfg)
    df = filter_has_structure(df, cfg)
    df = deduplicate_rows(df)
    log.info("After all filters: %d entries", len(df))

    out_path = cfg.processed_dir / "filtered_summary.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved filtered summary to %s", out_path)
    return df
