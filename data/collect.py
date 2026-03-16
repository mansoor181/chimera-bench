"""Step 1-2: Download SAbDab summary and PDB structures."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

from config import Config

log = logging.getLogger(__name__)

SABDAB_SUMMARY_COLS = [
    "pdb", "Hchain", "Lchain", "model", "antigen_chain", "antigen_type",
    "antigen_het_name", "antigen_name", "short_header", "date", "compound",
    "organism", "heavy_subclass", "light_subclass", "light_ctype",
    "affinity", "delta_g", "affinity_method", "temperature", "pmid",
    "resolution", "method", "r_free", "r_factor", "scfv", "engineered",
    "heavy_species", "light_species", "antigen_species",
]


def download_sabdab_summary(cfg: Config) -> pd.DataFrame:
    """Download the SAbDab summary TSV and return as DataFrame."""
    out_path = cfg.raw_dir / "sabdab_summary_all.tsv"
    if out_path.exists():
        log.info("SAbDab summary already exists at %s", out_path)
        return pd.read_csv(out_path, sep="\t")

    log.info("Downloading SAbDab summary from %s", cfg.sabdab_summary_url)
    resp = requests.get(cfg.sabdab_summary_url, params={"output": "tsv"},
                        timeout=120)
    resp.raise_for_status()
    out_path.write_text(resp.text)
    log.info("Saved SAbDab summary (%d bytes) to %s", len(resp.text), out_path)
    return pd.read_csv(out_path, sep="\t")


def download_structure(pdb_id: str, cfg: Config) -> Path:
    """Download a single IMGT-numbered PDB from SAbDab."""
    out_path = cfg.structures_dir / f"{pdb_id}.pdb"
    if out_path.exists():
        return out_path

    url = cfg.sabdab_pdb_url.format(pdb_id)
    resp = requests.get(url, timeout=60)
    resp.raise_for_status()
    out_path.write_text(resp.text)
    return out_path


def download_structures(pdb_ids: list[str], cfg: Config) -> dict[str, Path]:
    """Download PDB structures in parallel. Returns {pdb_id: path}."""
    results = {}
    failed = []

    log.info("Downloading %d structures (workers=%d)", len(pdb_ids),
             cfg.n_workers)
    with ThreadPoolExecutor(max_workers=cfg.n_workers) as pool:
        futures = {
            pool.submit(download_structure, pid, cfg): pid
            for pid in pdb_ids
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                results[pid] = future.result()
            except Exception as e:
                log.warning("Failed to download %s: %s", pid, e)
                failed.append(pid)

    log.info("Downloaded %d/%d structures (%d failed)",
             len(results), len(pdb_ids), len(failed))
    if failed:
        fail_path = cfg.raw_dir / "download_failures.txt"
        fail_path.write_text("\n".join(failed))
    return results


def download_skempi(cfg: Config) -> Path:
    """Download SKEMPI v2 database."""
    out_path = cfg.raw_dir / "skempi_v2.csv"
    if out_path.exists():
        log.info("SKEMPI v2 already exists at %s", out_path)
        return out_path

    log.info("Downloading SKEMPI v2 from %s", cfg.skempi_url)
    resp = requests.get(cfg.skempi_url, timeout=120)
    resp.raise_for_status()
    out_path.write_bytes(resp.content)
    log.info("Saved SKEMPI v2 to %s", out_path)
    return out_path


def collect(cfg: Config) -> pd.DataFrame:
    """Run full collection: download summary, filter to Ab-Ag, download PDBs."""
    cfg.ensure_dirs()

    # Step 1: Get summary
    df = download_sabdab_summary(cfg)
    log.info("SAbDab summary: %d entries", len(df))

    # Step 2: Download SKEMPI
    download_skempi(cfg)

    # Step 3: Download structures for all entries
    pdb_ids = df["pdb"].dropna().unique().tolist()
    download_structures(pdb_ids, cfg)

    return df
