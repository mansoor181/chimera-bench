"""Step 3: Sequence-based deduplication via MMseqs2."""

import logging
import subprocess
import tempfile
from pathlib import Path

import pandas as pd
from Bio.Data.PDBData import protein_letters_3to1_extended as _3to1
from Bio.PDB import PDBParser

from config import Config

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)


def extract_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    """Extract amino acid sequence from a PDB chain using CA atoms."""
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]
    if chain_id not in model:
        return ""

    seq = []
    for residue in model[chain_id].get_residues():
        if residue.id[0] != " ":  # skip HETATMs
            continue
        try:
            seq.append(_3to1[residue.get_resname()])
        except KeyError:
            seq.append("X")
    return "".join(seq)


def build_fasta(df: pd.DataFrame, cfg: Config) -> Path:
    """Build a FASTA file with concatenated VH+VL sequences for clustering."""
    fasta_path = cfg.processed_dir / "antibody_sequences.fasta"
    entries = []

    for _, row in df.iterrows():
        pdb_path = cfg.structures_dir / f"{row['pdb']}.pdb"
        if not pdb_path.exists():
            continue
        vh = extract_sequence_from_pdb(pdb_path, row["Hchain"])
        vl = extract_sequence_from_pdb(pdb_path, row["Lchain"])
        if not vh or not vl:
            continue
        # Concatenate VH+VL for clustering
        complex_id = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
        entries.append(f">{complex_id}\n{vh}{vl}\n")

    fasta_path.write_text("".join(entries))
    log.info("Built FASTA with %d sequences at %s", len(entries), fasta_path)
    return fasta_path


def run_mmseqs2(fasta_path: Path, cfg: Config) -> dict[str, str]:
    """Run MMseqs2 easy-cluster and return {sequence_id: cluster_rep_id}."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result_prefix = Path(tmpdir) / "result"
        tmp_dir = Path(tmpdir) / "tmp"

        cmd = [
            "mmseqs", "easy-cluster",
            str(fasta_path),
            str(result_prefix),
            str(tmp_dir),
            "--min-seq-id", str(cfg.seq_identity_threshold),
            "-c", "0.8",
            "--cov-mode", "0",
        ]
        log.info("Running MMseqs2: %s", " ".join(cmd))
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Parse cluster TSV: columns are (representative, member)
        cluster_file = Path(f"{result_prefix}_cluster.tsv")
        clusters = {}
        for line in cluster_file.read_text().splitlines():
            rep, member = line.strip().split("\t")
            clusters[member] = rep

    log.info("MMseqs2 found %d clusters from %d sequences",
             len(set(clusters.values())), len(clusters))
    return clusters


def select_representatives(df: pd.DataFrame, clusters: dict[str, str],
                           cfg: Config) -> pd.DataFrame:
    """Keep only cluster representatives (highest resolution per cluster)."""
    # Build complex_id to match FASTA headers
    df = df.copy()
    df["complex_id"] = (
        df["pdb"] + "_" + df["Hchain"] + "_" + df["Lchain"]
        + "_" + df["antigen_chain"]
    )

    # Map each entry to its cluster representative
    df["cluster_rep"] = df["complex_id"].map(clusters)
    df = df[df["cluster_rep"].notna()]

    # Keep highest resolution per cluster
    df["resolution_num"] = pd.to_numeric(df["resolution"], errors="coerce")
    df = df.sort_values("resolution_num")
    df = df.drop_duplicates(subset="cluster_rep", keep="first")
    df = df.drop(columns=["cluster_rep", "resolution_num"])

    log.info("After deduplication: %d unique complexes", len(df))
    return df


def deduplicate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Full deduplication pipeline: extract seqs -> cluster -> select reps."""
    fasta_path = build_fasta(df, cfg)
    clusters = run_mmseqs2(fasta_path, cfg)
    df = select_representatives(df, clusters, cfg)

    out_path = cfg.processed_dir / "deduplicated_summary.csv"
    df.to_csv(out_path, index=False)
    log.info("Saved deduplicated summary to %s", out_path)
    return df
