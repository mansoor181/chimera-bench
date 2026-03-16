"""Category C converter: PDB + FASTA format for IgGM and ProteinMPNN.

IgGM: FASTA with 'X' at designable CDR positions + antigen PDB + epitope list.
ProteinMPNN: PDB with chain specs (designable vs fixed).
"""

import logging
import shutil
from pathlib import Path

from Bio.PDB import PDBParser

from config import Config
from data.annotate import ComplexAnnotation, CDR_DEFS

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)


def _mask_cdr_positions(sequence: str, cdr_mask: list[int],
                        target_cdrs: list[int]) -> str:
    """Replace CDR positions with 'X' in sequence."""
    result = list(sequence)
    for i, label in enumerate(cdr_mask):
        if i < len(result) and label in target_cdrs:
            result[i] = "X"
    return "".join(result)


def convert_for_iggm(pdb_path: Path, h_chain: str, l_chain: str,
                     ag_chain: str, ann: ComplexAnnotation,
                     output_dir: Path, cdr_types: list[str] = None,
                     scheme: str = "imgt") -> dict[str, Path]:
    """Convert a single complex for IgGM.

    Creates:
    - {complex_id}.fasta: H, L, A sequences with X at design positions
    - {complex_id}_antigen.pdb: antigen structure
    - {complex_id}_epitope.txt: space-separated epitope residue numbers
    """
    cdr_ranges = CDR_DEFS.get(scheme, CDR_DEFS["imgt"])
    if cdr_types is None:
        cdr_types = ["H3"]

    # Map CDR names to annotation label values
    cdr_name_to_label = {"H1": 0, "H2": 1, "H3": 2, "L1": 3, "L2": 4, "L3": 5}
    target_labels = [cdr_name_to_label[c] for c in cdr_types if c in cdr_name_to_label]

    # Build masked sequences
    h_seq = ann.sequences.get("heavy", "")
    l_seq = ann.sequences.get("light", "")
    ag_seq = ann.sequences.get("antigen", "")

    h_mask = ann.cdr_masks.get(scheme, {}).get("heavy", [])
    l_mask = ann.cdr_masks.get(scheme, {}).get("light", [])

    h_masked = _mask_cdr_positions(h_seq, h_mask, target_labels)
    l_masked = _mask_cdr_positions(l_seq, l_mask, target_labels)

    # Write FASTA
    fasta_path = output_dir / f"{ann.complex_id}.fasta"
    with open(fasta_path, "w") as f:
        f.write(f">H\n{h_masked}\n")
        f.write(f">L\n{l_masked}\n")
        f.write(f">A\n{ag_seq}\n")

    # Copy antigen PDB (or the full complex for epitope calculation)
    pdb_out = output_dir / f"{ann.complex_id}_antigen.pdb"
    shutil.copy2(pdb_path, pdb_out)

    # Write epitope residue numbers (1-indexed)
    epitope_nums = sorted(r for _, r, _ in ann.epitope_residues)
    epi_path = output_dir / f"{ann.complex_id}_epitope.txt"
    epi_path.write_text(" ".join(str(n) for n in epitope_nums))

    return {"fasta": fasta_path, "pdb": pdb_out, "epitope": epi_path}


def convert_for_proteinmpnn(pdb_path: Path, h_chain: str, l_chain: str,
                            ag_chain: str, ann: ComplexAnnotation,
                            output_dir: Path, cdr_types: list[str] = None,
                            scheme: str = "imgt") -> dict[str, Path]:
    """Convert a single complex for ProteinMPNN.

    Creates:
    - {complex_id}.pdb: full complex structure
    - {complex_id}_chains.txt: designable chain info
    - {complex_id}_positions.txt: designable residue positions
    """
    cdr_ranges = CDR_DEFS.get(scheme, CDR_DEFS["imgt"])
    if cdr_types is None:
        cdr_types = ["H3"]

    cdr_name_to_label = {"H1": 0, "H2": 1, "H3": 2, "L1": 3, "L2": 4, "L3": 5}
    target_labels = {cdr_name_to_label[c] for c in cdr_types if c in cdr_name_to_label}

    # Copy PDB
    pdb_out = output_dir / f"{ann.complex_id}.pdb"
    shutil.copy2(pdb_path, pdb_out)

    # Find designable positions per chain
    designable = {}
    for chain_type, chain_id in [("heavy", h_chain), ("light", l_chain)]:
        mask = ann.cdr_masks.get(scheme, {}).get(chain_type, [])
        positions = [str(i + 1) for i, label in enumerate(mask)
                     if label in target_labels]
        if positions:
            designable[chain_id] = positions

    # Write chain specification
    chains_path = output_dir / f"{ann.complex_id}_chains.txt"
    chains_path.write_text(" ".join(designable.keys()))

    # Write designable positions
    pos_path = output_dir / f"{ann.complex_id}_positions.txt"
    lines = [f"{chain} {' '.join(pos)}" for chain, pos in designable.items()]
    pos_path.write_text("\n".join(lines))

    return {"pdb": pdb_out, "chains": chains_path, "positions": pos_path}


def convert_dataset(df, annotations: list[ComplexAnnotation], cfg: Config,
                    method: str = "iggm", cdr_types: list[str] = None,
                    output_dir: Path = None) -> list[dict]:
    """Convert all complexes for IgGM or ProteinMPNN."""
    output_dir = output_dir or cfg.processed_dir / f"{method}_format"
    output_dir.mkdir(parents=True, exist_ok=True)

    convert_fn = convert_for_iggm if method == "iggm" else convert_for_proteinmpnn
    ann_map = {a.complex_id: a for a in annotations}
    results = []

    for _, row in df.iterrows():
        cid = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
        ann = ann_map.get(cid)
        if ann is None:
            continue

        pdb_path = cfg.structures_dir / f"{row['pdb']}.pdb"
        if not pdb_path.exists():
            continue

        try:
            paths = convert_fn(pdb_path, row["Hchain"], row["Lchain"],
                               row["antigen_chain"], ann, output_dir, cdr_types)
            results.append(paths)
        except Exception as e:
            log.warning("Failed to convert %s for %s: %s", cid, method, e)

    log.info("Converted %d complexes for %s", len(results), method)
    return results
