"""Category B converter: JSONL format for RefineGNN.

Output: JSONL file where each line is a JSON object with
{pdb, seq, cdr, coords: {N, CA, C, O}}.
"""

import json
import logging
from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended as _3to1
from Bio.PDB import PDBParser

from config import Config
from data.annotate import ComplexAnnotation

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)

# CDR label characters for RefineGNN
# annotation labels: 0=H1, 1=H2, 2=H3, 3=L1, 4=L2, 5=L3, -1=FR
CDR_LABEL_MAP = {0: "1", 1: "2", 2: "3", 3: "4", 4: "5", 5: "6", -1: "0"}

BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def _extract_backbone(model, chain_id: str) -> dict[str, list]:
    """Extract backbone atom coordinates for a chain."""
    if chain_id not in model:
        return {a: [] for a in BACKBONE_ATOMS}

    coords = {a: [] for a in BACKBONE_ATOMS}
    for res in model[chain_id].get_residues():
        if res.id[0] != " ":
            continue
        for atom_name in BACKBONE_ATOMS:
            if atom_name in res:
                c = res[atom_name].get_vector().get_array().tolist()
                coords[atom_name].append(c)
            else:
                coords[atom_name].append([float("nan")] * 3)
    return coords


def convert_complex(pdb_path: Path, h_chain: str, l_chain: str,
                    ann: ComplexAnnotation, scheme: str = "chothia") -> dict:
    """Convert a single complex to RefineGNN JSONL entry.

    RefineGNN operates on the heavy chain only (for CDR-H3 design).
    """
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]

    # Build heavy chain sequence and CDR mask
    seq = ""
    cdr_str = ""
    h_cdr_mask = ann.cdr_masks.get(scheme, {}).get("heavy", [])

    for i, res in enumerate(model[h_chain].get_residues()):
        if res.id[0] != " ":
            continue
        try:
            seq += _3to1[res.get_resname()]
        except KeyError:
            seq += "X"

        if i < len(h_cdr_mask):
            cdr_str += CDR_LABEL_MAP.get(h_cdr_mask[i], "0")
        else:
            cdr_str += "0"

    coords = _extract_backbone(model, h_chain)

    return {
        "pdb": ann.complex_id,
        "seq": seq,
        "cdr": cdr_str,
        "coords": coords,
    }


def convert_dataset(df, annotations: list[ComplexAnnotation], cfg: Config,
                    output_dir: Path = None) -> Path:
    """Convert all complexes to RefineGNN JSONL format."""
    output_dir = output_dir or cfg.processed_dir / "refinegnn_format"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "data.jsonl"

    ann_map = {a.complex_id: a for a in annotations}
    count = 0

    with open(output_path, "w") as f:
        for _, row in df.iterrows():
            cid = f"{row['pdb']}_{row['Hchain']}_{row['Lchain']}_{row['antigen_chain']}"
            ann = ann_map.get(cid)
            if ann is None:
                continue

            pdb_path = cfg.structures_dir / f"{row['pdb']}.pdb"
            if not pdb_path.exists():
                continue

            try:
                entry = convert_complex(pdb_path, row["Hchain"],
                                        row["Lchain"], ann)
                f.write(json.dumps(entry) + "\n")
                count += 1
            except Exception as e:
                log.warning("Failed to convert %s: %s", cid, e)

    log.info("Wrote %d entries to %s", count, output_path)
    return output_path
