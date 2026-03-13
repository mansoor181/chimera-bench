"""Category E converter: CDR-only format for AbODE and AbDockGen.

Extracts CDR sequences and structures as standalone entries.
"""

import json
import logging
from pathlib import Path

import numpy as np
from Bio.Data.PDBData import protein_letters_3to1_extended as _3to1
from Bio.PDB import PDBParser

from benchmark.config import Config
from benchmark.data.annotate import ComplexAnnotation, CDR_DEFS

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)


def extract_cdr_data(pdb_path: Path, chain_id: str, cdr_mask: list[int],
                     cdr_label: int) -> dict:
    """Extract coordinates and sequence for a single CDR region."""
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]
    if chain_id not in model:
        return None

    residues = [r for r in model[chain_id].get_residues() if r.id[0] == " "]
    seq = ""
    coords_n, coords_ca, coords_c, coords_o = [], [], [], []

    for i, res in enumerate(residues):
        if i >= len(cdr_mask) or cdr_mask[i] != cdr_label:
            continue
        try:
            seq += _3to1[res.get_resname()]
        except KeyError:
            seq += "X"

        for name, store in [("N", coords_n), ("CA", coords_ca),
                            ("C", coords_c), ("O", coords_o)]:
            if name in res:
                store.append(res[name].get_vector().get_array().tolist())
            else:
                store.append([0.0, 0.0, 0.0])

    if not seq:
        return None

    return {
        "sequence": seq,
        "length": len(seq),
        "coords": {"N": coords_n, "CA": coords_ca, "C": coords_c, "O": coords_o},
    }


def convert_complex(pdb_path: Path, h_chain: str, l_chain: str,
                    ag_chain: str, ann: ComplexAnnotation,
                    scheme: str = "chothia") -> list[dict]:
    """Extract all CDR regions from a complex."""
    cdr_names = {0: "H1", 1: "H2", 2: "H3", 3: "L1", 4: "L2", 5: "L3"}
    chain_map = {"heavy": h_chain, "light": l_chain}
    entries = []

    for chain_type, chain_id in chain_map.items():
        cdr_mask = ann.cdr_masks.get(scheme, {}).get(chain_type, [])
        for label, cdr_name in cdr_names.items():
            # Heavy chain CDRs have labels 0,1,2; light chain 3,4,5
            if chain_type == "heavy" and label > 2:
                continue
            if chain_type == "light" and label < 3:
                continue

            cdr_data = extract_cdr_data(pdb_path, chain_id, cdr_mask, label)
            if cdr_data is None:
                continue

            cdr_data["complex_id"] = ann.complex_id
            cdr_data["cdr_type"] = cdr_name

            # Add epitope info for AbDockGen (CDR-H3 + epitope)
            if cdr_name == "H3":
                cdr_data["epitope_residues"] = [
                    {"chain": c, "resid": r, "resname": n}
                    for c, r, n in ann.epitope_residues
                ]

            entries.append(cdr_data)

    return entries


def convert_dataset(df, annotations: list[ComplexAnnotation], cfg: Config,
                    output_dir: Path = None) -> Path:
    """Convert all complexes to CDR-only JSONL format."""
    output_dir = output_dir or cfg.processed_dir / "cdr_only_format"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "cdr_data.jsonl"

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
                entries = convert_complex(pdb_path, row["Hchain"],
                                          row["Lchain"], row["antigen_chain"], ann)
                for entry in entries:
                    f.write(json.dumps(entry) + "\n")
                    count += 1
            except Exception as e:
                log.warning("Failed to convert %s: %s", cid, e)

    log.info("Wrote %d CDR entries to %s", count, output_path)
    return output_path
