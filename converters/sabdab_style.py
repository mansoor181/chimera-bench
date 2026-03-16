"""Category A converter: SAbDab-style tensors.

Target methods: DiffAb, AbFlowNet, AbX, MEAN, dyMEAN, dyAb, RAAD,
AbEgDiffuser, AbMEGD, LEAD, RADAb (11 methods).

Output per complex: dict with tensors matching DiffAb's dataset format.
"""

import logging
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser

from config import Config
from data.annotate import ComplexAnnotation, CDR_DEFS

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)

# DiffAb uses 15 heavy atoms max
MAX_ATOMS = 15
ATOM_ORDER = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2",
    "CE", "CE1", "CE2", "OXT",
]

# Fragment types: 1=Heavy, 2=Light, 3=Antigen
FRAGMENT_HEAVY = 1
FRAGMENT_LIGHT = 2
FRAGMENT_ANTIGEN = 3

# CDR flags: 0=FR, 1=H1, 2=H2, 3=H3, 4=L1, 5=L2, 6=L3
AA_MAP = {
    "ALA": 0, "CYS": 1, "ASP": 2, "GLU": 3, "PHE": 4, "GLY": 5,
    "HIS": 6, "ILE": 7, "LYS": 8, "LEU": 9, "MET": 10, "ASN": 11,
    "PRO": 12, "GLN": 13, "ARG": 14, "SER": 15, "THR": 16, "VAL": 17,
    "TRP": 18, "TYR": 19,
}
UNK_AA = 20


def _parse_chain(model, chain_id: str) -> tuple:
    """Extract residues from a chain: aa, coords(N,15,3), mask(N,15)."""
    if chain_id not in model:
        return [], np.zeros((0, MAX_ATOMS, 3)), np.zeros((0, MAX_ATOMS), dtype=bool)

    residues = [r for r in model[chain_id].get_residues() if r.id[0] == " "]
    n = len(residues)
    aa = []
    coords = np.zeros((n, MAX_ATOMS, 3), dtype=np.float32)
    mask = np.zeros((n, MAX_ATOMS), dtype=bool)

    for i, res in enumerate(residues):
        aa.append(AA_MAP.get(res.get_resname(), UNK_AA))
        for atom in res.get_atoms():
            name = atom.get_name()
            if name in ATOM_ORDER:
                idx = ATOM_ORDER.index(name)
                coords[i, idx] = atom.get_vector().get_array()
                mask[i, idx] = True

    return aa, coords, mask


def convert_complex(pdb_path: Path, h_chain: str, l_chain: str,
                    ag_chain: str, ann: ComplexAnnotation,
                    cdr_types: list[str] = None,
                    scheme: str = "chothia") -> dict:
    """Convert a single complex to SAbDab-style tensor dict.

    Args:
        cdr_types: which CDRs to mark as generate targets (e.g. ["H3"] or
                   ["H1","H2","H3","L1","L2","L3"]). None = all CDRs.
    """
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]

    # Parse each chain
    chains = []
    for chain_id, frag_type in [(h_chain, FRAGMENT_HEAVY),
                                 (l_chain, FRAGMENT_LIGHT),
                                 (ag_chain, FRAGMENT_ANTIGEN)]:
        aa, coords, mask = _parse_chain(model, chain_id)
        chains.append((aa, coords, mask, frag_type, chain_id))

    # Concatenate
    all_aa = []
    all_coords = []
    all_mask = []
    all_fragment = []
    all_chain_nb = []
    all_res_nb = []
    all_cdr_flag = []
    all_generate = []

    cdr_ranges = CDR_DEFS.get(scheme, CDR_DEFS["chothia"])
    if cdr_types is None:
        cdr_types = list(cdr_ranges.keys())

    for chain_idx, (aa, coords, mask, frag_type, chain_id) in enumerate(chains):
        n = len(aa)
        all_aa.extend(aa)
        all_coords.append(coords)
        all_mask.append(mask)
        all_fragment.extend([frag_type] * n)
        all_chain_nb.extend([chain_idx] * n)
        all_res_nb.extend(list(range(1, n + 1)))

        # CDR flags from annotation
        cdr_mask_data = ann.cdr_masks.get(scheme, {})
        chain_type = {FRAGMENT_HEAVY: "heavy", FRAGMENT_LIGHT: "light"}.get(frag_type)

        if chain_type and chain_type in cdr_mask_data:
            cdr_labels = cdr_mask_data[chain_type]
            # Map annotation labels to DiffAb convention
            # annotation: -1=FR, 0=H1, 1=H2, 2=H3, 3=L1, 4=L2, 5=L3
            # DiffAb:      0=FR, 1=H1, 2=H2, 3=H3, 4=L1, 5=L2, 6=L3
            mapped = [0 if c == -1 else c + 1 for c in cdr_labels]
            # Pad or truncate to match chain length
            if len(mapped) < n:
                mapped.extend([0] * (n - len(mapped)))
            all_cdr_flag.extend(mapped[:n])

            # Generate flag: 1 for target CDR residues
            cdr_name_to_flag = {"H1": 1, "H2": 2, "H3": 3,
                                "L1": 4, "L2": 5, "L3": 6}
            target_flags = {cdr_name_to_flag[c] for c in cdr_types
                            if c in cdr_name_to_flag}
            gen = [m in target_flags for m in mapped[:n]]
            all_generate.extend(gen)
        else:
            all_cdr_flag.extend([0] * n)
            all_generate.extend([False] * n)

    all_coords = np.concatenate(all_coords, axis=0) if all_coords else np.zeros((0, MAX_ATOMS, 3))
    all_mask = np.concatenate(all_mask, axis=0) if all_mask else np.zeros((0, MAX_ATOMS), dtype=bool)

    return {
        "id": ann.complex_id,
        "aa": torch.LongTensor(all_aa),
        "pos_heavyatom": torch.FloatTensor(all_coords),
        "mask_heavyatom": torch.BoolTensor(all_mask),
        "chain_nb": torch.LongTensor(all_chain_nb),
        "res_nb": torch.LongTensor(all_res_nb),
        "fragment_type": torch.LongTensor(all_fragment),
        "cdr_flag": torch.LongTensor(all_cdr_flag),
        "generate_flag": torch.BoolTensor(all_generate),
    }


def convert_dataset(df, annotations: list[ComplexAnnotation], cfg: Config,
                    cdr_types: list[str] = None,
                    output_dir: Path = None) -> list[dict]:
    """Convert all complexes to SAbDab-style format."""
    output_dir = output_dir or cfg.processed_dir / "sabdab_format"
    output_dir.mkdir(parents=True, exist_ok=True)

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
            data = convert_complex(pdb_path, row["Hchain"], row["Lchain"],
                                   row["antigen_chain"], ann, cdr_types)
            torch.save(data, output_dir / f"{cid}.pt")
            results.append(data)
        except Exception as e:
            log.warning("Failed to convert %s: %s", cid, e)

    log.info("Converted %d complexes to SAbDab-style format", len(results))
    return results
