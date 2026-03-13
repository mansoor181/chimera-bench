"""Step 4-5: ANARCI numbering, CDR masks, epitope/paratope labels, contacts."""

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from anarci import anarci as run_anarci
from Bio.Data.PDBData import protein_letters_3to1_extended as _3to1
from Bio.PDB import PDBParser, NeighborSearch, Selection

from benchmark.config import Config

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)

# IMGT CDR definitions (residue number ranges, inclusive)
# Reference: dymean/configs.py IMGT class, IMGT/V-QUEST documentation
IMGT_CDR = {
    "H1": (27, 38), "H2": (56, 65), "H3": (105, 117),
    "L1": (27, 38), "L2": (56, 65), "L3": (105, 117),
}
# Chothia CDR definitions
# Reference: dymean/configs.py Chothia class
CHOTHIA_CDR = {
    "H1": (26, 32), "H2": (52, 56), "H3": (95, 102),
    "L1": (24, 34), "L2": (50, 56), "L3": (89, 97),
}

CDR_DEFS = {"imgt": IMGT_CDR, "chothia": CHOTHIA_CDR}


@dataclass
class ComplexAnnotation:
    """Annotation results for a single antibody-antigen complex."""
    complex_id: str
    numbering: dict = field(default_factory=dict)  # scheme -> chain -> [(resid, aa)]
    cdr_masks: dict = field(default_factory=dict)   # scheme -> per-residue labels
    epitope_residues: list = field(default_factory=list)
    paratope_residues: list = field(default_factory=list)
    contact_pairs: list = field(default_factory=list)
    sequences: dict = field(default_factory=dict)   # chain_type -> sequence


def number_with_anarci(sequence: str, scheme: str = "imgt") -> list[tuple]:
    """Run ANARCI on a single sequence. Returns [(position, aa), ...]."""
    results = run_anarci([("seq", sequence)], scheme=scheme, output=False)
    numbering_list, alignment_details, _ = results

    if numbering_list[0] is None:
        return []

    # numbering_list[0] is a list of (numbering, chain_type) tuples
    # Each numbering is a list of ((position, insertion_code), aa) tuples
    numbered = []
    for (pos, icode), aa in numbering_list[0][0][0]:
        if aa != "-":
            numbered.append((pos, icode, aa))
    return numbered


def get_cdr_mask(numbering: list[tuple], scheme: str,
                 chain_type: str) -> list[int]:
    """Assign CDR labels to numbered residues.

    Returns per-residue labels:
        -1=framework, 0=H1, 1=H2, 2=H3, 3=L1, 4=L2, 5=L3.
    """
    prefix = chain_type[0].upper()  # 'H' or 'L'
    cdr_ranges = CDR_DEFS[scheme]
    mask = []

    for pos, _icode, _aa in numbering:
        label = -1  # framework
        for i, cdr_idx in enumerate(["1", "2", "3"]):
            key = f"{prefix}{cdr_idx}"
            if key in cdr_ranges:
                lo, hi = cdr_ranges[key]
                if lo <= pos <= hi:
                    label = i + (0 if prefix == "H" else 3)
                    break
        mask.append(label)
    return mask


def compute_contacts(pdb_path: Path, h_chain: str, l_chain: str,
                     ag_chain: str, cutoff: float) -> tuple[list, list, list]:
    """Compute Ab-Ag contacts at atom level, derive epitope and paratope.

    Returns (contact_pairs, epitope_residues, paratope_residues).
    contact_pairs: [(ab_chain, ab_resid, ag_chain, ag_resid, distance)]
    epitope_residues: [(ag_chain, resid, resname)]
    paratope_residues: [(ab_chain, resid, resname)]
    """
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]

    ab_chains = [h_chain, l_chain]
    ab_atoms = []
    ag_atoms = []

    for chain in model:
        cid = chain.id
        for residue in chain.get_residues():
            if residue.id[0] != " ":
                continue
            atoms = list(residue.get_atoms())
            if cid in ab_chains:
                ab_atoms.extend(atoms)
            elif cid == ag_chain:
                ag_atoms.extend(atoms)

    if not ab_atoms or not ag_atoms:
        return [], [], []

    # Use NeighborSearch for efficient distance computation
    ns = NeighborSearch(ag_atoms)
    epitope_set = set()
    paratope_set = set()
    # Track minimum distance per residue pair for deduplication
    pair_min_dist = {}

    for ab_atom in ab_atoms:
        nearby = ns.search(ab_atom.get_vector().get_array(), cutoff, "A")
        for ag_atom in nearby:
            ab_res = ab_atom.get_parent()
            ag_res = ag_atom.get_parent()
            dist = ab_atom - ag_atom

            ab_key = (ab_res.get_parent().id, ab_res.id[1], ab_res.get_resname())
            ag_key = (ag_res.get_parent().id, ag_res.id[1], ag_res.get_resname())
            pair_key = (ab_key, ag_key)

            if pair_key not in pair_min_dist or dist < pair_min_dist[pair_key]:
                pair_min_dist[pair_key] = dist
            epitope_set.add(ag_key)
            paratope_set.add(ab_key)

    # One entry per residue pair with minimum atom-atom distance
    contact_pairs = [
        (*ab_key, *ag_key, round(d, 2))
        for (ab_key, ag_key), d in sorted(pair_min_dist.items())
    ]

    return contact_pairs, sorted(epitope_set), sorted(paratope_set)


def annotate_complex(pdb_path: Path, h_chain: str, l_chain: str,
                     ag_chain: str, cfg: Config) -> ComplexAnnotation:
    """Full annotation pipeline for a single complex."""
    complex_id = f"{pdb_path.stem}_{h_chain}_{l_chain}_{ag_chain}"
    ann = ComplexAnnotation(complex_id=complex_id)

    # Extract sequences
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]

    for chain_id, chain_type in [(h_chain, "heavy"), (l_chain, "light"),
                                  (ag_chain, "antigen")]:
        if chain_id not in model:
            continue
        seq = []
        for res in model[chain_id].get_residues():
            if res.id[0] != " ":
                continue
            try:
                seq.append(_3to1[res.get_resname()])
            except KeyError:
                seq.append("X")
        ann.sequences[chain_type] = "".join(seq)

    # Number heavy and light chains with ANARCI
    for scheme in cfg.numbering_schemes:
        ann.numbering[scheme] = {}
        ann.cdr_masks[scheme] = {}
        for chain_id, chain_type in [(h_chain, "heavy"), (l_chain, "light")]:
            seq = ann.sequences.get(chain_type, "")
            if not seq:
                continue
            numbered = number_with_anarci(seq, scheme=scheme)
            ann.numbering[scheme][chain_type] = numbered
            ann.cdr_masks[scheme][chain_type] = get_cdr_mask(
                numbered, scheme, chain_type
            )

    # Compute contacts
    ann.contact_pairs, ann.epitope_residues, ann.paratope_residues = (
        compute_contacts(pdb_path, h_chain, l_chain, ag_chain, cfg.contact_cutoff)
    )

    return ann


def annotate_all(df: pd.DataFrame, cfg: Config) -> list[ComplexAnnotation]:
    """Annotate all complexes in the filtered DataFrame."""
    annotations = []
    for i, row in df.iterrows():
        pdb_path = cfg.structures_dir / f"{row['pdb']}.pdb"
        if not pdb_path.exists():
            continue
        try:
            ann = annotate_complex(
                pdb_path, row["Hchain"], row["Lchain"],
                row["antigen_chain"], cfg
            )
            annotations.append(ann)
        except Exception as e:
            log.warning("Failed to annotate %s: %s", row["pdb"], e)

    log.info("Annotated %d / %d complexes", len(annotations), len(df))
    return annotations
