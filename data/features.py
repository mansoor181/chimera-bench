"""Step: Coordinate extraction, SKEMPI linking, and complex tensor construction.

Graph construction is handled separately in benchmark.data.graphs.
PLM embeddings are handled separately in benchmark.data.embeddings.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio.PDB import PDBParser

from benchmark.config import Config
from benchmark.data.annotate import ComplexAnnotation

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)

# Standard heavy atoms (14 max) in PDB order
ATOM14_NAMES = [
    "N", "CA", "C", "O", "CB", "CG", "CG1", "CG2", "CD", "CD1", "CD2",
    "CE", "CE1", "CE2",
]


def extract_atom14_coords(pdb_path: Path, chain_id: str
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Extract 14-atom coordinates and mask for a chain.

    Returns:
        coords: (N_res, 14, 3) float array
        mask: (N_res, 14) bool array
    """
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]
    if chain_id not in model:
        return np.zeros((0, 14, 3)), np.zeros((0, 14), dtype=bool)

    residues = [r for r in model[chain_id].get_residues() if r.id[0] == " "]
    n_res = len(residues)
    coords = np.zeros((n_res, 14, 3), dtype=np.float32)
    mask = np.zeros((n_res, 14), dtype=bool)

    for i, res in enumerate(residues):
        for atom in res.get_atoms():
            name = atom.get_name()
            if name in ATOM14_NAMES:
                idx = ATOM14_NAMES.index(name)
                coords[i, idx] = atom.get_vector().get_array()
                mask[i, idx] = True

    return coords, mask


def link_skempi(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Link SKEMPI v2 affinity data to matching PDB entries."""
    skempi_path = cfg.raw_dir / "skempi_v2.csv"
    if not skempi_path.exists():
        log.warning("SKEMPI file not found at %s, skipping linking", skempi_path)
        return df

    skempi = pd.read_csv(skempi_path, sep=";")
    # Extract PDB codes from SKEMPI's #Pdb column (format: "1A22")
    skempi["pdb_lower"] = skempi["#Pdb"].str[:4].str.lower()

    # Get unique PDBs with ddG data
    skempi_pdbs = set(skempi["pdb_lower"].unique())
    df = df.copy()
    df["has_skempi"] = df["pdb"].str.lower().isin(skempi_pdbs)

    n_linked = df["has_skempi"].sum()
    log.info("Linked %d / %d complexes to SKEMPI v2", n_linked, len(df))
    return df


def _extract_ca_coords(pdb_path: Path, chain_id: str) -> np.ndarray:
    """Extract CA coordinates for a chain. Returns (N_res, 3)."""
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]
    if chain_id not in model:
        return np.zeros((0, 3), dtype=np.float32)

    coords = []
    for res in model[chain_id].get_residues():
        if res.id[0] != " ":
            continue
        if "CA" in res:
            coords.append(res["CA"].get_vector().get_array())
    return np.array(coords, dtype=np.float32) if coords else np.zeros((0, 3), dtype=np.float32)


def build_complex_tensor(pdb_path: Path, ann: ComplexAnnotation,
                         h_chain: str, l_chain: str, ag_chain: str) -> dict:
    """Build the processed tensor dict for a single complex.

    Extracts per-chain coordinates (atom14 + CA) and packages annotations.
    Graph construction is handled separately by the graphs step.
    """
    result = {"complex_id": ann.complex_id}

    chain_map = {"heavy": h_chain, "light": l_chain, "antigen": ag_chain}

    for chain_type in ["heavy", "light", "antigen"]:
        chain_id = chain_map[chain_type]
        seq = ann.sequences.get(chain_type, "")
        result[f"{chain_type}_sequence"] = seq

        # Atom14 coordinates
        coords14, mask14 = extract_atom14_coords(pdb_path, chain_id)
        result[f"{chain_type}_atom14_coords"] = coords14
        result[f"{chain_type}_atom14_mask"] = mask14

        # CA coordinates
        ca = _extract_ca_coords(pdb_path, chain_id)
        result[f"{chain_type}_ca_coords"] = ca

    # Annotations
    result["epitope_residues"] = ann.epitope_residues
    result["paratope_residues"] = ann.paratope_residues
    result["contact_pairs"] = ann.contact_pairs
    result["numbering"] = ann.numbering
    result["cdr_masks"] = ann.cdr_masks

    return result


def compute_all_features(df: pd.DataFrame,
                         annotations: list[ComplexAnnotation],
                         cfg: Config) -> int:
    """Build and save complex tensors (coords + annotations) for all complexes.

    Saves each complex as a .pt file in processed/complex_features/.
    Returns the number of successfully processed complexes.
    """
    out_dir = cfg.processed_dir / "complex_features"
    out_dir.mkdir(parents=True, exist_ok=True)

    ann_map = {a.complex_id: a for a in annotations}

    n_ok = 0
    n_total = len(df)
    for _, row in df.iterrows():
        pdb_id = row["pdb"]
        h_chain, l_chain = row["Hchain"], row["Lchain"]
        ag_chain = row["antigen_chain"]
        complex_id = f"{pdb_id}_{h_chain}_{l_chain}_{ag_chain}"

        ann = ann_map.get(complex_id)
        if ann is None:
            log.debug("No annotation for %s, skipping", complex_id)
            continue

        pdb_path = cfg.structures_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue

        out_path = out_dir / f"{complex_id}.pt"
        if out_path.exists():
            n_ok += 1
            continue

        try:
            tensor_dict = build_complex_tensor(
                pdb_path, ann, h_chain, l_chain, ag_chain
            )
            torch.save(tensor_dict, out_path)
            n_ok += 1
        except Exception as e:
            log.warning("Failed to build features for %s: %s", complex_id, e)

        if (n_ok % 500) == 0 and n_ok > 0:
            log.info("Processed %d / %d complexes", n_ok, n_total)

    log.info("Built features for %d / %d complexes -> %s", n_ok, n_total, out_dir)
    return n_ok
