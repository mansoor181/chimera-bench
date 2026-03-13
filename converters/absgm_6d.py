"""Category D converter: AbSGM 6D coordinate representation.

Target methods: AbSGM (1 method).

Produces pairwise 6D feature matrices from backbone coordinates:
  - dist6d: Cb-Cb distance matrix (normalized)
  - omega6d: Ca-Cb-Cb-Ca dihedral angles
  - theta6d: N-Ca-Cb-Cb dihedral angles
  - phi6d: Ca-Cb-Cb planar angles

Adapted from trRosetta (Yang et al., 2020).
"""

import logging
import math
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import PDBParser

from benchmark.config import Config
from benchmark.data.annotate import ComplexAnnotation

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)

DMAX = 20.0  # max Cb-Cb distance for contact definition


def _get_dihedrals(a: np.ndarray, b: np.ndarray,
                   c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Compute dihedral angles between 4 sets of points.

    Args: a,b,c,d each (N, 3)
    Returns: (N,) angles in [-pi, pi]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        b0 = -(b - a)
        b1 = c - b
        b2 = d - c

        b1 /= np.linalg.norm(b1, axis=-1)[:, None]
        v = b0 - np.sum(b0 * b1, axis=-1)[:, None] * b1
        w = b2 - np.sum(b2 * b1, axis=-1)[:, None] * b1

        x = np.sum(v * w, axis=-1)
        y = np.sum(np.cross(b1, v) * w, axis=-1)
    return np.arctan2(y, x)


def _get_angles(a: np.ndarray, b: np.ndarray,
                c: np.ndarray) -> np.ndarray:
    """Compute planar angles at point b.

    Args: a,b,c each (N, 3)
    Returns: (N,) angles in [0, pi]
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        v = a - b
        v /= np.linalg.norm(v, axis=-1)[:, None]
        w = c - b
        w /= np.linalg.norm(w, axis=-1)[:, None]
        x = np.sum(v * w, axis=1)
    return np.arccos(np.clip(x, -1, 1))


def _virtual_cb(n: np.ndarray, ca: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Reconstruct virtual Cb coordinates from backbone N, Ca, C.

    Uses trRosetta coefficients for ideal tetrahedral geometry.
    """
    b = ca - n
    cv = c - ca
    a = np.cross(b, cv)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * cv + ca


def get_coords6d(bb_coords: np.ndarray,
                 normalize: bool = True) -> np.ndarray:
    """Compute 6D coordinate representation from backbone atoms.

    Args:
        bb_coords: (N_res, 3, 3) array with [N, Ca, C] per residue
        normalize: scale to [-1, 1] range

    Returns:
        coords_6d: (N_res, N_res, 4) array with channels
                   [dist6d, omega6d, theta6d, phi6d]
    """
    N_atoms = bb_coords[:, 0]
    Ca = bb_coords[:, 1]
    C_atoms = bb_coords[:, 2]
    Cb = _virtual_cb(N_atoms, Ca, C_atoms)

    nres = len(Ca)

    # Find pairs within DMAX using Cb-Cb distances
    from scipy.spatial import cKDTree
    tree = cKDTree(Cb)
    pairs = tree.query_pairs(DMAX, output_type="ndarray")

    if len(pairs) == 0:
        return np.zeros((nres, nres, 4), dtype=np.float32)

    idx0, idx1 = pairs[:, 0], pairs[:, 1]

    # Cb-Cb distances
    dist6d = np.full((nres, nres), DMAX, dtype=np.float32)
    d = np.linalg.norm(Cb[idx1] - Cb[idx0], axis=-1)
    dist6d[idx0, idx1] = d
    dist6d[idx1, idx0] = d

    # Ca-Cb-Cb-Ca dihedral (omega)
    omega6d = np.zeros((nres, nres), dtype=np.float32)
    omega6d[idx0, idx1] = _get_dihedrals(Ca[idx0], Cb[idx0], Cb[idx1], Ca[idx1])
    omega6d[idx1, idx0] = _get_dihedrals(Ca[idx1], Cb[idx1], Cb[idx0], Ca[idx0])

    # N-Ca-Cb-Cb dihedral (theta)
    theta6d = np.zeros((nres, nres), dtype=np.float32)
    theta6d[idx0, idx1] = _get_dihedrals(N_atoms[idx0], Ca[idx0], Cb[idx0], Cb[idx1])
    theta6d[idx1, idx0] = _get_dihedrals(N_atoms[idx1], Ca[idx1], Cb[idx1], Cb[idx0])

    # Ca-Cb-Cb planar angle (phi)
    phi6d = np.zeros((nres, nres), dtype=np.float32)
    phi6d[idx0, idx1] = _get_angles(Ca[idx0], Cb[idx0], Cb[idx1])
    phi6d[idx1, idx0] = _get_angles(Ca[idx1], Cb[idx1], Cb[idx0])

    if normalize:
        dist6d = dist6d / DMAX * 2 - 1
        omega6d = omega6d / math.pi
        theta6d = theta6d / math.pi
        phi6d = phi6d / math.pi * 2 - 1

    return np.stack([dist6d, omega6d, theta6d, phi6d], axis=-1)


def _extract_backbone(model, chain_id: str) -> np.ndarray:
    """Extract N, Ca, C backbone coordinates for a chain.

    Returns: (N_res, 3, 3) array with [N, Ca, C] per residue.
    Residues missing any of N/Ca/C are skipped.
    """
    if chain_id not in model:
        return np.zeros((0, 3, 3), dtype=np.float32)

    coords = []
    for res in model[chain_id].get_residues():
        if res.id[0] != " ":
            continue
        if all(a in res for a in ["N", "CA", "C"]):
            coords.append([
                res["N"].get_vector().get_array(),
                res["CA"].get_vector().get_array(),
                res["C"].get_vector().get_array(),
            ])
    if not coords:
        return np.zeros((0, 3, 3), dtype=np.float32)
    return np.array(coords, dtype=np.float32)


def convert_complex(pdb_path: Path, h_chain: str, l_chain: str,
                    ag_chain: str, ann: ComplexAnnotation) -> dict:
    """Convert a single complex to AbSGM 6D representation.

    Returns dict with:
        coords_6d: (N_total, N_total, 4) tensor
        sequence: full concatenated sequence
        chain_lengths: (3,) array [n_heavy, n_light, n_antigen]
        cdr_mask: (N_total,) CDR labels (0=FR, 1-6=CDR)
    """
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]

    # Extract backbone coords for all chains
    h_bb = _extract_backbone(model, h_chain)
    l_bb = _extract_backbone(model, l_chain)
    ag_bb = _extract_backbone(model, ag_chain)

    # Concatenate
    all_bb = np.concatenate([h_bb, l_bb, ag_bb], axis=0)
    chain_lengths = np.array([len(h_bb), len(l_bb), len(ag_bb)])

    # Compute 6D representation
    coords_6d = get_coords6d(all_bb, normalize=True)

    # Build CDR mask for the concatenated sequence
    cdr_mask = np.zeros(len(all_bb), dtype=np.int64)
    cdr_data = ann.cdr_masks.get("imgt", {})
    offset = 0
    for chain_type, n in [("heavy", len(h_bb)), ("light", len(l_bb))]:
        if chain_type in cdr_data:
            labels = cdr_data[chain_type]
            for i, c in enumerate(labels[:n]):
                if c >= 0:
                    cdr_mask[offset + i] = c + 1
        offset += n

    return {
        "id": ann.complex_id,
        "coords_6d": torch.FloatTensor(coords_6d),
        "chain_lengths": torch.LongTensor(chain_lengths),
        "cdr_mask": torch.LongTensor(cdr_mask),
    }


def convert_dataset(df, annotations: list[ComplexAnnotation], cfg: Config,
                    output_dir: Path = None) -> list[dict]:
    """Convert all complexes to AbSGM 6D format."""
    output_dir = output_dir or cfg.processed_dir / "absgm_format"
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
                                   row["antigen_chain"], ann)
            torch.save(data, output_dir / f"{cid}.pt")
            results.append(data)
        except Exception as e:
            log.warning("Failed to convert %s: %s", cid, e)

    log.info("Converted %d complexes to AbSGM 6D format", len(results))
    return results
