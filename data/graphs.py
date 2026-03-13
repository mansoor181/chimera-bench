"""Step: Residue graph construction using PyG HeteroData.

Builds multi-relational residue graphs for each antibody-antigen complex
with three node types (heavy, light, antigen) and 4 intra-chain edge
relation types plus inter-chain contact edges.

Output: single pkl file containing a list of HeteroData objects.
"""

import logging
import math
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree
from torch_geometric.data import HeteroData

from benchmark.config import Config
from benchmark.data.annotate import ComplexAnnotation

log = logging.getLogger(__name__)

PARSER = PDBParser(QUIET=True)

AA_TYPES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]
AA_TO_IDX = {aa: i for i, aa in enumerate(AA_TYPES)}
UNK_AA_IDX = 20  # unknown residue type

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

BACKBONE_ATOMS = ["N", "CA", "C", "O"]

# Inter-chain spatial cutoff (used by MEAN/dyMEAN/dyAb/RAAD for Ab-Ag edges)
INTER_CHAIN_CUTOFF = 12.0


def _positional_encoding(position: int, dim: int = 16) -> torch.Tensor:
    """Sinusoidal positional encoding."""
    pe = torch.zeros(dim)
    pos = torch.tensor([position], dtype=torch.float)
    div = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))
    pe[0::2] = torch.sin(pos * div)
    pe[1::2] = torch.cos(pos * div)
    return pe


def _rbf_encode(distances: torch.Tensor, n_bins: int = 16,
                d_max: float = 20.0) -> torch.Tensor:
    """Radial basis function encoding of distances."""
    centers = torch.linspace(0, d_max, n_bins)
    width = 1.0
    return torch.exp(-0.5 * ((distances.unsqueeze(-1) - centers) / width) ** 2)


def _extract_chain_residues(pdb_path: Path, chain_id: str) -> list[dict]:
    """Extract residue info for a chain.

    Each residue dict includes:
        resname: 3-letter AA code
        pdb_resid: PDB residue number (for annotation mapping)
        ca_coord: CA coordinates (3,)
        bb_coords: backbone coords [4, 3] (N, CA, C, O), NaN for missing
        atoms: dict of all heavy atom coords (for feature computation)
    """
    structure = PARSER.get_structure(pdb_path.stem, pdb_path)
    model = structure[0]
    if chain_id not in model:
        return []

    residues = []
    for res in model[chain_id].get_residues():
        if res.id[0] != " ":
            continue

        atoms = {}
        for atom in res.get_atoms():
            if atom.element != "H":
                atoms[atom.get_name()] = atom.get_coord()

        if "CA" not in atoms:
            continue

        # Backbone coords: [4, 3] for N, CA, C, O
        bb = np.full((4, 3), np.nan, dtype=np.float32)
        for i, aname in enumerate(BACKBONE_ATOMS):
            if aname in atoms:
                bb[i] = atoms[aname]

        residues.append({
            "resname": res.get_resname(),
            "pdb_resid": res.id[1],
            "ca_coord": atoms["CA"],
            "bb_coords": bb,
            "atoms": atoms,
        })
    return residues


def _compute_node_features(residues: list[dict]) -> torch.Tensor:
    """Compute per-residue node features.

    Features (105D total):
        - Residue type one-hot (20D)
        - Positional encoding (16D)
        - Backbone dihedral sin/cos (12D: 6 angles x 2)
        - CA-backbone RBF distances (48D: 3 atoms x 16 RBF)
        - Local coordinate frame (9D)
    """
    n = len(residues)
    if n == 0:
        return torch.zeros((0, 105), dtype=torch.float)

    features = []
    for idx, res in enumerate(residues):
        parts = []

        # 1. Residue type one-hot (20D)
        aa_idx = AA_TO_IDX.get(res["resname"], 0)
        aa_oh = torch.zeros(20)
        aa_oh[aa_idx] = 1.0
        parts.append(aa_oh)

        # 2. Positional encoding (16D)
        parts.append(_positional_encoding(idx))

        atoms = res["atoms"]
        ca = torch.tensor(atoms["CA"], dtype=torch.float)

        # 3. Backbone angles (12D)
        angles = torch.zeros(6)
        try:
            if "N" in atoms and "C" in atoms:
                n_pos = torch.tensor(atoms["N"], dtype=torch.float)
                c_pos = torch.tensor(atoms["C"], dtype=torch.float)
                # Bond angles
                v1 = n_pos - ca
                v2 = c_pos - ca
                cos_a = torch.dot(F.normalize(v1, dim=0), F.normalize(v2, dim=0))
                angles[3] = torch.acos(cos_a.clamp(-1, 1))
                if "O" in atoms:
                    o_pos = torch.tensor(atoms["O"], dtype=torch.float)
                    v3 = o_pos - c_pos
                    cos_b = torch.dot(F.normalize(v2, dim=0), F.normalize(v3, dim=0))
                    angles[4] = torch.acos(cos_b.clamp(-1, 1))
        except Exception:
            pass
        angle_feat = torch.stack([torch.sin(angles), torch.cos(angles)], dim=1).flatten()
        parts.append(angle_feat)

        # 4. CA-backbone RBF distances (48D)
        dist_feats = []
        for atom_name in ["C", "N", "O"]:
            if atom_name in atoms:
                d = torch.norm(ca - torch.tensor(atoms[atom_name], dtype=torch.float))
                dist_feats.append(_rbf_encode(d.unsqueeze(0)).squeeze(0))
            else:
                dist_feats.append(_rbf_encode(torch.tensor([1.5])).squeeze(0))
        parts.append(torch.cat(dist_feats))

        # 5. Local coordinate frame (9D)
        if all(a in atoms for a in ["CA", "C", "N"]):
            v1 = torch.tensor(atoms["C"], dtype=torch.float) - ca
            v2 = torch.tensor(atoms["N"], dtype=torch.float) - ca
            u1 = F.normalize(v1, dim=0)
            u2_t = v2 - torch.dot(v2, u1) * u1
            u2 = F.normalize(u2_t, dim=0)
            u3 = torch.linalg.cross(u1, u2)
            parts.append(torch.stack([u1, u2, u3], dim=1).flatten())
        else:
            parts.append(torch.eye(3).flatten())

        features.append(torch.cat(parts))

    return torch.stack(features)


def _build_edge_features(i: int, j: int, ca_coords: torch.Tensor,
                         rel_type: int) -> torch.Tensor:
    """Compute edge features between residues i and j.

    Features (37D total):
        - Edge type one-hot (4D)
        - Relative positional encoding (16D)
        - CA-CA distance RBF (16D)
        - Normalized direction vector (3D) -- should be 2 but keeping for alignment
    Simplified from epiformer's 100D.
    """
    parts = []

    # Edge type (4D)
    etype = torch.zeros(4)
    etype[rel_type] = 1.0
    parts.append(etype)

    # Relative position encoding (16D)
    parts.append(_positional_encoding(j - i))

    # CA-CA distance RBF (16D)
    dist = torch.norm(ca_coords[i] - ca_coords[j])
    parts.append(_rbf_encode(dist.unsqueeze(0)).squeeze(0))

    # Direction vector (3D)
    direction = ca_coords[j] - ca_coords[i]
    norm = torch.norm(direction)
    if norm > 1e-6:
        direction = direction / norm
    parts.append(direction)

    return torch.cat(parts)  # 4 + 16 + 16 + 3 = 39D


EDGE_FEAT_DIM = 39


def _build_chain_edges(ca_coords: torch.Tensor, k_nn: int = 10,
                       spatial_cutoff: float = 8.0
                       ) -> dict[int, list[list[int]]]:
    """Build 4 relation edge lists for a single chain.

    r0: sequential +/-1
    r1: sequential +/-2
    r2: k-NN spatial
    r3: spatial cutoff (excluding r0-r2)
    """
    n = len(ca_coords)
    if n == 0:
        return {r: [] for r in range(4)}

    edge_lists = {r: [] for r in range(4)}
    connected = set()

    # Sequential edges
    for i in range(n):
        for offset, rel in [(1, 0), (-1, 0), (2, 1), (-2, 1)]:
            j = i + offset
            if 0 <= j < n:
                edge_lists[rel].append([i, j])
                connected.add((i, j))

    # k-NN edges
    np_coords = ca_coords.numpy()
    tree = cKDTree(np_coords)
    _, indices = tree.query(np_coords, k=min(k_nn + 1, n))
    for i in range(n):
        for j_idx in range(1, indices.shape[1]):
            j = int(indices[i, j_idx])
            edge_lists[2].append([i, j])
            connected.add((i, j))

    # Spatial edges (not already connected)
    pairs = tree.query_pairs(spatial_cutoff)
    for i, j in pairs:
        if (i, j) not in connected:
            edge_lists[3].append([i, j])
            edge_lists[3].append([j, i])

    return edge_lists


def build_complex_graph(pdb_path: Path, ann: ComplexAnnotation,
                        h_chain: str, l_chain: str, ag_chain: str,
                        cfg: Config) -> HeteroData:
    """Build a PyG HeteroData residue graph for a single complex.

    Node types: heavy_res, light_res, ag_res
    Intra-chain edges: r0 (seq±1), r1 (seq±2), r2 (k-NN), r3 (spatial)
    Inter-chain edges: contact (from annotation contact_pairs)
    """
    data = HeteroData()
    data.complex_id = ann.complex_id

    chain_info = {
        "heavy": (h_chain, "heavy_res"),
        "light": (l_chain, "light_res"),
        "antigen": (ag_chain, "ag_res"),
    }

    chain_residues = {}
    chain_ca = {}
    # Map (chain_id, pdb_resid) -> (chain_type, sequential_index)
    resid_to_idx = {}

    for chain_type, (chain_id, node_type) in chain_info.items():
        residues = _extract_chain_residues(pdb_path, chain_id)
        chain_residues[chain_type] = residues
        n = len(residues)

        # Build PDB resid -> sequential index mapping
        for i, r in enumerate(residues):
            resid_to_idx[(chain_id, r["pdb_resid"])] = (chain_type, i)

        # -- Node features (105D) --
        node_feat = _compute_node_features(residues)
        data[node_type].x = node_feat
        data[node_type].num_nodes = n

        # -- CA coordinates (N, 3) --
        ca = torch.from_numpy(
            np.array([r["ca_coord"] for r in residues], dtype=np.float32)
        ) if residues else torch.zeros((0, 3), dtype=torch.float)
        data[node_type].pos = ca
        chain_ca[chain_type] = ca

        # -- Backbone coordinates (N, 4, 3): N, CA, C, O --
        bb = torch.from_numpy(
            np.stack([r["bb_coords"] for r in residues])
        ) if residues else torch.zeros((0, 4, 3), dtype=torch.float)
        data[node_type].bb_coords = bb

        # -- Residue type index (N,) for embedding layers --
        aa = torch.tensor(
            [AA_TO_IDX.get(r["resname"], UNK_AA_IDX) for r in residues],
            dtype=torch.long
        )
        data[node_type].aa = aa

        # -- Sequence string --
        seq = "".join(AA_3TO1.get(r["resname"], "X") for r in residues)
        data[node_type].seq = seq

        # -- Residue index within chain (0-indexed) --
        data[node_type].res_idx = torch.arange(n, dtype=torch.long)

        # -- CDR flags --
        cdr_masks = ann.cdr_masks.get("imgt", {})
        if chain_type in ("heavy", "light") and chain_type in cdr_masks:
            labels = cdr_masks[chain_type]
            if len(labels) < n:
                labels = labels + [-1] * (n - len(labels))
            # Map: -1=FR(0), 0=H1(1), 1=H2(2), 2=H3(3), 3=L1(4), 4=L2(5), 5=L3(6)
            mapped = torch.tensor([0 if c == -1 else c + 1 for c in labels[:n]],
                                  dtype=torch.long)
            data[node_type].cdr_flag = mapped
            # Generate flag: True for all CDR residues (designable targets)
            data[node_type].generate_flag = mapped > 0
        else:
            data[node_type].cdr_flag = torch.zeros(n, dtype=torch.long)
            data[node_type].generate_flag = torch.zeros(n, dtype=torch.bool)

        # -- Epitope / paratope labels --
        # Format: epitope_residues = [(ag_chain_id, resid, resname), ...]
        #         paratope_residues = [(ab_chain_id, resid, resname), ...]
        if chain_type == "antigen":
            epi = torch.zeros(n, dtype=torch.long)
            for entry in ann.epitope_residues:
                key = (entry[0], entry[1])
                if key in resid_to_idx and resid_to_idx[key][0] == "antigen":
                    epi[resid_to_idx[key][1]] = 1
            data[node_type].epitope = epi
        elif chain_type in ("heavy", "light"):
            para = torch.zeros(n, dtype=torch.long)
            for entry in ann.paratope_residues:
                key = (entry[0], entry[1])
                if key in resid_to_idx and resid_to_idx[key][0] == chain_type:
                    para[resid_to_idx[key][1]] = 1
            data[node_type].paratope = para

        # -- Intra-chain edges (4 relation types) --
        _add_intra_edges(data, node_type, ca, n, cfg)

    # -- Inter-chain contact edges (from annotation, 4.5A cutoff) --
    _add_contact_edges(data, ann, resid_to_idx)

    # -- Inter-chain spatial edges (12A cutoff for broader context) --
    _add_interchain_spatial_edges(data, chain_ca)

    return data


def _add_contact_edges(data: HeteroData, ann: ComplexAnnotation,
                       resid_to_idx: dict):
    """Add annotation-based contact edges (4.5A cutoff) between Ab and Ag."""
    h_ag_src, h_ag_dst = [], []
    l_ag_src, l_ag_dst = [], []

    for cp in ann.contact_pairs:
        ab_cid, ab_resid = cp[0], cp[1]
        ag_cid, ag_resid = cp[3], cp[4]

        ab_key = (ab_cid, ab_resid)
        ag_key = (ag_cid, ag_resid)

        if ab_key not in resid_to_idx or ag_key not in resid_to_idx:
            continue

        ab_chain_type, ab_idx = resid_to_idx[ab_key]
        ag_chain_type, ag_idx = resid_to_idx[ag_key]

        if ag_chain_type != "antigen":
            continue

        if ab_chain_type == "heavy":
            h_ag_src.append(ab_idx)
            h_ag_dst.append(ag_idx)
        elif ab_chain_type == "light":
            l_ag_src.append(ab_idx)
            l_ag_dst.append(ag_idx)

    for ab_type, src, dst in [("heavy_res", h_ag_src, h_ag_dst),
                               ("light_res", l_ag_src, l_ag_dst)]:
        if src:
            data[ab_type, "contacts", "ag_res"].edge_index = torch.tensor(
                [src, dst], dtype=torch.long)
        else:
            data[ab_type, "contacts", "ag_res"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long)


def _add_interchain_spatial_edges(data: HeteroData,
                                  chain_ca: dict[str, torch.Tensor]):
    """Add inter-chain spatial edges at 12A CA-CA cutoff with RBF features.

    These broader edges (vs 4.5A contacts) provide the spatial context
    used by MEAN/dyMEAN/dyAb/RAAD for message passing across chains.
    """
    pairs = [("heavy", "heavy_res", "ag_res"),
             ("light", "light_res", "ag_res")]

    for ab_type, ab_node, ag_node in pairs:
        ab_ca = chain_ca.get(ab_type, torch.zeros((0, 3)))
        ag_ca = chain_ca.get("antigen", torch.zeros((0, 3)))

        if len(ab_ca) == 0 or len(ag_ca) == 0:
            data[ab_node, "spatial", ag_node].edge_index = torch.zeros(
                (2, 0), dtype=torch.long)
            data[ab_node, "spatial", ag_node].edge_attr = torch.zeros(
                (0, 16), dtype=torch.float)
            continue

        # Compute pairwise CA-CA distances
        dists = torch.cdist(ab_ca, ag_ca)
        src, dst = torch.nonzero(dists < INTER_CHAIN_CUTOFF, as_tuple=True)

        if len(src) > 0:
            edge_dists = dists[src, dst]
            edge_attr = _rbf_encode(edge_dists)  # (E, 16)
            data[ab_node, "spatial", ag_node].edge_index = torch.stack(
                [src, dst], dim=0)
            data[ab_node, "spatial", ag_node].edge_attr = edge_attr
        else:
            data[ab_node, "spatial", ag_node].edge_index = torch.zeros(
                (2, 0), dtype=torch.long)
            data[ab_node, "spatial", ag_node].edge_attr = torch.zeros(
                (0, 16), dtype=torch.float)

    # Also add heavy-light spatial edges for inter-Ab-chain context
    h_ca = chain_ca.get("heavy", torch.zeros((0, 3)))
    l_ca = chain_ca.get("light", torch.zeros((0, 3)))
    if len(h_ca) > 0 and len(l_ca) > 0:
        dists = torch.cdist(h_ca, l_ca)
        src, dst = torch.nonzero(dists < INTER_CHAIN_CUTOFF, as_tuple=True)
        if len(src) > 0:
            edge_dists = dists[src, dst]
            data["heavy_res", "spatial", "light_res"].edge_index = torch.stack(
                [src, dst], dim=0)
            data["heavy_res", "spatial", "light_res"].edge_attr = _rbf_encode(
                edge_dists)
        else:
            data["heavy_res", "spatial", "light_res"].edge_index = torch.zeros(
                (2, 0), dtype=torch.long)
            data["heavy_res", "spatial", "light_res"].edge_attr = torch.zeros(
                (0, 16), dtype=torch.float)
    else:
        data["heavy_res", "spatial", "light_res"].edge_index = torch.zeros(
            (2, 0), dtype=torch.long)
        data["heavy_res", "spatial", "light_res"].edge_attr = torch.zeros(
            (0, 16), dtype=torch.float)


def _add_intra_edges(data: HeteroData, node_type: str,
                     ca: torch.Tensor, n: int, cfg: Config):
    """Add 4-relation intra-chain edges to a HeteroData node type."""
    if n > 0:
        edge_lists = _build_chain_edges(ca, k_nn=cfg.graph_k_nn,
                                        spatial_cutoff=cfg.graph_spatial_cutoff)
        for rel_type in range(4):
            rel_name = f"r{rel_type}"
            edges = edge_lists[rel_type]
            if edges:
                ei = torch.tensor(edges, dtype=torch.long).t().contiguous()
                attrs = []
                for src, dst in edges:
                    attrs.append(_build_edge_features(src, dst, ca, rel_type))
                data[node_type, rel_name, node_type].edge_index = ei
                data[node_type, rel_name, node_type].edge_attr = torch.stack(attrs)
            else:
                data[node_type, rel_name, node_type].edge_index = torch.zeros(
                    (2, 0), dtype=torch.long)
                data[node_type, rel_name, node_type].edge_attr = torch.zeros(
                    (0, EDGE_FEAT_DIM), dtype=torch.float)
    else:
        for rel_type in range(4):
            rel_name = f"r{rel_type}"
            data[node_type, rel_name, node_type].edge_index = torch.zeros(
                (2, 0), dtype=torch.long)
            data[node_type, rel_name, node_type].edge_attr = torch.zeros(
                (0, EDGE_FEAT_DIM), dtype=torch.float)


def build_all_graphs(df, annotations: list[ComplexAnnotation],
                     cfg: Config) -> int:
    """Build HeteroData residue graphs for all complexes.

    Saves a single pkl file containing a dict mapping complex_id -> HeteroData.
    Returns number of successfully processed complexes.
    """
    out_path = cfg.processed_dir / "residue_graphs.pkl"

    # Load existing if resuming
    graphs = {}
    if out_path.exists():
        with open(out_path, "rb") as f:
            graphs = pickle.load(f)
        log.info("Loaded %d existing graphs from %s", len(graphs), out_path)

    ann_map = {a.complex_id: a for a in annotations}
    n_total = len(df)
    n_new = 0

    for idx, (_, row) in enumerate(df.iterrows()):
        pdb_id = row["pdb"]
        h_chain, l_chain = row["Hchain"], row["Lchain"]
        ag_chain = row["antigen_chain"]
        complex_id = f"{pdb_id}_{h_chain}_{l_chain}_{ag_chain}"

        if complex_id in graphs:
            continue

        ann = ann_map.get(complex_id)
        if ann is None:
            continue

        pdb_path = cfg.structures_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            continue

        try:
            g = build_complex_graph(pdb_path, ann, h_chain, l_chain, ag_chain, cfg)
            graphs[complex_id] = g
            n_new += 1
        except Exception as e:
            log.warning("Failed to build graph for %s: %s", complex_id, e)

        if (n_new % 500) == 0 and n_new > 0:
            log.info("Built %d new graphs (%d total, %d remaining)",
                     n_new, len(graphs), n_total - len(graphs))
            # Checkpoint save
            with open(out_path, "wb") as f:
                pickle.dump(graphs, f)

    # Final save
    with open(out_path, "wb") as f:
        pickle.dump(graphs, f)

    log.info("Built %d new graphs, %d total -> %s", n_new, len(graphs), out_path)
    return len(graphs)
