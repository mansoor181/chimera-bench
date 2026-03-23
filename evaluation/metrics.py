"""Evaluation metrics for CHIMERA-Bench (7 groups).

Groups:
    1. Sequence quality: AAR, CAAR, perplexity, liabilities
    2. Structure quality: RMSD (Kabsch), TM-score
    3. Binding interface: Fnat, iRMSD, DockQ
    4. Epitope specificity: precision, recall, F1
    5. Diversity: pairwise sequence diversity, structural diversity
    6. Designability: sequence liabilities
    7. Novelty: distance to training set sequences

All aggregate functions support bootstrap 95% CIs.
"""

import numpy as np
from scipy.spatial import cKDTree


# -- Group 1: Sequence Quality --

def aar(pred_seq: str, true_seq: str) -> float:
    """Amino Acid Recovery: fraction of matching residues."""
    matches = sum(p == t for p, t in zip(pred_seq, true_seq))
    return matches / len(true_seq) if true_seq else 0.0


def caar(pred_seq: str, true_seq: str, contact_mask: np.ndarray) -> float:
    """Contact AAR: AAR restricted to paratope residues."""
    matches = sum(
        p == t for p, t, m in zip(pred_seq, true_seq, contact_mask) if m
    )
    n_contact = contact_mask.sum()
    return matches / n_contact if n_contact > 0 else 0.0


def perplexity(log_probs: np.ndarray) -> float:
    """Perplexity from per-residue log probabilities."""
    return float(np.exp(-log_probs.mean()))


# -- Group 2: Structure Quality --

def kabsch_rmsd(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """Ca RMSD after Kabsch alignment.

    Args:
        pred_coords: (N, 3) predicted Ca coordinates
        true_coords: (N, 3) native Ca coordinates
    """
    # Center
    pred_c = pred_coords - pred_coords.mean(axis=0)
    true_c = true_coords - true_coords.mean(axis=0)

    # Kabsch via SVD
    H = pred_c.T @ true_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S = np.diag([1, 1, d])
    R = Vt.T @ S @ U.T

    pred_aligned = pred_c @ R.T
    diff = pred_aligned - true_c
    return float(np.sqrt((diff ** 2).sum(axis=1).mean()))


def tm_score(pred_coords: np.ndarray, true_coords: np.ndarray) -> float:
    """TM-score (Zhang & Skolnick, 2004).

    Simplified implementation using Kabsch-aligned coordinates.
    For publication, use the official TMscore binary.
    """
    L = len(true_coords)
    if L == 0:
        return 0.0

    d0 = 1.24 * np.cbrt(L - 15) - 1.8
    d0 = max(d0, 0.5)

    # Align first
    pred_c = pred_coords - pred_coords.mean(axis=0)
    true_c = true_coords - true_coords.mean(axis=0)
    H = pred_c.T @ true_c
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1, 1, d]) @ U.T
    pred_aligned = pred_c @ R.T

    di = np.sqrt(((pred_aligned - true_c) ** 2).sum(axis=1))
    score = (1.0 / (1.0 + (di / d0) ** 2)).sum() / L
    return float(score)


# -- Group 3: Binding Interface Quality --

def compute_ca_contacts(ab_ca: np.ndarray, ag_ca: np.ndarray,
                        cutoff: float = 8.0) -> set[tuple[int, int]]:
    """Compute residue-level contacts from CA-CA distances.

    Args:
        ab_ca: (N_ab, 3) antibody CA coordinates
        ag_ca: (N_ag, 3) antigen CA coordinates
        cutoff: CA-CA distance threshold in Angstroms

    Returns:
        set of (ab_residue_idx, ag_residue_idx) pairs within cutoff
    """
    if len(ab_ca) == 0 or len(ag_ca) == 0:
        return set()
    tree = cKDTree(ag_ca)
    contacts = set()
    for i, coord in enumerate(ab_ca):
        for j in tree.query_ball_point(coord, cutoff):
            contacts.add((i, j))
    return contacts


def fnat(pred_contacts: set[tuple], true_contacts: set[tuple]) -> float:
    """Fraction of native contacts recovered."""
    if not true_contacts:
        return 0.0
    return len(pred_contacts & true_contacts) / len(true_contacts)


def interface_rmsd(pred_ab: np.ndarray, pred_ag: np.ndarray,
                   true_ab: np.ndarray, true_ag: np.ndarray,
                   contact_pairs: list[tuple[int, int]]) -> float:
    """Interface RMSD: RMSD of interface residue Ca atoms after alignment."""
    if not contact_pairs:
        return float("inf")

    ab_idx = sorted(set(p[0] for p in contact_pairs))
    ag_idx = sorted(set(p[1] for p in contact_pairs))

    pred_iface = np.concatenate([pred_ab[ab_idx], pred_ag[ag_idx]])
    true_iface = np.concatenate([true_ab[ab_idx], true_ag[ag_idx]])
    return kabsch_rmsd(pred_iface, true_iface)


def dockq_score(fnat_val: float, irmsd: float, lrmsd: float) -> float:
    """DockQ composite score (Basu & Wallner, 2016).

    DockQ = (fnat + 1/(1+(irmsd/1.5)^2) + 1/(1+(lrmsd/8.5)^2)) / 3
    """
    term1 = fnat_val
    term2 = 1.0 / (1.0 + (irmsd / 1.5) ** 2)
    term3 = 1.0 / (1.0 + (lrmsd / 8.5) ** 2)
    return (term1 + term2 + term3) / 3.0


# -- Group 4: Epitope Specificity --

def compute_epitope_mask(ab_ca: np.ndarray, ag_ca: np.ndarray,
                         cutoff: float = 8.0) -> np.ndarray:
    """Compute epitope mask: antigen residues contacted by antibody CA atoms.

    Args:
        ab_ca: (N_ab, 3) antibody CA coordinates (CDR-only for CDR-specific)
        ag_ca: (N_ag, 3) antigen CA coordinates
        cutoff: CA-CA distance threshold in Angstroms

    Returns:
        (N_ag,) boolean array -- True for epitope residues
    """
    mask = np.zeros(len(ag_ca), dtype=bool)
    if len(ab_ca) == 0 or len(ag_ca) == 0:
        return mask
    tree = cKDTree(ag_ca)
    for coord in ab_ca:
        for j in tree.query_ball_point(coord, cutoff):
            mask[j] = True
    return mask


def epitope_metrics(pred_ab_coords: np.ndarray, ag_coords: np.ndarray,
                    true_epitope_mask: np.ndarray,
                    cutoff: float = 8.0) -> dict[str, float]:
    """Compute epitope precision, recall, F1 using CA-CA contacts.

    For CDR-specific evaluation, pass CDR-only CA coords as pred_ab_coords
    and a CDR-specific true_epitope_mask (from compute_epitope_mask with
    CDR-only native coords).

    Args:
        pred_ab_coords: (N, 3) Ca coords (CDR-only for CDR-specific eval)
        ag_coords: (N_ag, 3) Ca coords of antigen
        true_epitope_mask: (N_ag,) binary mask of true epitope residues
        cutoff: CA-CA contact distance threshold (default 8.0 A)
    """
    if len(pred_ab_coords) == 0 or len(ag_coords) == 0:
        return {"epitope_precision": 0, "epitope_recall": 0, "epitope_f1": 0}

    pred_epi_mask = compute_epitope_mask(pred_ab_coords, ag_coords, cutoff)
    pred_set = set(np.where(pred_epi_mask)[0])
    true_epi = set(np.where(true_epitope_mask)[0])

    if not pred_set and not true_epi:
        return {"epitope_precision": 1.0, "epitope_recall": 1.0,
                "epitope_f1": 1.0}

    tp = len(pred_set & true_epi)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(true_epi) if true_epi else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"epitope_precision": precision, "epitope_recall": recall,
            "epitope_f1": f1}


# -- Group 6: Designability --

LIABILITY_PATTERNS = ["NG", "DG", "DS", "DD", "NS", "NT", "M"]


def count_liabilities(sequence: str) -> int:
    """Count known sequence liability motifs."""
    count = 0
    for pat in LIABILITY_PATTERNS:
        start = 0
        while True:
            idx = sequence.find(pat, start)
            if idx == -1:
                break
            count += 1
            start = idx + 1
    return count


# -- Group 7: Diversity --

def pairwise_diversity(sequences: list[str]) -> float:
    """Mean pairwise normalized edit distance among sequences."""
    n = len(sequences)
    if n < 2:
        return 0.0

    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            d = _edit_distance(sequences[i], sequences[j])
            max_len = max(len(sequences[i]), len(sequences[j]))
            total += d / max_len if max_len > 0 else 0
            pairs += 1
    return total / pairs


def _edit_distance(s1: str, s2: str) -> int:
    """Levenshtein distance."""
    if len(s1) < len(s2):
        return _edit_distance(s2, s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1,
                            prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


# -- Composite Scores --

def chimera_s(dockq: float, rmsd: float, tmscore: float) -> float:
    """CHIMERA-S: structural + interface quality composite."""
    rmsd_norm = max(0, 1 - rmsd / 10.0)
    return 0.35 * dockq + 0.25 * rmsd_norm + 0.25 * tmscore


def chimera_b(epitope_f1: float, fnat_val: float) -> float:
    """CHIMERA-B: binding specificity composite."""
    return 0.5 * epitope_f1 + 0.3 * fnat_val


# -- Group 5 (cont.): Structural Diversity --

def structural_diversity(coords_list: list[np.ndarray]) -> float:
    """Mean pairwise RMSD among a set of predicted structures.

    Args:
        coords_list: list of (N, 3) Ca coordinate arrays (same length)
    """
    n = len(coords_list)
    if n < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += kabsch_rmsd(coords_list[i], coords_list[j])
            pairs += 1
    return total / pairs


# -- Group 7 (cont.): Novelty --

def novelty_score(pred_seq: str, training_seqs: list[str]) -> float:
    """Minimum normalized edit distance from prediction to training set.

    Returns value in [0, 1]; higher = more novel.
    """
    if not training_seqs:
        return 1.0
    min_dist = float("inf")
    for ref in training_seqs:
        d = _edit_distance(pred_seq, ref)
        max_len = max(len(pred_seq), len(ref))
        norm_d = d / max_len if max_len > 0 else 0
        if norm_d < min_dist:
            min_dist = norm_d
    return min_dist


# -- Aggregation --

def mean_std(values: list[float]) -> dict:
    """Compute mean and standard deviation for a list of metric values.

    Args:
        values: per-complex metric values

    Returns:
        dict with keys: mean, std
    """
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def bootstrap_ci(values: list[float], n_bootstrap: int = 10000,
                 ci: float = 0.95) -> dict:
    """Compute mean with bootstrap confidence interval.

    Args:
        values: per-complex metric values
        n_bootstrap: number of bootstrap samples
        ci: confidence level (default 0.95)

    Returns:
        dict with keys: mean, ci_lower, ci_upper, std
    """
    values = np.array(values, dtype=float)
    if len(values) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    observed_mean = float(np.mean(values))
    rng = np.random.default_rng(42)
    bootstrap_means = np.array([
        np.mean(rng.choice(values, size=len(values), replace=True))
        for _ in range(n_bootstrap)
    ])

    alpha = 1 - ci
    ci_lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        "mean": observed_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std": float(np.std(values)),
    }
