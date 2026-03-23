"""Unified evaluation entry point for CHIMERA-Bench.

Usage:
    python -m evaluation.evaluate \
        --predictions /path/to/predictions/ \
        --split epitope_group \
        --cdr-type H3 \
        --output results.json

Interface metrics (fnat, irmsd, dockq) and epitope metrics use CDR-specific
CA-CA 8A contacts when --cdr-type is specified: only contacts where the
antibody partner is a CDR residue are counted, because framework contacts
are trivially preserved by construction and would dominate the metrics.

Predictions directory should contain one .pt or .json file per complex,
each with keys: complex_id, pred_sequence, pred_coords (N,3), etc.
"""

import argparse
import json
import logging
import pickle
from pathlib import Path
import torch

import numpy as np

from config import Config
from evaluation.metrics import (
    aar, caar, kabsch_rmsd, tm_score, fnat, interface_rmsd, dockq_score,
    epitope_metrics, compute_epitope_mask, count_liabilities,
    pairwise_diversity, structural_diversity, novelty_score, bootstrap_ci,
    chimera_s, chimera_b, compute_ca_contacts,
)

log = logging.getLogger(__name__)


class DesignResult:
    """A single design prediction for one complex."""

    __slots__ = ("complex_id", "cdr_type", "pred_sequence", "pred_coords",
                 "log_probs", "pred_ab_full_coords")

    def __init__(self, complex_id: str, cdr_type: str = "all",
                 pred_sequence: str = "", pred_coords: np.ndarray = None,
                 log_probs: np.ndarray = None,
                 pred_ab_full_coords: np.ndarray = None):
        self.complex_id = complex_id
        self.cdr_type = cdr_type
        self.pred_sequence = pred_sequence
        self.pred_coords = pred_coords
        self.log_probs = log_probs
        self.pred_ab_full_coords = pred_ab_full_coords


class NativeComplex:
    """Ground truth for one complex."""

    __slots__ = ("complex_id", "true_sequence", "true_coords", "ag_coords",
                 "epitope_mask", "contact_mask", "contact_pairs",
                 "ab_full_coords", "cdr_indices")

    def __init__(self, complex_id: str, true_sequence: str,
                 true_coords: np.ndarray, ag_coords: np.ndarray,
                 epitope_mask: np.ndarray, contact_mask: np.ndarray,
                 contact_pairs: set, ab_full_coords: np.ndarray = None,
                 cdr_indices: np.ndarray = None):
        self.complex_id = complex_id
        self.true_sequence = true_sequence
        self.true_coords = true_coords
        self.ag_coords = ag_coords
        self.epitope_mask = epitope_mask
        self.contact_mask = contact_mask
        self.contact_pairs = contact_pairs
        self.ab_full_coords = ab_full_coords
        self.cdr_indices = cdr_indices


def evaluate_single(design: DesignResult, native: NativeComplex) -> dict:
    """Evaluate a single design against its native complex."""
    results = {"complex_id": design.complex_id}

    # Group 1: Sequence
    results["aar"] = aar(design.pred_sequence, native.true_sequence)
    results["caar"] = caar(design.pred_sequence, native.true_sequence,
                           native.contact_mask)
    results["n_liabilities"] = count_liabilities(design.pred_sequence)

    # Group 2: Structure
    if design.pred_coords is not None and native.true_coords is not None:
        results["rmsd"] = kabsch_rmsd(design.pred_coords, native.true_coords)
        results["tm_score"] = tm_score(design.pred_coords, native.true_coords)

    # CDR-specific filtering: only count contacts where antibody partner
    # is a CDR residue. Framework contacts are trivially preserved by
    # construct_pred_ab_full_coords and would dominate the metrics.
    cdr_set = set(native.cdr_indices.tolist()) if native.cdr_indices is not None else None

    # Group 4: Epitope specificity (CDR-specific when cdr_indices available)
    if design.pred_ab_full_coords is not None:
        if cdr_set is not None:
            # CDR-specific: use CDR-only CA coords for epitope
            pred_cdr_ca = design.pred_ab_full_coords[native.cdr_indices]
            epi = epitope_metrics(pred_cdr_ca, native.ag_coords,
                                  native.epitope_mask)
        else:
            epi = epitope_metrics(design.pred_ab_full_coords,
                                  native.ag_coords, native.epitope_mask)
        results.update(epi)

    # Group 3: Interface (CDR-specific contacts when cdr_indices available)
    if design.pred_ab_full_coords is not None and native.ab_full_coords is not None:
        all_pred_contacts = compute_ca_contacts(
            design.pred_ab_full_coords, native.ag_coords, cutoff=8.0)

        if cdr_set is not None:
            # Filter to CDR-specific contacts
            pred_contacts = {(i, j) for i, j in all_pred_contacts if i in cdr_set}
            true_contacts = native.contact_pairs
        else:
            pred_contacts = all_pred_contacts
            true_contacts = native.contact_pairs

        fnat_val = fnat(pred_contacts, true_contacts)
        results["fnat"] = fnat_val

        irmsd = interface_rmsd(
            design.pred_ab_full_coords, native.ag_coords,
            native.ab_full_coords, native.ag_coords,
            list(true_contacts),
        )
        results["irmsd"] = irmsd

        lrmsd = kabsch_rmsd(design.pred_ab_full_coords, native.ab_full_coords)
        results["lrmsd"] = lrmsd
        results["dockq"] = dockq_score(fnat_val, irmsd, lrmsd)

    # Composites
    results["chimera_s"] = chimera_s(
        dockq=results.get("dockq", 0),
        rmsd=results.get("rmsd", 10),
        tmscore=results.get("tm_score", 0),
    )
    results["chimera_b"] = chimera_b(
        epitope_f1=results.get("epitope_f1", 0),
        fnat_val=results.get("fnat", 0),
    )
    return results


def evaluate_designs(designs: list[DesignResult], native: NativeComplex,
                     training_seqs: list[str] = None) -> dict:
    """Evaluate K designs for one complex. Report best/median/diversity."""
    per_design = [evaluate_single(d, native) for d in designs]

    result = {"complex_id": native.complex_id, "n_designs": len(designs)}

    metric_keys = ["aar", "caar", "rmsd", "tm_score", "fnat", "dockq",
                   "irmsd", "epitope_f1", "chimera_s", "chimera_b"]

    for key in metric_keys:
        vals = [r[key] for r in per_design if key in r]
        if not vals:
            continue
        if key in ("rmsd", "irmsd", "lrmsd"):
            result[f"best_{key}"] = min(vals)
        else:
            result[f"best_{key}"] = max(vals)
        result[f"median_{key}"] = float(np.median(vals))

    # Sequence diversity
    seqs = [d.pred_sequence for d in designs if d.pred_sequence]
    if len(seqs) > 1:
        result["seq_diversity"] = pairwise_diversity(seqs)

    # Structural diversity
    coord_list = [d.pred_coords for d in designs
                  if d.pred_coords is not None]
    if len(coord_list) > 1 and all(len(c) == len(coord_list[0]) for c in coord_list):
        result["struct_diversity"] = structural_diversity(coord_list)

    # Novelty
    if training_seqs and seqs:
        nov = [novelty_score(s, training_seqs) for s in seqs]
        result["novelty_mean"] = float(np.mean(nov))

    # Liability count
    liab = [r["n_liabilities"] for r in per_design if "n_liabilities" in r]
    if liab:
        result["mean_liabilities"] = float(np.mean(liab))

    return result


def evaluate_benchmark(all_designs: dict[str, list[DesignResult]],
                       natives: dict[str, NativeComplex],
                       training_seqs: list[str] = None,
                       output_path: Path = None) -> dict:
    """Evaluate all complexes in a benchmark split.

    Args:
        all_designs: {complex_id: [DesignResult, ...]}
        natives: {complex_id: NativeComplex}
        training_seqs: optional training CDR sequences for novelty
        output_path: optional path to save results JSON

    Returns:
        dict with per_complex results and summary with bootstrap CIs
    """
    results = []
    for cid, designs in all_designs.items():
        if cid not in natives:
            log.warning("No native found for %s, skipping", cid)
            continue
        result = evaluate_designs(designs, natives[cid], training_seqs)
        results.append(result)

    # Summary with bootstrap CIs
    summary = {}
    ci_keys = ["best_aar", "best_caar", "best_rmsd", "best_tm_score",
               "best_fnat", "best_dockq", "best_epitope_f1",
               "best_chimera_s", "best_chimera_b",
               "median_aar", "median_rmsd",
               "seq_diversity", "struct_diversity",
               "novelty_mean", "mean_liabilities"]

    for key in ci_keys:
        vals = [r[key] for r in results if key in r]
        if vals:
            summary[key] = bootstrap_ci(vals)

    log.info("Evaluation summary (%d complexes):", len(results))
    for k, v in summary.items():
        log.info("  %s: %.4f [%.4f, %.4f]",
                 k, v["mean"], v["ci_lower"], v["ci_upper"])

    output = {"per_complex": results, "summary": summary}
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2, default=str)
        log.info("Saved results to %s", output_path)

    return output


# -- CLI --

def _load_predictions(pred_dir: Path) -> dict[str, list[DesignResult]]:
    """Load prediction files from a directory."""
    
    all_designs = {}
    for f in sorted(pred_dir.glob("*.pt")):
        data = torch.load(f, map_location="cpu", weights_only=False)
        cid = data.get("complex_id", f.stem)
        design = DesignResult(
            complex_id=cid,
            pred_sequence=data.get("pred_sequence", ""),
            pred_coords=data.get("pred_coords"),
            pred_ab_full_coords=data.get("pred_ab_full_coords"),
        )
        if isinstance(design.pred_coords, np.ndarray) is False and design.pred_coords is not None:
            design.pred_coords = np.array(design.pred_coords)
        all_designs.setdefault(cid, []).append(design)
    return all_designs


def _get_cdr_indices(feat, cdr_type, scheme="imgt"):
    """Get CDR indices from complex_features cdr_masks."""
    _CDR_LABEL_TO_CODE = {"H1": 0, "H2": 1, "H3": 2, "L1": 3, "L2": 4, "L3": 5}
    masks = feat["cdr_masks"][scheme]
    h_mask = np.array(masks["heavy"])
    l_mask = np.array(masks["light"])
    h_ca_len = len(feat["heavy_ca_coords"])

    code = _CDR_LABEL_TO_CODE.get(cdr_type)
    if code is None:
        return None
    if code <= 2:
        return np.where(h_mask == code)[0]
    else:
        return np.where(l_mask == code)[0] + h_ca_len


def _load_natives(cfg: Config, split_name: str,
                  split_type: str = "test",
                  cdr_type: str = None,
                  numbering_scheme: str = "imgt") -> dict[str, NativeComplex]:
    """Load native complexes from annotations + split file.

    When cdr_type is specified, contacts and epitope masks are CDR-specific:
    only contacts where the antibody partner is a CDR residue are included.
    This prevents framework contacts (trivially preserved by construction)
    from dominating the interface metrics.
    """

    # Load split
    split_path = cfg.splits_dir / f"{split_name}.json"
    with open(split_path) as f:
        split = json.load(f)
    test_ids = set(split.get(split_type, []))

    # Load annotations
    with open(cfg.processed_dir / "annotations.pkl", "rb") as f:
        annotations = pickle.load(f)
    ann_map = {a.complex_id: a for a in annotations}

    natives = {}
    feat_dir = cfg.processed_dir / "complex_features"
    for cid in test_ids:
        ann = ann_map.get(cid)
        if ann is None:
            continue

        # Load features for coords
        feat_path = feat_dir / f"{cid}.pt"
        if not feat_path.exists():
            continue
        feat = torch.load(feat_path, map_location="cpu", weights_only=False)

        h_ca = feat.get("heavy_ca_coords", np.zeros((0, 3)))
        l_ca = feat.get("light_ca_coords", np.zeros((0, 3)))
        ag_ca = feat.get("antigen_ca_coords", np.zeros((0, 3)))
        ab_ca = np.concatenate([h_ca, l_ca], axis=0) if len(h_ca) + len(l_ca) > 0 else np.zeros((0, 3))

        # Get CDR indices if cdr_type specified
        cdr_indices = None
        if cdr_type is not None:
            cdr_indices = _get_cdr_indices(feat, cdr_type, numbering_scheme)

        # Compute contacts using symmetric CA-CA 8A definition
        all_contacts = compute_ca_contacts(ab_ca, ag_ca, cutoff=8.0)

        # CDR-specific filtering: only contacts where ab partner is CDR residue
        if cdr_indices is not None and len(cdr_indices) > 0:
            cdr_set = set(cdr_indices.tolist())
            contact_set = {(i, j) for i, j in all_contacts if i in cdr_set}
            # CDR-specific epitope: only ag residues contacted by CDR
            epi_mask = compute_epitope_mask(ab_ca[cdr_indices], ag_ca, cutoff=8.0)
        else:
            contact_set = all_contacts
            epi_mask = np.zeros(len(ag_ca), dtype=bool)
            for _, ag_j in contact_set:
                epi_mask[ag_j] = True

        para_mask = np.zeros(len(ab_ca), dtype=bool)
        for ab_i, _ in contact_set:
            para_mask[ab_i] = True

        # Use sequences for AAR
        h_seq = ann.sequences.get("heavy", "")
        l_seq = ann.sequences.get("light", "")
        full_seq = h_seq + l_seq

        natives[cid] = NativeComplex(
            complex_id=cid,
            true_sequence=full_seq,
            true_coords=ab_ca,
            ag_coords=ag_ca,
            epitope_mask=epi_mask,
            contact_mask=para_mask,
            contact_pairs=contact_set,
            ab_full_coords=ab_ca,
            cdr_indices=cdr_indices,
        )

    return natives


def main():
    parser = argparse.ArgumentParser(
        description="CHIMERA-Bench evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--predictions", type=Path, required=True,
                        help="Directory with prediction .pt files")
    parser.add_argument("--split", default="epitope_group",
                        choices=["epitope_group", "antigen_fold", "temporal"],
                        help="Split type to evaluate on")
    parser.add_argument("--cdr-type", default=None,
                        choices=["H1", "H2", "H3", "L1", "L2", "L3"],
                        help="CDR type for CDR-specific evaluation")
    parser.add_argument("--numbering-scheme", default="imgt",
                        choices=["imgt", "chothia"],
                        help="Numbering scheme for CDR definition")
    parser.add_argument("--output", type=Path, default=None,
                        help="Output JSON path (default: predictions/results.json)")
    parser.add_argument("--data-root", type=Path, default=None,
                        help="Override data root directory")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    cfg = Config()
    if args.data_root:
        cfg.data_root = args.data_root

    log.info("Loading predictions from %s", args.predictions)
    all_designs = _load_predictions(args.predictions)
    log.info("Loaded %d complexes with %d total designs",
             len(all_designs), sum(len(v) for v in all_designs.values()))

    log.info("Loading natives for split=%s cdr_type=%s", args.split, args.cdr_type)
    natives = _load_natives(cfg, args.split, cdr_type=args.cdr_type,
                            numbering_scheme=args.numbering_scheme)
    log.info("Loaded %d native complexes", len(natives))

    output_path = args.output or args.predictions / "results.json"
    evaluate_benchmark(all_designs, natives, output_path=output_path)


if __name__ == "__main__":
    main()
