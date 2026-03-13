"""Unified evaluation entry point for CHIMERA-Bench.

Usage:
    python -m benchmark.evaluation.evaluate \
        --predictions /path/to/predictions/ \
        --split epitope_group \
        --output results.json

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
from scipy.spatial import cKDTree

from benchmark.config import Config
from benchmark.evaluation.metrics import (
    aar, caar, kabsch_rmsd, tm_score, fnat, interface_rmsd, dockq_score,
    epitope_metrics, count_liabilities, pairwise_diversity,
    structural_diversity, novelty_score, bootstrap_ci,
    chimera_s, chimera_b,
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
                 "ab_full_coords")

    def __init__(self, complex_id: str, true_sequence: str,
                 true_coords: np.ndarray, ag_coords: np.ndarray,
                 epitope_mask: np.ndarray, contact_mask: np.ndarray,
                 contact_pairs: set, ab_full_coords: np.ndarray = None):
        self.complex_id = complex_id
        self.true_sequence = true_sequence
        self.true_coords = true_coords
        self.ag_coords = ag_coords
        self.epitope_mask = epitope_mask
        self.contact_mask = contact_mask
        self.contact_pairs = contact_pairs
        self.ab_full_coords = ab_full_coords


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

    # Group 4: Epitope specificity
    if design.pred_ab_full_coords is not None:
        epi = epitope_metrics(
            design.pred_ab_full_coords, native.ag_coords,
            native.epitope_mask,
        )
        results.update(epi)

    # Group 3: Interface (requires full Ab coords)
    if design.pred_ab_full_coords is not None and native.ab_full_coords is not None:
        tree = cKDTree(native.ag_coords)
        pred_contacts = set()
        for i, coord in enumerate(design.pred_ab_full_coords):
            nearby = tree.query_ball_point(coord, 4.5)
            for j in nearby:
                pred_contacts.add((i, j))

        fnat_val = fnat(pred_contacts, native.contact_pairs)
        results["fnat"] = fnat_val

        irmsd = interface_rmsd(
            design.pred_ab_full_coords, native.ag_coords,
            native.ab_full_coords, native.ag_coords,
            list(native.contact_pairs),
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


def _load_natives(cfg: Config, split_name: str,
                  split_type: str = "test") -> dict[str, NativeComplex]:
    """Load native complexes from annotations + split file."""

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

        # Compute epitope/paratope masks and contact pairs from atom14 coords
        h_a14 = feat.get("heavy_atom14_coords", np.zeros((0, 14, 3)))
        h_m14 = feat.get("heavy_atom14_mask", np.zeros((0, 14), dtype=bool))
        l_a14 = feat.get("light_atom14_coords", np.zeros((0, 14, 3)))
        l_m14 = feat.get("light_atom14_mask", np.zeros((0, 14), dtype=bool))
        ab_a14 = np.concatenate([h_a14, l_a14], axis=0)
        ab_m14 = np.concatenate([h_m14, l_m14], axis=0)
        ag_a14 = feat.get("antigen_atom14_coords", np.zeros((0, 14, 3)))
        ag_m14 = feat.get("antigen_atom14_mask", np.zeros((0, 14), dtype=bool))

        n_ab = len(ab_a14)
        n_ag = len(ag_a14)
        epi_mask = np.zeros(n_ag, dtype=bool)
        para_mask = np.zeros(n_ab, dtype=bool)
        contact_set = set()

        # Build KD-tree from valid antigen atoms
        ag_atoms_list = []
        ag_res_indices = []
        for i in range(n_ag):
            valid = ag_m14[i]
            if valid.any():
                ag_atoms_list.append(ag_a14[i][valid])
                ag_res_indices.extend([i] * int(valid.sum()))
        if ag_atoms_list:
            ag_atoms_all = np.concatenate(ag_atoms_list, axis=0)
            ag_res_indices = np.array(ag_res_indices)
            ag_tree = cKDTree(ag_atoms_all)
            for ab_i in range(n_ab):
                valid = ab_m14[ab_i]
                if not valid.any():
                    continue
                ab_res_atoms = ab_a14[ab_i][valid]
                for atom_coord in ab_res_atoms:
                    nearby = ag_tree.query_ball_point(atom_coord, 4.5)
                    if nearby:
                        para_mask[ab_i] = True
                        for idx in nearby:
                            ag_j = ag_res_indices[idx]
                            epi_mask[ag_j] = True
                            contact_set.add((ab_i, ag_j))

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

    log.info("Loading natives for split=%s", args.split)
    natives = _load_natives(cfg, args.split)
    log.info("Loaded %d native complexes", len(natives))

    output_path = args.output or args.predictions / "results.json"
    evaluate_benchmark(all_designs, natives, output_path=output_path)


if __name__ == "__main__":
    main()
