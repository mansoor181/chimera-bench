"""Re-evaluate AbODE with fixed coordinate alignment.

This script re-runs inference using the fixed reconstruct_ca_coords that
aligns predictions to native PDB coordinates instead of AbODE's local frame.
"""

import argparse
import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model_function import Adobe_cond
from utils import (
    get_graph_data_polar_with_sidechains_angle,
    get_antibody_entries,
    _get_cartesian,
    ALPHABET,
)
from torchdiffeq import odeint

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, save_predictions, run_full_evaluation, FULL_METRIC_KEYS,
)

sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", ".."))
from benchmark.evaluation.metrics import aar as chimera_aar


def kabsch_align(mobile, target):
    """Kabsch alignment: align mobile to target."""
    centroid_m = mobile.mean(axis=0)
    centroid_t = target.mean(axis=0)
    m_centered = mobile - centroid_m
    t_centered = target - centroid_t

    H = m_centered.T @ t_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (mobile - centroid_m) @ R.T + centroid_t
    return aligned


def load_native_cdr_coords(cid, cdr_type, data_root):
    """Load native CDR CA coordinates from complex_features."""
    feat_path = os.path.join(data_root, "processed", "complex_features", f"{cid}.pt")
    if not os.path.exists(feat_path):
        return None

    feat = torch.load(feat_path, map_location="cpu", weights_only=False)
    cdr_idx = int(cdr_type) - 1  # H1=0, H2=1, H3=2
    cdr_masks = feat.get("cdr_masks", {}).get("imgt", {})
    heavy_mask = cdr_masks.get("heavy", [])

    if not heavy_mask:
        return None

    cdr_indices = np.array([i for i, v in enumerate(heavy_mask) if v == cdr_idx])
    if len(cdr_indices) == 0:
        return None

    heavy_ca = feat.get("heavy_ca_coords")
    if heavy_ca is None:
        return None

    return heavy_ca[cdr_indices]


def reconstruct_ca_coords_fixed(pred_tensor, gt_tensor, first_res, native_cdr_coords):
    """Reconstruct CA coordinates with alignment to native PDB frame."""
    pred_polar = pred_tensor[:, 20:29].cpu().detach()
    truth_polar = gt_tensor[:, 20:29].cpu().detach()
    first_residue_coord = first_res[:, 1, :].detach().numpy().reshape(-1, 3)

    Cart_pred, Cart_truth = _get_cartesian(
        pred_polar.view(-1, 9), truth_polar.view(-1, 9)
    )

    ca_pred = Cart_pred[:, 3:6].numpy()
    ca_truth_local = Cart_truth[:, 3:6].numpy()

    for i in range(len(ca_pred)):
        if i == 0:
            ca_pred[i] = ca_pred[i] + first_residue_coord
            ca_truth_local[i] = ca_truth_local[i] + first_residue_coord
        else:
            ca_pred[i] = ca_pred[i] + ca_pred[i - 1]
            ca_truth_local[i] = ca_truth_local[i] + ca_truth_local[i - 1]

    # Align to native and use native as truth
    if native_cdr_coords is not None and len(native_cdr_coords) == len(ca_pred):
        ca_pred = kabsch_align(ca_pred, native_cdr_coords)
        ca_truth = native_cdr_coords.copy()
    else:
        ca_truth = ca_truth_local

    return ca_pred, ca_truth


def decode_sequence(pred_tensor):
    """Decode predicted sequence from AbODE 20D logits."""
    labels = pred_tensor[:, :20].detach().cpu()
    _, tags = torch.max(torch.log_softmax(labels, dim=1), dim=1)
    return "".join(ALPHABET[t] for t in tags)


def load_data_with_cids(cdr_type, json_path):
    """Load AbODE graph data and extract parallel complex_id list."""
    with open(json_path, "r") as f:
        entries = json.load(f)

    surviving_cids = []
    for entry in entries:
        ab_seq = entry["ab_seq"]
        ag_seq = entry["ag_seq"]
        antibody_cdr_len = len(ab_seq) - 2

        if antibody_cdr_len <= 1:
            continue
        if "*" in ag_seq:
            continue
        surviving_cids.append(entry["complex_id"])

    graph_data = get_graph_data_polar_with_sidechains_angle(int(cdr_type), json_path, 0)

    if len(graph_data) != len(surviving_cids):
        final_cids = []
        idx = 0
        for entry in entries:
            ab_seq = entry["ab_seq"]
            ag_seq = entry["ag_seq"]
            antibody_cdr_len = len(ab_seq) - 2
            if antibody_cdr_len <= 1:
                continue
            if "*" in ag_seq:
                continue
            if idx < len(graph_data):
                final_cids.append(entry["complex_id"])
                idx += 1
            if idx >= len(graph_data):
                break
        surviving_cids = final_cids

    return graph_data, surviving_cids


def run_inference(model, data_list, cid_list, device, cdr_type, data_root, t_nsamples=200):
    """Run inference with fixed coordinate alignment."""
    model.eval()
    t_space = np.linspace(0, 1, t_nsamples)
    predictions = []
    cdr_label = f"H{cdr_type}"

    with torch.no_grad():
        for batch, cid in tqdm(zip(data_list, cid_list), desc=f"Inference {cdr_label}",
                               total=len(data_list)):
            data = batch.x.to(device)
            t = torch.tensor(t_space).to(device)

            params_list = [
                batch.edge_index.to(device),
                batch.order.to(device),
                batch.a_index.to(device),
            ]
            model.update_param(params_list)

            options = {"dtype": torch.float64}
            y_pred = odeint(
                model, data, t,
                method="adaptive_heun", rtol=0.5, atol=0.5,
                options=options,
            )

            y_gt = batch.y.to(device)
            antibody_len = [batch.ab_len.item()]
            antigen_len = [batch.ag_len.item()]
            final_pred = get_antibody_entries(
                y_pred[-1], torch.zeros(len(data), dtype=torch.long, device=device),
                antibody_len, antigen_len,
            )

            pred_seq = decode_sequence(final_pred)
            true_seq = decode_sequence(y_gt)

            # Load native coords for alignment
            native_cdr_coords = load_native_cdr_coords(cid, cdr_type, data_root)

            try:
                pred_ca, true_ca = reconstruct_ca_coords_fixed(
                    final_pred, y_gt, batch.first_res, native_cdr_coords)
            except Exception as e:
                print(f"Warning: coord reconstruction failed for {cid}: {e}")
                pred_ca = np.zeros((len(pred_seq), 3))
                true_ca = np.zeros((len(true_seq), 3))

            # Compute PPL
            pred_logits = final_pred[:, :20].detach().cpu()
            true_labels = y_gt[:, :20].detach().cpu()
            ce_loss = torch.nn.functional.cross_entropy(
                pred_logits, true_labels, reduction="mean"
            )
            ppl_val = float(torch.exp(ce_loss).item())

            pred_dict = {
                "complex_id": cid,
                "cdr_type": cdr_label,
                "pred_sequence": pred_seq,
                "true_sequence": true_seq,
                "pred_coords": pred_ca,
                "true_coords": true_ca,
                "ppl": ppl_val,
            }
            predictions.append(pred_dict)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cdr_type", type=str, required=True, choices=["1", "2", "3"])
    parser.add_argument("--split", type=str, default="epitope_group")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    shared = load_shared_config()
    data_root = shared["paths"]["data_root"]
    results_root = shared["paths"]["results_root"]
    trans_dir = os.path.join(shared["paths"]["trans_baselines"], "abode")

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    run_name = f"abode_cdr{args.cdr_type}_{args.split}"
    ckpt_path = os.path.join(results_root, "abode", run_name, "checkpoints", "best.pt")
    test_json = os.path.join(trans_dir, args.split, f"cdrh{args.cdr_type}", "test.json")

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # Load model
    model = Adobe_cond(30, 29).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    print(f"Loaded checkpoint from {ckpt_path}")

    # Load test data
    test_data, test_cids = load_data_with_cids(args.cdr_type, test_json)
    print(f"Loaded {len(test_data)} test complexes")

    # Run inference
    predictions = run_inference(model, test_data, test_cids, device,
                                args.cdr_type, data_root)

    # Save predictions
    pred_dir = os.path.join(results_root, "abode", run_name, "predictions_fixed")
    save_predictions(predictions, pred_dir)
    print(f"Saved {len(predictions)} predictions to {pred_dir}")

    # Run full evaluation
    cdr_label = f"H{args.cdr_type}"
    per_complex, summary, _ = run_full_evaluation(
        pred_dir, args.split, data_root, cdr_type_hint=cdr_label,
        numbering_scheme="imgt"
    )

    # Print results
    print(f"\n=== AbODE {cdr_label} Results (Fixed) ===")
    for k in FULL_METRIC_KEYS:
        if k in summary:
            print(f"  {k}: {summary[k]['mean']:.3f} +/- {summary[k]['std']:.3f}")


if __name__ == "__main__":
    main()
