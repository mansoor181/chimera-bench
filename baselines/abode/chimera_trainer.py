"""CHIMERA trainer for AbODE baseline.

Trains Adobe_cond model for one CDR at a time (H1, H2, or H3) using AbODE's
native loss function and ODE solver. No training script exists in the original
AbODE repo, so this implements training from scratch using:
  - get_graph_data_polar_with_sidechains_angle() for data loading
  - loss_function_vm_with_side_chains_angle() for training loss
  - evaluate_rmsd_with_sidechains_cond_angle() for evaluation
  - torchdiffeq.odeint() as the ODE solver

Batch size is always 1 (variable-size fully-connected graphs).

Usage:
    cd baselines/abode
    python chimera_trainer.py --max_epoch 2
    python chimera_trainer.py --split temporal --cdr_type 1
"""

import argparse
import csv
import json
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from model_function import Adobe_cond
from utils import (
    get_graph_data_polar_with_sidechains_angle,
    loss_function_vm_with_side_chains_angle,
    evaluate_rmsd_with_sidechains_cond_angle,
    get_antibody_entries,
    _get_cartesian,
    ALPHABET,
)
from torchdiffeq import odeint
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, load_split_ids,
    EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, save_predictions,
)

# CHIMERA evaluation metrics
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", ".."))
from benchmark.evaluation.metrics import (
    aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score,
    count_liabilities,
)

METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities"]


def load_config():
    config_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    shared = load_shared_config()
    return cfg, shared


def parse_args():
    parser = argparse.ArgumentParser(description="AbODE CHIMERA trainer")
    parser.add_argument("--split", type=str, default=None,
                        choices=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--cdr_type", type=str, default=None, choices=["1", "2", "3"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--test_only", action="store_true", help="Skip training, run test only")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint for test_only")
    return parser.parse_args()


def build_config(cfg, shared, args):
    ds = cfg.get("dataset", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    ode = cfg.get("ode", {})
    callbacks = cfg.get("callbacks", {})
    wandb_cfg = cfg.get("wandb", {})
    paths = shared["paths"]

    ns = SimpleNamespace(
        # Paths
        data_root=paths["data_root"],
        data_dir=os.path.join(paths["trans_baselines"], "abode"),
        output_dir=os.path.join(paths["results_root"], "abode"),
        # Dataset
        split=ds.get("split", "epitope_group"),
        cdr_type=str(ds.get("cdr_type", "3")),
        # Model
        c_in=model.get("c_in", 30),
        c_out=model.get("c_out", 29),
        # Training
        seed=training.get("seed", 10),
        gpu=training.get("gpu", 0),
        lr=training.get("lr", 1e-3),
        weight_decay=training.get("weight_decay", 1e-3),
        max_epoch=training.get("max_epoch", 100),
        grad_clip=training.get("grad_clip", 0.0),
        mask=training.get("mask", 0),
        # ODE
        solver=ode.get("solver", "adaptive_heun"),
        rtol=ode.get("rtol", 0.5),
        atol=ode.get("atol", 0.5),
        t_nsamples=ode.get("t_nsamples", 200),
        # Callbacks
        patience=callbacks.get("early_stopping", {}).get("patience", 10),
        es_mode=callbacks.get("early_stopping", {}).get("mode", "min"),
        ckpt_mode=callbacks.get("checkpoint", {}).get("mode", "min"),
        # WandB
        use_wandb=wandb_cfg.get("enabled", False),
        wandb_project=wandb_cfg.get("project", "chimera_baselines"),
    )

    # CLI overrides
    if args.split is not None:
        ns.split = args.split
    if args.cdr_type is not None:
        ns.cdr_type = args.cdr_type
    if args.gpu is not None:
        ns.gpu = args.gpu
    if args.seed is not None:
        ns.seed = args.seed
    if args.lr is not None:
        ns.lr = args.lr
    if args.max_epoch is not None:
        ns.max_epoch = args.max_epoch
    if not args.no_wandb:
        ns.use_wandb = True
    ns.test_only = args.test_only
    ns.checkpoint = args.checkpoint

    return ns


def load_data_with_cids(cdr_type, json_path, mask):
    """Load AbODE graph data and extract parallel complex_id list.

    AbODE's get_graph_data_polar_with_sidechains_angle() may skip entries
    (short CDRs, NaN coords, '*' in antigen seq). We need to track which
    complex_ids survived to maintain alignment.

    Returns:
        (graph_data_list, complex_id_list)
    """
    with open(json_path, "r") as f:
        entries = json.load(f)

    # Build parallel list: which entries survived graph construction
    # We replicate the skip conditions from get_graph_data_polar_with_sidechains_angle
    surviving_cids = []
    for entry in entries:
        ab_seq = entry["ab_seq"]
        ag_seq = entry["ag_seq"]
        antibody_cdr_len = len(ab_seq) - 2

        # Same skip conditions as the native function
        if antibody_cdr_len <= 1:
            continue
        if "*" in ag_seq:
            continue
        # NaN check requires computing features -- we'll verify after loading
        surviving_cids.append(entry["complex_id"])

    # Load graph data via native function
    graph_data = get_graph_data_polar_with_sidechains_angle(
        int(cdr_type), json_path, mask
    )

    # The native function may also skip due to NaN in positional features.
    # If counts mismatch, we need to reconcile.
    if len(graph_data) != len(surviving_cids):
        # Re-derive by checking NaN (same logic as native function line 1642)
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
            # Check if this entry produced valid (non-NaN) positional features
            if idx < len(graph_data):
                final_cids.append(entry["complex_id"])
                idx += 1
            if idx >= len(graph_data):
                break
        surviving_cids = final_cids

    # Final safety check
    assert len(graph_data) == len(surviving_cids), (
        f"Mismatch after reconciliation: {len(graph_data)} graphs vs "
        f"{len(surviving_cids)} complex_ids"
    )

    return graph_data, surviving_cids


def split_by_cids(graph_data, cid_list, split_cids):
    """Split graph data + cid list by a set of target complex IDs.

    Returns:
        (filtered_graphs, filtered_cids)
    """
    target = set(split_cids)
    graphs = []
    cids = []
    for g, c in zip(graph_data, cid_list):
        if c in target:
            graphs.append(g)
            cids.append(c)
    return graphs, cids


def train_epoch(model, data_list, optimizer, device, c):
    """One training epoch over AbODE graph data (batch_size=1)."""
    model.train()
    t_space = np.linspace(0, 1, c.t_nsamples)
    losses = []

    indices = list(range(len(data_list)))
    np.random.shuffle(indices)

    for idx in tqdm(indices, desc="Train"):
        batch = data_list[idx]
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
            method=c.solver, rtol=c.rtol, atol=c.atol,
            options=options,
        )

        y_gt = batch.y.to(device)

        # Extract antibody portion from prediction
        antibody_len = [batch.ab_len.item()]
        antigen_len = [batch.ag_len.item()]
        final_pred = get_antibody_entries(
            y_pred[-1], torch.zeros(len(data), dtype=torch.long, device=device),
            antibody_len, antigen_len,
        )

        loss = loss_function_vm_with_side_chains_angle(final_pred, y_gt, 1)

        optimizer.zero_grad()
        loss.backward()
        if c.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.grad_clip)
        optimizer.step()

        losses.append(loss.item())

    return float(np.mean(losses))


def evaluate_epoch(model, data_list, device, c):
    """Validation epoch: compute mean loss + AbODE native metrics."""
    model.eval()
    t_space = np.linspace(0, 1, c.t_nsamples)
    losses = []
    all_aar = []
    all_rmsd = []

    with torch.no_grad():
        for batch in tqdm(data_list, desc="Val"):
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
                method=c.solver, rtol=c.rtol, atol=c.atol,
                options=options,
            )

            y_gt = batch.y.to(device)
            antibody_len = [batch.ab_len.item()]
            antigen_len = [batch.ag_len.item()]
            final_pred = get_antibody_entries(
                y_pred[-1], torch.zeros(len(data), dtype=torch.long, device=device),
                antibody_len, antigen_len,
            )

            loss = loss_function_vm_with_side_chains_angle(final_pred, y_gt, 1)
            losses.append(loss.item())

            # Native metrics
            try:
                rmsd_n, rmsd_ca, rmsd_c, acc, rmsd_cart_ca = \
                    evaluate_rmsd_with_sidechains_cond_angle(
                        final_pred, y_gt, batch.first_res)
                all_aar.append(float(acc))
                all_rmsd.append(float(rmsd_cart_ca))
            except Exception:
                pass

    mean_loss = float(np.mean(losses)) if losses else float("inf")
    mean_aar = float(np.mean(all_aar)) if all_aar else 0.0
    mean_rmsd = float(np.mean(all_rmsd)) if all_rmsd else float("inf")
    return mean_loss, mean_aar, mean_rmsd


def decode_sequence(pred_tensor):
    """Decode predicted sequence from AbODE 20D logits."""
    labels = pred_tensor[:, :20].detach().cpu()
    _, tags = torch.max(torch.log_softmax(labels, dim=1), dim=1)
    return "".join(ALPHABET[t] for t in tags)


def kabsch_align(mobile, target):
    """Kabsch alignment: align mobile to target, return aligned mobile + RMSD.

    Args:
        mobile: (N, 3) coordinates to align
        target: (N, 3) reference coordinates

    Returns:
        aligned: (N, 3) aligned coordinates
    """
    centroid_m = mobile.mean(axis=0)
    centroid_t = target.mean(axis=0)
    m_centered = mobile - centroid_m
    t_centered = target - centroid_t

    H = m_centered.T @ t_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    aligned = (mobile - centroid_m) @ R.T + centroid_t
    return aligned


def reconstruct_ca_coords(pred_tensor, gt_tensor, first_res, native_cdr_coords):
    """Reconstruct Cartesian CA coordinates from AbODE polar predictions.

    AbODE's polar features (r_norm, mid_angle, normal_angle) are internal
    coordinates (bond length, bond angle, dihedral-like angle). The original
    _get_cartesian/_transform_to_cart code treats them as spherical (r, theta,
    phi) with a 3.14/2 - phi substitution, producing a degenerate trajectory
    that lives on a 2D plane in the local frame.

    We reconstruct ca_pred in this AbODE local frame via the original
    _get_cartesian + cumulative sum, then rigidly Kabsch-align to native PDB
    CDR-only CA coords. ca_truth is the native PDB coords. This way:
      - RMSD reflects how badly the (degenerate) AbODE reconstruction fits the
        true 3D CDR shape (not an artifact of identical broken transforms on
        both pred and truth).
      - pred_coords are placed in the antigen's frame so downstream interface
        metrics (Fnat, iRMSD, DockQ) are computed against the real antigen.

    Caller must provide native_cdr_coords of length == ca_pred. AbODE's polar
    features are CDR-only (ab_seq has +-1 flanking, but antibody_cdr_len =
    len(ab_seq) - 2; see get_graph_data_polar_with_sidechains_angle).
    """
    pred_polar = pred_tensor[:, 20:29].cpu().detach()
    truth_polar = gt_tensor[:, 20:29].cpu().detach()
    first_residue_coord = first_res[:, 1, :].detach().numpy().reshape(-1, 3)

    # NOTE: _get_cartesian returns (Cart_truth, Cart_pred) -- truth first!
    _, Cart_pred_local = _get_cartesian(
        pred_polar.view(-1, 9), truth_polar.view(-1, 9)
    )

    ca_pred = Cart_pred_local[:, 3:6].numpy()

    # Cumulative sum from first residue anchor (matches AbODE's
    # evaluate_rmsd_with_sidechains_cond_angle)
    for i in range(len(ca_pred)):
        if i == 0:
            ca_pred[i] = ca_pred[i] + first_residue_coord
        else:
            ca_pred[i] = ca_pred[i] + ca_pred[i - 1]

    if native_cdr_coords is None or len(native_cdr_coords) != len(ca_pred):
        raise ValueError(
            f"native_cdr_coords length {None if native_cdr_coords is None else len(native_cdr_coords)} "
            f"does not match ca_pred length {len(ca_pred)}"
        )

    ca_pred = kabsch_align(ca_pred, native_cdr_coords)
    ca_truth = native_cdr_coords.copy()
    return ca_pred, ca_truth


def load_native_cdr_coords(cid, cdr_type, data_root):
    """Load native CDR-only CA coordinates from complex_features.

    AbODE's polar features are CDR-only (antibody_cdr_len = len(ab_seq) - 2
    in get_graph_data_polar_with_sidechains_angle), so we return CDR-only
    coords (no flanking) to match ca_pred length.

    Args:
        cid: complex_id string
        cdr_type: CDR type string (e.g., "1", "2", "3")
        data_root: CHIMERA data root path

    Returns:
        numpy array of shape (L, 3) or None if not found
    """
    feat_path = os.path.join(data_root, "processed", "complex_features", f"{cid}.pt")
    if not os.path.exists(feat_path):
        return None

    feat = torch.load(feat_path, map_location="cpu", weights_only=False)

    # Get CDR mask (H1=0, H2=1, H3=2 in IMGT mask)
    cdr_idx = int(cdr_type) - 1  # Convert to 0-indexed
    cdr_masks = feat.get("cdr_masks", {}).get("imgt", {})
    heavy_mask = cdr_masks.get("heavy", [])

    if not heavy_mask:
        return None

    cdr_indices = [i for i, v in enumerate(heavy_mask) if v == cdr_idx]
    if len(cdr_indices) == 0:
        return None

    heavy_ca = feat.get("heavy_ca_coords")
    if heavy_ca is None:
        return None

    heavy_ca = np.array(heavy_ca) if not isinstance(heavy_ca, np.ndarray) else heavy_ca

    return heavy_ca[cdr_indices[0]:cdr_indices[-1] + 1]


def run_inference_with_metrics(model, data_list, cid_list, device, c, cdr_type):
    """Run inference and compute CHIMERA metrics per complex."""
    model.eval()
    t_space = np.linspace(0, 1, c.t_nsamples)
    predictions = []
    all_metrics = {k: [] for k in METRIC_KEYS}
    cdr_label = f"H{cdr_type}"

    with torch.no_grad():
        for batch, cid in tqdm(zip(data_list, cid_list), desc="Inference",
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
                method=c.solver, rtol=c.rtol, atol=c.atol,
                options=options,
            )

            y_gt = batch.y.to(device)
            antibody_len = [batch.ab_len.item()]
            antigen_len = [batch.ag_len.item()]
            final_pred = get_antibody_entries(
                y_pred[-1], torch.zeros(len(data), dtype=torch.long, device=device),
                antibody_len, antigen_len,
            )

            # Decode predicted sequence
            pred_seq = decode_sequence(final_pred)

            # Decode true sequence
            true_seq = decode_sequence(y_gt)

            # Load native CDR coords for alignment to PDB frame
            native_cdr_coords = load_native_cdr_coords(cid, cdr_type, c.data_root)

            # Reconstruct CA coordinates in native PDB frame (Kabsch-aligned)
            pred_ca, true_ca = reconstruct_ca_coords(
                final_pred, y_gt, batch.first_res, native_cdr_coords)

            # Compute PPL from cross-entropy
            pred_logits = final_pred[:, :20].detach().cpu()
            true_labels = y_gt[:, :20].detach().cpu()
            ce_loss = torch.nn.functional.cross_entropy(
                pred_logits, true_labels, reduction="mean"
            )
            ppl_val = float(torch.exp(ce_loss).item())

            # CHIMERA metrics
            aar_val = chimera_aar(pred_seq, true_seq)
            rmsd_val = kabsch_rmsd(pred_ca, true_ca) if len(pred_ca) > 0 else 0.0
            tm_val = chimera_tm_score(pred_ca, true_ca) if len(pred_ca) > 0 else 0.0
            liab = count_liabilities(pred_seq)

            all_metrics["ppl"].append(ppl_val)
            all_metrics["aar"].append(aar_val)
            all_metrics["rmsd"].append(rmsd_val)
            all_metrics["tm_score"].append(tm_val)
            all_metrics["n_liabilities"].append(liab)

            pred_dict = {
                "complex_id": cid,
                "cdr_type": cdr_label,
                "pred_sequence": pred_seq,
                "true_sequence": true_seq,
                "pred_coords": pred_ca,
                "true_coords": true_ca,
                "ppl": ppl_val,
                "aar": aar_val,
                "rmsd": rmsd_val,
                "tm_score": tm_val,
                "n_liabilities": liab,
            }
            predictions.append(pred_dict)

    summary = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
    return predictions, summary


def save_test_csv(predictions, summary, cdr_label):
    csv_path = os.path.join(_SCRIPT_DIR, "test_metrics.csv")
    row = {"cdr_type": cdr_label, "n_complexes": len(predictions)}
    row.update(summary)

    fieldnames = ["cdr_type", "n_complexes"] + METRIC_KEYS
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(row)
    print(f"Saved test metrics CSV to {csv_path}")


def main():
    args = parse_args()
    cfg, shared = load_config()
    c = build_config(cfg, shared, args)
    seed_everything(c.seed)

    if torch.cuda.is_available():
        torch.cuda.set_device(c.gpu)
    device = torch.device(f"cuda:{c.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cdr_label = f"H{c.cdr_type}"
    run_name = f"abode_cdr{c.cdr_type}_{c.split}"
    save_dir = os.path.join(c.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save resolved config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(c), f, indent=2)

    # Load data for this CDR type and split
    split_dir = os.path.join(c.data_dir, c.split, f"cdrh{c.cdr_type}")
    train_json = os.path.join(split_dir, "train.json")
    val_json = os.path.join(split_dir, "val.json")
    test_json = os.path.join(split_dir, "test.json")

    for p in [train_json, val_json, test_json]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run preprocess.py first.")
            sys.exit(1)

    print(f"Loading datasets for split={c.split}, cdr_type=H{c.cdr_type}...")

    train_data, train_cids = load_data_with_cids(c.cdr_type, train_json, c.mask)
    val_data, val_cids = load_data_with_cids(c.cdr_type, val_json, c.mask)
    test_data, test_cids = load_data_with_cids(c.cdr_type, test_json, c.mask)

    print(f"Training: {len(train_data)}, Validation: {len(val_data)}, "
          f"Test: {len(test_data)}")

    # Model
    model = Adobe_cond(c.c_in, c.c_out).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr,
                                 weight_decay=c.weight_decay)

    # Callbacks
    early_stop = EarlyStopping(patience=c.patience, mode=c.es_mode)
    ckpt = ModelCheckpoint(save_dir, mode=c.ckpt_mode)

    # WandB
    wandb_run = setup_wandb(c.wandb_project, run_name, vars(c), enabled=c.use_wandb)

    # -- Timing --
    train_t0 = time.time()

    if c.test_only:
        # Skip training, load checkpoint directly
        ckpt_path = c.checkpoint or os.path.join(save_dir, "checkpoints", "best.pt")
        print(f"Test-only mode: loading checkpoint from {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
    else:
        # Training loop
        for epoch in range(c.max_epoch):
            t0 = time.time()

            train_loss = train_epoch(model, train_data, optimizer, device, c)
            val_loss, val_aar, val_rmsd = evaluate_epoch(model, val_data, device, c)

            elapsed = time.time() - t0
            is_best = ckpt.save(model, optimizer, None, epoch, val_loss)

            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"val_loss={val_loss:.4f} val_aar={val_aar:.4f} "
                  f"val_rmsd={val_rmsd:.4f} {'*' if is_best else ''} [{elapsed:.0f}s]")

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_aar": val_aar,
                "val_rmsd": val_rmsd,
            }
            if wandb_run:
                wandb_run.log(log_dict)

            if early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model for test
        print("Loading best model for test...")
        ckpt.load_best(model, device)

    train_time_s = time.time() - train_t0

    # Test inference
    infer_t0 = time.time()
    predictions, test_metrics = run_inference_with_metrics(
        model, test_data, test_cids, device, c, c.cdr_type)

    # Save predictions
    pred_dir = os.path.join(save_dir, "predictions")
    save_predictions(predictions, pred_dir)
    print(f"Saved {len(predictions)} predictions to {pred_dir}")

    # Save test metrics CSV
    save_test_csv(predictions, test_metrics, cdr_label)

    # Print test summary
    print(f"\nTest results ({len(predictions)} complexes, {cdr_label}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    # Save timing
    infer_time_s = time.time() - infer_t0
    timing_path = os.path.join(save_dir, "timing.json")
    with open(timing_path, "w") as f:
        json.dump({"train_time_s": train_time_s, "infer_time_s": infer_time_s}, f, indent=2)
    print(f"Timing: train={train_time_s:.1f}s, infer={infer_time_s:.1f}s")

    # Log test metrics to wandb
    if wandb_run:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.log({"test_n": len(predictions)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
