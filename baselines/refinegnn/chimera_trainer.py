"""CHIMERA trainer for RefineGNN baseline.

Loads RefineGNN JSONL data (from preprocess.py), trains HierarchicalDecoder
for one CDR at a time (H1, H2, or H3). Computes CHIMERA metrics on val/test
sets and saves per-complex predictions.

Usage:
    cd baselines/refinegnn
    python chimera_trainer.py --max_epoch 2
    python chimera_trainer.py --split temporal --cdr_type 1
"""

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from structgen import HierarchicalDecoder, AntibodyDataset, StructureLoader, completize
from structgen.data import alphabet
from structgen.utils import compute_rmsd

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, load_split_ids,
    EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, save_predictions,
    run_full_evaluation, FULL_METRIC_KEYS,
)

# CHIMERA evaluation metrics
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", ".."))
from benchmark.evaluation.metrics import (
    aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score,
    count_liabilities,
)

# Basic metrics for validation/inference (full 12 metrics computed via run_full_evaluation)
METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities"]
CDR_NUM_TO_LABEL = {"1": "H1", "2": "H2", "3": "H3"}


def load_config():
    """Load baseline config.yaml from script directory, merged with shared config."""
    config_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    shared = load_shared_config()
    return cfg, shared


def parse_args():
    parser = argparse.ArgumentParser(description="RefineGNN CHIMERA trainer")
    parser.add_argument("--split", type=str, default=None,
                        choices=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--cdr_type", type=str, default=None, choices=["1", "2", "3"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_tokens", type=int, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training, load checkpoint and run test only")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for --test_only mode")
    return parser.parse_args()


def build_config(cfg, shared, args):
    """Build flat config namespace from YAML + shared config + CLI overrides."""
    ds = cfg.get("dataset", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    callbacks = cfg.get("callbacks", {})
    wandb_cfg = cfg.get("wandb", {})
    paths = shared["paths"]

    ns = SimpleNamespace(
        # Paths
        data_root=paths["data_root"],
        data_dir=os.path.join(paths["trans_baselines"], "refinegnn"),
        output_dir=os.path.join(paths["results_root"], "refinegnn"),
        # Dataset
        split=ds.get("split", "epitope_group"),
        cdr_type=str(ds.get("cdr_type", "3")),
        # Model (must match HierarchicalDecoder args)
        hidden_size=model.get("hidden_size", 256),
        depth=model.get("depth", 4),
        k_neighbors=model.get("k_neighbors", 9),
        block_size=model.get("block_size", 8),
        vocab_size=model.get("vocab_size", 21),
        num_rbf=model.get("num_rbf", 16),
        update_freq=model.get("update_freq", 1),
        dropout=model.get("dropout", 0.1),
        # Training
        seed=training.get("seed", 7),
        gpu=training.get("gpu", 0),
        lr=training.get("lr", 1e-3),
        clip_norm=training.get("clip_norm", 5.0),
        batch_tokens=training.get("batch_tokens", 100),
        max_epoch=training.get("max_epoch", 10),
        anneal_rate=training.get("anneal_rate", 0.9),
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
    if args.batch_tokens is not None:
        ns.batch_tokens = args.batch_tokens
    if args.max_epoch is not None:
        ns.max_epoch = args.max_epoch
    if not args.no_wandb:
        ns.use_wandb = True

    # Test-only mode
    ns.test_only = args.test_only
    ns.checkpoint = args.checkpoint

    return ns


def train_epoch(model, loader, optimizer, c):
    """One training epoch following ab_train.py."""
    model.train()
    total_loss, total_snll, n_batches = 0.0, 0.0, 0

    for hbatch in tqdm(loader, desc="Train"):
        hchain = completize(hbatch)
        if hchain[-1].sum().item() == 0:
            continue
        optimizer.zero_grad()
        loss, snll = model(*hchain)
        loss.backward()
        if c.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), c.clip_norm)
        optimizer.step()
        total_loss += loss.item()
        total_snll += snll.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0
    return total_loss / n_batches, total_snll / n_batches


def evaluate_ppl_rmsd(model, loader, cdr_type):
    """Compute PPL and RMSD on a validation set (following ab_train.py evaluate)."""
    model.eval()
    val_nll = val_tot = 0.0
    val_rmsd = []

    with torch.no_grad():
        for hbatch in tqdm(loader, desc="Eval"):
            hX, hS, hL, hmask = completize(hbatch)
            for i in range(len(hbatch)):
                L = hmask[i:i+1].sum().long().item()
                if L > 0 and cdr_type in hL[i]:
                    out = model.log_prob(hS[i:i+1, :L], [hL[i]], hmask[i:i+1, :L])
                    nll, X_pred = out.nll, out.X_cdr
                    val_nll += nll.item() * hL[i].count(cdr_type)
                    val_tot += hL[i].count(cdr_type)
                    l = hL[i].index(cdr_type)
                    r = hL[i].rindex(cdr_type)
                    rmsd = compute_rmsd(
                        X_pred[:, :, 1, :],
                        hX[i:i+1, l:r+1, 1, :],
                        hmask[i:i+1, l:r+1],
                    )
                    val_rmsd.append(rmsd.item())

    ppl = math.exp(val_nll / val_tot) if val_tot > 0 else float("inf")
    avg_rmsd = sum(val_rmsd) / len(val_rmsd) if val_rmsd else float("inf")
    return ppl, avg_rmsd


def run_inference(model, dataset, cdr_type, cid_lookup):
    """Run generation on each complex and compute CHIMERA metrics.

    Args:
        model: trained HierarchicalDecoder
        dataset: list of JSONL entry dicts (from AntibodyDataset.data)
        cdr_type: '1', '2', or '3'
        cid_lookup: dict mapping pdb -> complex_id (for entries with complex_id)

    Returns:
        predictions: list of per-complex prediction dicts
        summary: dict of mean metric values
    """
    model.eval()
    predictions = []
    all_metrics = {k: [] for k in METRIC_KEYS}
    cdr_label = CDR_NUM_TO_LABEL[cdr_type]

    with torch.no_grad():
        for ab in tqdm(dataset, desc="Inference"):
            if cdr_type not in ab.get("cdr", ""):
                continue

            cid = ab.get("complex_id", ab.get("pdb", "unknown"))

            # Extract ground truth CDR sequence
            orig_cdr = "".join(
                [x for x, y in zip(ab["seq"], ab["cdr"]) if y == cdr_type]
            )

            # Generate with batch_size=1
            hX, hS, hL, hmask = completize([ab])
            gen_seqs, gen_ppl, gen_X = model.generate(
                hS, hL, hmask, return_ppl=True
            )

            pred_seq = gen_seqs[0]
            ppl_val = gen_ppl[0].item()

            # Extract CDR CA coords
            # gen_X is (1, CDR_len, 4, 3) -- backbone coords for CDR region
            pred_ca = gen_X[0, :, 1, :].cpu().numpy()  # CA at backbone idx 1

            # Ground truth CDR CA coords
            l = ab["cdr"].index(cdr_type)
            r = ab["cdr"].rindex(cdr_type)
            true_ca = np.array(ab["coords"]["CA"][l:r+1])

            # Compute CHIMERA metrics
            aar_val = chimera_aar(pred_seq, orig_cdr)
            rmsd_val = kabsch_rmsd(pred_ca, true_ca)
            tm_val = chimera_tm_score(pred_ca, true_ca)
            liab = count_liabilities(pred_seq)

            all_metrics["ppl"].append(ppl_val)
            all_metrics["aar"].append(aar_val)
            all_metrics["rmsd"].append(rmsd_val)
            all_metrics["tm_score"].append(tm_val)
            all_metrics["n_liabilities"].append(liab)

            pred = {
                "complex_id": cid,
                "cdr_type": cdr_label,
                "pred_sequence": pred_seq,
                "true_sequence": orig_cdr,
                "pred_coords": pred_ca,
                "true_coords": true_ca,
                "ppl": ppl_val,
                "aar": aar_val,
                "rmsd": rmsd_val,
                "tm_score": tm_val,
                "n_liabilities": liab,
            }
            predictions.append(pred)

    summary = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
    return predictions, summary


def save_test_csv(summary, cdr_label):
    """Save test metrics as CSV to baseline directory."""
    csv_path = os.path.join(_SCRIPT_DIR, "test_metrics.csv")
    row = {"cdr_type": cdr_label}
    # Format metrics as "mean±std" strings
    for k in FULL_METRIC_KEYS:
        v = summary.get(k, {})
        if isinstance(v, dict):
            mean_val = v.get("mean", 0)
            std_val = v.get("std", 0)
            row[k] = f"{mean_val:.2f}±{std_val:.2f}"
        else:
            row[k] = v

    fieldnames = ["cdr_type"] + list(FULL_METRIC_KEYS)
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

    cdr_label = CDR_NUM_TO_LABEL[c.cdr_type]
    run_name = f"refinegnn_cdr{c.cdr_type}_{c.split}"
    save_dir = os.path.join(c.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save resolved config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(c), f, indent=2)

    # Load JSONL data for the split
    split_dir = os.path.join(c.data_dir, c.split)
    train_path = os.path.join(split_dir, "train.jsonl")
    val_path = os.path.join(split_dir, "val.jsonl")
    test_path = os.path.join(split_dir, "test.jsonl")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found. Run preprocess.py first.")
            sys.exit(1)

    print(f"Loading datasets for split={c.split}, cdr_type=H{c.cdr_type}...")
    train_data = AntibodyDataset(train_path, cdr_type=c.cdr_type)
    val_data = AntibodyDataset(val_path, cdr_type=c.cdr_type)
    test_data = AntibodyDataset(test_path, cdr_type=c.cdr_type)

    train_loader = StructureLoader(
        train_data.data, batch_tokens=c.batch_tokens,
        interval_sort=int(c.cdr_type),
    )
    val_loader = StructureLoader(
        val_data.data, batch_tokens=c.batch_tokens,
        interval_sort=int(c.cdr_type),
    )
    test_loader = StructureLoader(
        test_data.data, batch_tokens=c.batch_tokens,
        interval_sort=int(c.cdr_type),
    )

    print(f"Training: {len(train_data.data)}, Validation: {len(val_data.data)}, "
          f"Test: {len(test_data.data)}")

    # Model
    model = HierarchicalDecoder(c).cuda()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Test-only mode: load checkpoint and skip training
    if c.test_only:
        if c.checkpoint:
            ckpt_path = c.checkpoint
        else:
            # Auto-find best checkpoint in save_dir
            ckpt_path = os.path.join(save_dir, "checkpoints", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found at {ckpt_path}")
            sys.exit(1)
        print(f"Loading checkpoint from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        wandb_run = None
    else:
        # Optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)

        # Callbacks
        early_stop = EarlyStopping(patience=c.patience, mode=c.es_mode)
        ckpt = ModelCheckpoint(save_dir, mode=c.ckpt_mode)

        # WandB
        wandb_run = setup_wandb(c.wandb_project, run_name, vars(c), enabled=c.use_wandb)

        # Training loop
        for epoch in range(c.max_epoch):
            t0 = time.time()

            train_loss, train_snll = train_epoch(model, train_loader, optimizer, c)
            val_ppl, val_rmsd = evaluate_ppl_rmsd(model, val_loader, c.cdr_type)

            elapsed = time.time() - t0
            train_ppl = math.exp(train_snll) if train_snll < 20 else float("inf")

            # Checkpoint on val PPL
            is_best = ckpt.save(model, optimizer, None, epoch, val_ppl)

            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} "
                  f"train_ppl={train_ppl:.2f} val_ppl={val_ppl:.3f} "
                  f"val_rmsd={val_rmsd:.3f} {'*' if is_best else ''} [{elapsed:.0f}s]")

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_ppl": train_ppl,
                "val_ppl": val_ppl,
                "val_rmsd": val_rmsd,
            }
            if wandb_run:
                wandb_run.log(log_dict)

            if early_stop(val_ppl):
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model for test
        print("Loading best model for test...")
        ckpt.load_best(model, device)

    test_ppl, test_rmsd = evaluate_ppl_rmsd(model, test_loader, c.cdr_type)
    print(f"Test PPL = {test_ppl:.3f}, Test RMSD = {test_rmsd:.3f}")

    # Run inference to generate predictions
    predictions, _ = run_inference(
        model, test_data.data, c.cdr_type, cid_lookup={},
    )

    # Save predictions to per-CDR subdirectory
    pred_dir = os.path.join(save_dir, "predictions", cdr_label)
    save_predictions(predictions, pred_dir)
    print(f"Saved {len(predictions)} predictions to {pred_dir}")

    # Run full CHIMERA evaluation (all 12 metrics)
    numbering_scheme = cfg.get("numbering_scheme", "chothia")
    _, full_summary, _ = run_full_evaluation(
        pred_dir, c.split, c.data_root, cdr_type_hint=cdr_label,
        numbering_scheme=numbering_scheme,
    )

    # Save test metrics CSV with all 12 metrics
    save_test_csv(full_summary, cdr_label)

    # Print test summary
    print(f"\nTest results ({len(predictions)} complexes, {cdr_label}):")
    for k in FULL_METRIC_KEYS:
        v = full_summary.get(k, {})
        if isinstance(v, dict):
            mean_val = v.get("mean", 0)
            std_val = v.get("std", 0)
            print(f"  {k}: {mean_val:.2f}±{std_val:.2f}")
        else:
            print(f"  {k}: {v}")

    # Log test metrics to wandb
    if wandb_run:
        for k in FULL_METRIC_KEYS:
            v = full_summary.get(k, {})
            if isinstance(v, dict):
                v = v.get("mean", 0)
            elif isinstance(v, str) and "±" in v:
                v = float(v.split("±")[0])
            wandb_run.log({f"test_{cdr_label}_{k}": v})
        wandb_run.log({"test_n": len(predictions), "test_ppl_eval": test_ppl,
                       "test_rmsd_eval": test_rmsd})
        wandb_run.finish()


if __name__ == "__main__":
    main()
