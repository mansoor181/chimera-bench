"""CHIMERA trainer for AbDockGen baseline.

AbDockGen is a H3-only epitope-conditioned CDR design model using a
hierarchical equivariant GNN. Generates CDR-H3 sequence + structure
autoregressively, conditioned on the epitope surface.

Follows lm_train.py closely, using AbDockGen's native AntibodyComplexDataset,
ComplexLoader, and CondRefineDecoder.

Usage:
    cd baselines/abdockgen
    python chimera_trainer.py
    python chimera_trainer.py --split temporal
    python chimera_trainer.py --split epitope_group --max_epoch 1  # smoke test
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
import torch.nn as nn
import yaml
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from bindgen import (
    AntibodyComplexDataset, ComplexLoader, make_batch,
    CondRefineDecoder, compute_rmsd,
)
from bindgen.data import ALPHABET

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, save_predictions,
)

sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", ".."))
from benchmark.evaluation.metrics import (
    aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score,
    count_liabilities,
)

METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities"]


def load_config():
    config_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    shared = load_shared_config()
    return cfg, shared


def parse_args():
    parser = argparse.ArgumentParser(description="AbDockGen CHIMERA trainer")
    parser.add_argument("--split", type=str, default=None,
                        choices=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_epoch", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    return parser.parse_args()


def build_config(cfg, shared, args):
    ds = cfg.get("dataset", {})
    model = cfg.get("model", {})
    training = cfg.get("training", {})
    callbacks = cfg.get("callbacks", {})
    wandb_cfg = cfg.get("wandb", {})
    paths = shared["paths"]

    ns = SimpleNamespace(
        # Paths
        data_root=paths["data_root"],
        data_dir=os.path.join(paths["trans_baselines"], "abdockgen"),
        output_dir=os.path.join(paths["results_root"], "abdockgen"),
        # Dataset
        split=ds.get("split", "epitope_group"),
        cdr=str(ds.get("cdr", "3")),
        # Model (used by CondRefineDecoder via args)
        hierarchical=model.get("hierarchical", True),
        hidden_size=model.get("hidden_size", 256),
        depth=model.get("depth", 4),
        k_neighbors=model.get("k_neighbors", 9),
        L_target=model.get("L_target", 20),
        clash_step=model.get("clash_step", 10),
        num_rbf=model.get("num_rbf", 16),
        dropout=model.get("dropout", 0.1),
        # Model flags (not used for CondRefineDecoder but kept for build_model compat)
        att_refine=False,
        no_target=False,
        sequence=False,
        # Training
        seed=training.get("seed", 42),
        gpu=training.get("gpu", 0),
        lr=training.get("lr", 1e-3),
        batch_tokens=training.get("batch_tokens", 100),
        max_epoch=training.get("max_epoch", 10),
        anneal_rate=training.get("anneal_rate", 0.9),
        clip_norm=training.get("clip_norm", 1.0),
        print_iter=training.get("print_iter", 50),
        # Callbacks
        patience=callbacks.get("early_stopping", {}).get("patience", 10),
        es_mode=callbacks.get("early_stopping", {}).get("mode", "min"),
        ckpt_mode=callbacks.get("checkpoint", {}).get("mode", "min"),
        # WandB
        use_wandb=wandb_cfg.get("enabled", False),
        wandb_project=wandb_cfg.get("project", "chimera_baselines"),
    )

    if args.split is not None:
        ns.split = args.split
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

    return ns


def evaluate(model, loader):
    """Evaluate model on a loader. Returns (PPL, [complex_rmsd, ab_bb_rmsd, ab_full_rmsd])."""
    model.eval()
    total_nll = total = 0
    complex_rmsd, ab_bb_rmsd, ab_full_rmsd = [], [], []
    with torch.no_grad():
        for batch in loader:
            batch_data = make_batch(batch)
            out = model(*batch_data)

            bind_X, _, bind_A, _ = batch_data[0]
            tgt_X, tgt_A = out.handle
            bind_mask = bind_A.clamp(max=1).float()
            tgt_mask = tgt_A.clamp(max=1).float()
            idx1, idx2, idx3 = torch.nonzero(bind_mask, as_tuple=True)

            total_nll += out.nll.item() * bind_mask[..., 1].sum().item()
            total += bind_mask[..., 1].sum().item()

            true_X = torch.cat([bind_X, tgt_X], dim=1)
            pred_X = torch.cat([out.bind_X, tgt_X], dim=1)
            true_mask = torch.cat([bind_mask, torch.zeros_like(tgt_mask)], dim=1)
            rmsd = compute_rmsd(pred_X[:, :, 1], true_X[:, :, 1], true_mask[:, :, 1])
            complex_rmsd.extend(rmsd.tolist())
            rmsd = compute_rmsd(out.bind_X[:, :, 1], bind_X[:, :, 1], bind_mask[:, :, 1])
            ab_bb_rmsd.extend(rmsd.tolist())
            rmsd = compute_rmsd(
                out.bind_X[idx1, idx2, idx3, :].view(1, -1, 3),
                bind_X[idx1, idx2, idx3, :].view(1, -1, 3),
                bind_mask[idx1, idx2, idx3].view(1, -1),
            )
            ab_full_rmsd.extend(rmsd.tolist())

    if total == 0:
        return float("inf"), [0, 0, 0]
    return math.exp(total_nll / total), [
        sum(x) / max(len(x), 1) for x in [complex_rmsd, ab_bb_rmsd, ab_full_rmsd]
    ]


def run_inference(model, dataset, c):
    """Run generation inference on each test complex, compute per-sample metrics.

    Returns:
        predictions: list of per-complex dicts (for save_predictions)
        summary: dict of mean metric values
    """
    model.eval()
    predictions = []
    all_metrics = {k: [] for k in METRIC_KEYS}

    with torch.no_grad():
        for ab in tqdm(dataset, desc="Inference"):
            # Generate one sample per complex
            _, epitope, surface = make_batch([ab])
            out = model.generate(epitope, surface)

            pred_seq = out.handle[0]
            pred_ppl = out.ppl[0].item()
            pred_ca = out.bind_X[0, :, 1].cpu().numpy()  # CA at atom index 1

            true_seq = ab["binder_seq"]
            true_ca = ab["binder_coords"][:, 1].numpy()  # CA at atom index 1

            cid = ab.get("complex_id", ab.get("pdb", "unknown"))

            aar_val = chimera_aar(pred_seq, true_seq)
            rmsd_val = kabsch_rmsd(pred_ca, true_ca)
            tm_val = chimera_tm_score(pred_ca, true_ca)
            liab = count_liabilities(pred_seq)

            all_metrics["ppl"].append(pred_ppl)
            all_metrics["aar"].append(aar_val)
            all_metrics["rmsd"].append(rmsd_val)
            all_metrics["tm_score"].append(tm_val)
            all_metrics["n_liabilities"].append(liab)

            predictions.append({
                "complex_id": cid,
                "cdr_type": "H3",
                "pred_sequence": pred_seq,
                "true_sequence": true_seq,
                "pred_coords": pred_ca,
                "true_coords": true_ca,
                "ppl": pred_ppl,
                "aar": aar_val,
                "rmsd": rmsd_val,
                "tm_score": tm_val,
                "n_liabilities": liab,
            })

    summary = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
    return predictions, summary


def save_test_csv(predictions, summary):
    csv_path = os.path.join(_SCRIPT_DIR, "test_metrics.csv")
    row = {"cdr_type": "H3", "n_complexes": len(predictions)}
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

    run_name = f"abdockgen_H3_{c.split}"
    save_dir = os.path.join(c.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(c), f, indent=2)

    # Load data
    split_dir = os.path.join(c.data_dir, c.split)
    print(f"Loading AbDockGen data from {split_dir}...")

    train_data = AntibodyComplexDataset(
        os.path.join(split_dir, "train.jsonl"), cdr_type=c.cdr, L_target=c.L_target)
    val_data = AntibodyComplexDataset(
        os.path.join(split_dir, "val.jsonl"), cdr_type=c.cdr, L_target=c.L_target)
    test_data = AntibodyComplexDataset(
        os.path.join(split_dir, "test.jsonl"), cdr_type=c.cdr, L_target=c.L_target)

    # Inject complex_id into dataset entries for inference tracking
    for dataset, subset in [(train_data, "train"), (val_data, "val"), (test_data, "test")]:
        for entry in dataset.data:
            if "complex_id" not in entry:
                entry["complex_id"] = entry.get("pdb", "unknown")

    loader_train = ComplexLoader(train_data, batch_tokens=c.batch_tokens)
    loader_val = ComplexLoader(val_data, batch_tokens=0)
    loader_test = ComplexLoader(test_data, batch_tokens=0)

    print(f"Training: {len(train_data)}, Validation: {len(val_data)}, Test: {len(test_data)}")

    # Model
    model = CondRefineDecoder(c).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    ckpt = ModelCheckpoint(save_dir, mode=c.ckpt_mode)
    early_stop = EarlyStopping(patience=c.patience, mode=c.es_mode)
    wandb_run = setup_wandb(c.wandb_project, run_name, vars(c), enabled=c.use_wandb)

    # Training loop
    best_ppl = float("inf")
    best_epoch = -1
    for epoch in range(c.max_epoch):
        t0 = time.time()
        model.train()
        meter = 0

        for i, batch in enumerate(tqdm(loader_train, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            out = model(*make_batch(batch))
            out.loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), c.clip_norm)
            optimizer.step()

            meter += out.loss.item()
            if (i + 1) % c.print_iter == 0:
                meter /= c.print_iter
                print(f"  [{i + 1}] Train Loss = {meter:.3f}")
                meter = 0

        # Validation
        val_ppl, val_rmsd = evaluate(model, loader_val)
        elapsed = time.time() - t0

        # Save checkpoint (uses val PPL as metric)
        is_best = ckpt.save(model, optimizer, None, epoch, val_ppl)

        if val_ppl < best_ppl:
            best_ppl = val_ppl
            best_epoch = epoch

        # Anneal learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] *= c.anneal_rate

        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d} | val_ppl={val_ppl:.3f} "
              f"complex_rmsd={val_rmsd[0]:.3f} ab_bb_rmsd={val_rmsd[1]:.3f} "
              f"ab_full_rmsd={val_rmsd[2]:.3f} "
              f"lr={lr:.6f} {'*' if is_best else ''} [{elapsed:.0f}s]")

        log_dict = {
            "epoch": epoch,
            "val_ppl": val_ppl,
            "val_complex_rmsd": val_rmsd[0],
            "val_ab_bb_rmsd": val_rmsd[1],
            "val_ab_full_rmsd": val_rmsd[2],
            "lr": lr,
        }
        if wandb_run:
            wandb_run.log(log_dict)

        if early_stop(val_ppl):
            print(f"Early stopping at epoch {epoch}")
            break

    # Test with best model
    print("Loading best model for test...")
    ckpt.load_best(model, device)

    predictions, test_metrics = run_inference(model, test_data, c)

    # Save predictions
    pred_dir = os.path.join(save_dir, "predictions", "H3")
    save_predictions(predictions, pred_dir)
    print(f"Saved {len(predictions)} predictions to {pred_dir}")

    save_test_csv(predictions, test_metrics)

    print(f"\nTest results ({len(predictions)} complexes, H3):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.4f}")

    if wandb_run:
        wandb_run.log({f"test_{k}": v for k, v in test_metrics.items()})
        wandb_run.log({"test_n": len(predictions)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
