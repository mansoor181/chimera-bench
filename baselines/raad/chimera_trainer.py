"""CHIMERA trainer for RAAD baseline.

Loads pre-cached dataset, splits by CHIMERA split IDs, trains with RAAD
hyperparameters from config.yaml. Computes CHIMERA metrics on val/test sets
and logs to wandb. RAAD generates one CDR at a time (H1, H2, or H3).

Usage:
    cd baselines/raad
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
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Fix RAAD import path conflict
_raad_code = os.path.join(_SCRIPT_DIR, "code")
_removed = []
for _p in list(sys.path):
    _abs = os.path.abspath(_p) if _p else os.path.abspath(".")
    if _abs == _SCRIPT_DIR or _abs == os.path.abspath("."):
        sys.path.remove(_p)
        _removed.append(_p)
sys.path.insert(0, _raad_code)

from models import AntiDesigner
from dataset import EquiAACDataset
from data.pdb_utils import AAComplex, Protein, VOCAB
from utils import print_log

for _p in _removed:
    if _p not in sys.path:
        sys.path.append(_p)

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, load_split_ids, split_dataset, DatasetView,
    EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, to_device, save_predictions,
    run_full_evaluation, FULL_METRIC_KEYS,
)

# CHIMERA evaluation metrics (for validation, not test)
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", ".."))
from benchmark.evaluation.metrics import (
    aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score,
    count_liabilities,
)

METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities"]  # Basic metrics for validation


class RobustEquiAACDataset(EquiAACDataset):
    """EquiAACDataset that skips PDBs that fail parsing."""

    def preprocess(self, file_path, save_dir, num_entry_per_file):
        with open(file_path, "r") as fin:
            lines = fin.read().strip().split("\n")
        for line in tqdm(lines, desc="Preprocessing"):
            item = json.loads(line)
            try:
                protein = Protein.from_pdb(item["pdb_data_path"])
            except Exception as e:
                print_log(f'parse {item["pdb"]} failed: {e}, skip', level="ERROR")
                continue
            pdb_id, peptides = item["pdb"], protein.peptides
            try:
                self.data.append(AAComplex(
                    pdb_id, peptides, item["heavy_chain"],
                    item["light_chain"], item["antigen_chains"]))
            except Exception as e:
                print_log(f'AAComplex {pdb_id} failed: {e}, skip', level="ERROR")
                continue
            if num_entry_per_file > 0 and len(self.data) >= num_entry_per_file:
                self._save_part(save_dir, num_entry_per_file)
        if len(self.data):
            self._save_part(save_dir, num_entry_per_file)


def load_config():
    """Load baseline config.yaml from script directory, merged with shared config."""
    config_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    shared = load_shared_config()
    return cfg, shared


def parse_args():
    parser = argparse.ArgumentParser(description="RAAD CHIMERA trainer")
    # CLI overrides for common fields
    parser.add_argument("--split", type=str, default=None,
                        choices=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--cdr_type", type=str, default=None, choices=["1", "2", "3"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
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
        # Paths (from shared config)
        data_root=paths["data_root"],
        data_dir=os.path.join(paths["trans_baselines"], "raad"),
        output_dir=os.path.join(paths["results_root"], "raad"),
        # Dataset
        split=ds.get("split", "epitope_group"),
        cdr_type=str(ds.get("cdr_type", "3")),
        # Model
        embed_size=model.get("embed_size", 32),
        hidden_size=model.get("hidden_size", 256),
        n_layers=model.get("n_layers", 4),
        dropout=model.get("dropout", 0.0),
        alpha=model.get("alpha", 0.8),
        beta=model.get("beta", 0.5),
        node_feats_mode=str(model.get("node_feats_mode", "1111")),
        edge_feats_mode=str(model.get("edge_feats_mode", "1111")),
        interface_only=model.get("interface_only", 1),
        mode=str(model.get("mode", "111")),
        # Training
        seed=training.get("seed", 42),
        gpu=training.get("gpu", 0),
        lr=training.get("lr", 1e-3),
        batch_size=training.get("batch_size", 8),
        max_epoch=training.get("max_epoch", 20),
        anneal_base=training.get("anneal_base", 0.9),
        grad_clip=training.get("grad_clip", 0.5),
        num_workers=training.get("num_workers", 4),
        # Callbacks
        patience=callbacks.get("early_stopping", {}).get("patience", 10),
        es_mode=callbacks.get("early_stopping", {}).get("mode", "min"),
        ckpt_mode=callbacks.get("checkpoint", {}).get("mode", "min"),
        # WandB
        use_wandb=wandb_cfg.get("enabled", True),
        wandb_project=wandb_cfg.get("project", "chimera_baselines"),
        # CDR numbering scheme
        numbering_scheme=cfg.get("numbering_scheme", "imgt"),
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
    if args.batch_size is not None:
        ns.batch_size = args.batch_size
    if args.max_epoch is not None:
        ns.max_epoch = args.max_epoch
    if not args.no_wandb:
        ns.use_wandb = True

    # Test-only mode
    ns.test_only = args.test_only
    ns.checkpoint = args.checkpoint

    return ns


def train_epoch(model, loader, optimizer, device, grad_clip):
    model.train()
    losses, snlls, closses = [], [], []
    for batch in tqdm(loader, desc="Train"):
        batch = to_device(batch, device)
        loss, snll, closs = model(batch["X"], batch["S"], batch["L"], batch["offsets"])
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        losses.append(loss.item())
        snlls.append(snll.item())
        closses.append(closs.item())
    return np.mean(losses), np.mean(snlls), np.mean(closses)


def valid_epoch(model, loader, device):
    model.eval()
    losses, snlls, closses = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            batch = to_device(batch, device)
            loss, snll, closs = model(batch["X"], batch["S"], batch["L"], batch["offsets"])
            losses.append(loss.item())
            snlls.append(snll.item())
            closses.append(closs.item())
    return np.mean(losses), np.mean(snlls), np.mean(closses)


def run_inference_with_metrics(model, dataset, loader, device, cdr_type, idx_to_cid):
    """Run inference and compute CHIMERA metrics per complex.

    Computable at CDR level: AAR, PPL, RMSD, TM-score, liabilities.
    Uses idx_to_cid (list indexed by full dataset position) to resolve
    dataset index -> chimera complex_id.

    Returns:
        predictions: list of per-complex prediction dicts
        summary: dict of mean metric values
    """
    model.eval()
    predictions = []
    all_metrics = {k: [] for k in METRIC_KEYS}
    idx = 0
    cdr_label = f"H{cdr_type}"

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch_size = len(batch["L"])
            ppls, seqs, xs, true_xs, aligned = model.infer(batch, device)
            for i in range(batch_size):
                dataset_idx = dataset.idx_mapping[idx]
                cplx = dataset.data[dataset_idx]
                cid = idx_to_cid[dataset_idx]

                origin_seq = cplx.get_cdr(cdr_label).get_seq()
                pred_seq = seqs[i]
                pred_x = xs[i]
                true_x = true_xs[i]

                # Extract CA coords
                pred_ca = pred_x[:, 1, :] if pred_x.ndim == 3 else pred_x
                true_ca = true_x[:, 1, :] if true_x.ndim == 3 else true_x
                if isinstance(pred_ca, torch.Tensor):
                    pred_ca = pred_ca.cpu().numpy()
                if isinstance(true_ca, torch.Tensor):
                    true_ca = true_ca.cpu().numpy()

                # Compute CHIMERA metrics
                aar_val = chimera_aar(pred_seq, origin_seq)
                rmsd_val = kabsch_rmsd(pred_ca, true_ca)
                tm_val = chimera_tm_score(pred_ca, true_ca)
                liab = count_liabilities(pred_seq)

                all_metrics["ppl"].append(ppls[i])
                all_metrics["aar"].append(aar_val)
                all_metrics["rmsd"].append(rmsd_val)
                all_metrics["tm_score"].append(tm_val)
                all_metrics["n_liabilities"].append(liab)

                pred = {
                    "complex_id": cid,
                    "cdr_type": cdr_label,
                    "pred_sequence": pred_seq,
                    "true_sequence": origin_seq,
                    "pred_coords": pred_ca,
                    "true_coords": true_ca,
                    "ppl": ppls[i],
                    "aar": aar_val,
                    "rmsd": rmsd_val,
                    "tm_score": tm_val,
                    "n_liabilities": liab,
                }
                predictions.append(pred)
                idx += 1

    summary = {k: float(np.mean(v)) for k, v in all_metrics.items() if v}
    return predictions, summary


def save_test_csv(summary, cdr_label):
    """Save test metrics as CSV (one row per CDR type) to baseline directory."""
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
    cdr_label = f"H{c.cdr_type}"
    run_name = f"raad_cdr{c.cdr_type}_{c.split}"
    save_dir = os.path.join(c.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save resolved config
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(c), f, indent=2)

    # Load pre-cached dataset
    data_dir = os.path.abspath(c.data_dir)
    jsonl_path = os.path.join(data_dir, "all.jsonl")
    print(f"Loading RAAD dataset from {data_dir}...")
    dataset = RobustEquiAACDataset(jsonl_path, interface_only=c.interface_only)
    dataset.mode = c.mode
    print(f"Dataset: {dataset.num_entry} complexes")

    # Load idx_to_cid mapping (ordered list: dataset_index -> chimera complex_id)
    idx_to_cid_path = os.path.join(data_dir, "idx_to_cid.json")
    with open(idx_to_cid_path) as f:
        idx_to_cid = json.load(f)
    assert len(idx_to_cid) == dataset.num_entry, (
        f"Mismatch: {len(idx_to_cid)} idx_to_cid vs {dataset.num_entry} dataset entries"
    )
    print(f"Loaded idx_to_cid mapping: {len(idx_to_cid)} entries")

    # Build cid_to_idx (inverse lookup for splitting)
    cid_to_idx = {cid: i for i, cid in enumerate(idx_to_cid)}

    # Split by CHIMERA split IDs
    split_ids = load_split_ids(c.split, c.data_root)
    print("Splitting dataset:")
    views = split_dataset(dataset, split_ids, cid_to_idx)
    train_set, valid_set, test_set = views["train"], views["val"], views["test"]

    n_channel = dataset[0]["X"].shape[1]

    # Model
    model = AntiDesigner(
        c.embed_size, c.hidden_size, n_channel,
        n_layers=c.n_layers, dropout=c.dropout,
        cdr_type=c.cdr_type, args=c,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Test-only mode: load checkpoint and skip training
    if c.test_only:
        if c.checkpoint:
            ckpt_path = c.checkpoint
        else:
            ckpt_path = os.path.join(save_dir, "checkpoints", "best.pt")
        if not os.path.exists(ckpt_path):
            print(f"ERROR: Checkpoint not found at {ckpt_path}")
            sys.exit(1)
        print(f"Loading checkpoint from {ckpt_path}...")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model_state_dict"])
        wandb_run = None
        # Test loader only
        test_loader = DataLoader(
            test_set, batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=EquiAACDataset.collate_fn)
    else:
        # Optimizer / Scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: c.anneal_base ** epoch)

        # Loaders
        train_loader = DataLoader(
            train_set, batch_size=c.batch_size, shuffle=True,
            num_workers=c.num_workers, collate_fn=EquiAACDataset.collate_fn)
        valid_loader = DataLoader(
            valid_set, batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=EquiAACDataset.collate_fn)
        test_loader = DataLoader(
            test_set, batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=EquiAACDataset.collate_fn)

        # Callbacks
        early_stop = EarlyStopping(patience=c.patience, mode=c.es_mode)
        ckpt = ModelCheckpoint(save_dir, mode=c.ckpt_mode)

        # WandB
        wandb_run = setup_wandb(c.wandb_project, run_name, vars(c), enabled=c.use_wandb)

        # Training loop
        for epoch in range(c.max_epoch):
            t0 = time.time()
            train_loss, train_snll, train_closs = train_epoch(
                model, train_loader, optimizer, device, c.grad_clip)

            val_loss, val_snll, val_closs = valid_epoch(model, valid_loader, device)
            scheduler.step()

            # Val inference with CHIMERA metrics
            _, val_metrics = run_inference_with_metrics(
                model, valid_set, valid_loader, device, c.cdr_type, idx_to_cid)

            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - t0
            is_best = ckpt.save(model, optimizer, scheduler, epoch, val_loss)

            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_ppl={np.exp(train_snll):.2f} val_ppl={np.exp(val_snll):.2f} "
                  f"val_aar={val_metrics.get('aar', 0):.4f} "
                  f"val_rmsd={val_metrics.get('rmsd', 0):.4f} "
                  f"val_tm={val_metrics.get('tm_score', 0):.4f} "
                  f"lr={lr:.6f} {'*' if is_best else ''} [{elapsed:.0f}s]")

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_snll": train_snll, "val_snll": val_snll,
                "train_closs": train_closs, "val_closs": val_closs,
                "train_ppl": float(np.exp(train_snll)),
                "val_ppl": float(np.exp(val_snll)),
                "lr": lr,
            }
            log_dict.update({f"val_{k}": v for k, v in val_metrics.items()})
            if wandb_run:
                wandb_run.log(log_dict)

            if early_stop(val_loss):
                print(f"Early stopping at epoch {epoch}")
                break

        # Load best model for test
        print("Loading best model for test...")
        ckpt.load_best(model, device)

    # Test inference (runs for both training and test_only modes)
    predictions, _ = run_inference_with_metrics(
        model, test_set, test_loader, device, c.cdr_type, idx_to_cid)

    # Save predictions to per-CDR subdirectory
    pred_dir = os.path.join(save_dir, "predictions", cdr_label)
    save_predictions(predictions, pred_dir)
    print(f"Saved {len(predictions)} predictions to {pred_dir}")

    # Run full CHIMERA evaluation (all 12 metrics)
    _, full_summary, _ = run_full_evaluation(
        pred_dir, c.split, c.data_root, cdr_type_hint=cdr_label,
        numbering_scheme=c.numbering_scheme)

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
        wandb_run.log({"test_n": len(predictions)})
        wandb_run.finish()


if __name__ == "__main__":
    main()
