"""CHIMERA trainer for dyAb baseline.

dyAb is a flow matching model built on the dyMEAN codebase. It uses IMGT
numbering, custom VOCAB with special tokens, and the same E2EDataset/collate_fn
as dyMEAN. Key differences from dyMEAN: dyAbModel has additional `flexible` and
`module_type` parameters, and uses dyAbEncoder + dyAbEGNN + PIE modules.

Usage:
    cd baselines/dyab
    python chimera_trainer.py
    python chimera_trainer.py --split temporal --cdr H3
"""

import argparse
import csv
import json
import os
import sys
import time
from math import cos, pi, log, exp
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from data.dataset import E2EDataset
from data.pdb_utils import VOCAB, AgAbComplex
from utils.logger import print_log
from models.dyMEAN.dyAb_model import dyAbModel

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, load_split_ids, split_dataset, DatasetView,
    EarlyStopping, ModelCheckpoint,
    setup_wandb, seed_everything, to_device, save_predictions,
    run_full_evaluation, FULL_METRIC_KEYS,
)

# CHIMERA evaluation metrics
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "..", ".."))
from evaluation.metrics import (
    aar as chimera_aar, kabsch_rmsd, tm_score as chimera_tm_score,
    count_liabilities,
)

METRIC_KEYS = ["ppl", "aar", "rmsd", "tm_score", "n_liabilities"]
CDR_TYPES = ["H1", "H2", "H3", "L1", "L2", "L3"]


class RobustE2EDataset(E2EDataset):
    """E2EDataset with broad error handling in preprocessing."""

    def preprocess(self, file_path, save_dir, num_entry_per_file):
        with open(file_path, "r") as fin:
            lines = fin.read().strip().split("\n")
        for line in tqdm(lines, desc="Preprocessing"):
            item = json.loads(line)
            try:
                cplx = AgAbComplex.from_pdb(
                    item["pdb_data_path"], item["heavy_chain"],
                    item["light_chain"], item["antigen_chains"])
            except Exception as e:
                print_log(f'parse {item["pdb"]} failed: {e}, skip', level="ERROR")
                continue
            self.data.append(cplx)
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
    parser = argparse.ArgumentParser(description="dyAb CHIMERA trainer")
    parser.add_argument("--split", type=str, default=None,
                        choices=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--cdr", type=str, default=None, nargs="+",
                        help="CDRs to generate: H1 H2 H3 L1 L2 L3, None for all")
    parser.add_argument("--paratope", type=str, default=None, nargs="+",
                        help="CDRs defining the paratope")
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
        data_dir=os.path.join(paths["trans_baselines"], "dyab"),
        output_dir=os.path.join(paths["results_root"], "dyab"),
        # Dataset
        split=ds.get("split", "epitope_group"),
        cdr=ds.get("cdr", None),
        paratope=ds.get("paratope", ["L3", "H3"]),
        # Model
        embed_dim=model.get("embed_dim", 64),
        hidden_size=model.get("hidden_size", 128),
        n_layers=model.get("n_layers", 3),
        iter_round=model.get("iter_round", 3),
        k_neighbors=model.get("k_neighbors", 9),
        bind_dist_cutoff=model.get("bind_dist_cutoff", 6.6),
        struct_only=model.get("struct_only", False),
        backbone_only=model.get("backbone_only", False),
        no_pred_edge_dist=model.get("no_pred_edge_dist", False),
        fix_channel_weights=model.get("fix_channel_weights", False),
        flexible=model.get("flexible", 0),
        module_type=model.get("module_type", 0),
        # Training
        seed=training.get("seed", 42),
        gpu=training.get("gpu", 0),
        lr=training.get("lr", 1e-3),
        final_lr=training.get("final_lr", 1e-4),
        batch_size=training.get("batch_size", 4),
        max_epoch=training.get("max_epoch", 10),
        grad_clip=training.get("grad_clip", 1.0),
        num_workers=training.get("num_workers", 4),
        # Callbacks
        patience=callbacks.get("early_stopping", {}).get("patience", 10),
        es_mode=callbacks.get("early_stopping", {}).get("mode", "min"),
        ckpt_mode=callbacks.get("checkpoint", {}).get("mode", "min"),
        # WandB
        use_wandb=wandb_cfg.get("enabled", False),
        wandb_project=wandb_cfg.get("project", "chimera_baselines"),
        # CDR numbering scheme
        numbering_scheme=cfg.get("numbering_scheme", "imgt"),
    )

    # CLI overrides
    if args.split is not None:
        ns.split = args.split
    if args.cdr is not None:
        ns.cdr = args.cdr
    if args.paratope is not None:
        ns.paratope = args.paratope
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


def _cdr_label(cdr):
    """Build a CDR label string from config cdr value."""
    if cdr is None:
        return "all"
    if isinstance(cdr, str):
        return cdr
    return "+".join(cdr)


def get_context_ratio(global_step, max_step):
    """Cosine-annealed context ratio from 0.9 -> 0."""
    return 0.5 * (cos(global_step / max_step * pi) + 1) * 0.9


def _val(x):
    """Extract scalar from tensor or number."""
    return x.item() if isinstance(x, torch.Tensor) else float(x)


def train_epoch(model, loader, optimizer, scheduler, device, grad_clip,
                global_step, max_step):
    model.train()
    total_loss, total_snll, total_aar = [], [], []
    total_struct, total_dock = [], []

    for batch in tqdm(loader, desc="Train"):
        batch = to_device(batch, device)
        batch["context_ratio"] = get_context_ratio(global_step, max_step)

        loss, seq_detail, struct_detail, dock_detail, pdev_detail = model(**batch)
        snll, aar = seq_detail
        struct_loss = struct_detail[0]
        dock_loss = dock_detail[0]

        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss.append(loss.item())
        total_snll.append(_val(snll))
        total_aar.append(_val(aar))
        total_struct.append(_val(struct_loss))
        total_dock.append(_val(dock_loss))
        global_step += 1

    return (np.mean(total_loss), np.mean(total_snll), np.mean(total_aar),
            np.mean(total_struct), np.mean(total_dock), global_step)


def valid_epoch(model, loader, device):
    model.eval()
    total_loss, total_snll, total_aar = [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val"):
            batch = to_device(batch, device)
            batch["context_ratio"] = 0

            loss, seq_detail, struct_detail, dock_detail, pdev_detail = model(**batch)
            snll, aar = seq_detail

            total_loss.append(loss.item())
            total_snll.append(_val(snll))
            total_aar.append(_val(aar))

    return np.mean(total_loss), np.mean(total_snll), np.mean(total_aar)


def get_cdr_ranges_in_concat(cplx, ag_len, hc_len, lc_len):
    """Get CDR ranges in the concatenated [AG, BOH, HC, BOL, LC] tensor format.

    Returns:
        dict: {cdr_type: (start_idx, end_idx)} in the concatenated tensor
    """
    cdr_ranges = {}
    for cdr_type in CDR_TYPES:
        cdr_pos = cplx.get_cdr_pos(cdr_type)
        if cdr_pos is None:
            continue
        start, end = cdr_pos
        if cdr_type.startswith("H"):
            # Heavy chain: offset = ag_len + 1 (BOH token)
            offset = ag_len + 1
        else:
            # Light chain: offset = ag_len + 1 (BOH) + hc_len + 1 (BOL)
            offset = ag_len + 1 + hc_len + 1
        cdr_ranges[cdr_type] = (offset + start, offset + end + 1)  # end+1 for slice
    return cdr_ranges


def run_inference_with_metrics(model, dataset, loader, device, cdr, idx_to_cid):
    """Run inference via model.sample() and compute CHIMERA metrics.

    For multi-CDR generation (cdr=None), computes per-CDR metrics.

    Returns:
        predictions_by_cdr: dict {cdr_type: list of prediction dicts}
        summary_by_cdr: dict {cdr_type: dict of mean metric values}
    """
    model.eval()
    is_multi_cdr = cdr is None

    if is_multi_cdr:
        predictions_by_cdr = {c: [] for c in CDR_TYPES}
        metrics_by_cdr = {c: {k: [] for k in METRIC_KEYS} for c in CDR_TYPES}
    else:
        cdr_lab = _cdr_label(cdr)
        predictions_by_cdr = {cdr_lab: []}
        metrics_by_cdr = {cdr_lab: {k: [] for k in METRIC_KEYS}}

    idx = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            batch = to_device(batch, device)
            lengths = batch["lengths"]
            batch_size = len(lengths)

            # Clone S and X before sample() -- model.sample() corrupts in-place
            orig_S = batch["S"].clone()
            orig_X = batch["X"].clone()

            sample_keys = {"X", "S", "cmask", "smask", "paratope_mask",
                          "residue_pos", "template", "lengths"}
            sample_batch = {k: v for k, v in batch.items() if k in sample_keys}
            gen_X, gen_S, nll = model.sample(**sample_batch)

            # nll is per-complex mean NLL: shape [batch_size]
            ppls = torch.exp(nll).cpu()

            # Split concatenated tensors by complex
            tensor_offset = 0
            for i in range(batch_size):
                n = lengths[i].item()
                dataset_idx = dataset.idx_mapping[idx]
                cid = idx_to_cid[dataset_idx]

                # Get the underlying AgAbComplex
                cplx = dataset.data[dataset_idx]

                seq_indices = gen_S[tensor_offset:tensor_offset + n]
                coords = gen_X[tensor_offset:tensor_offset + n]
                true_S = orig_S[tensor_offset:tensor_offset + n]
                true_X = orig_X[tensor_offset:tensor_offset + n]
                ppl_val = float(ppls[i])

                if is_multi_cdr:
                    # Get chain lengths to compute CDR offsets
                    ag_len = len(cplx.get_epitope())
                    hc_len = len(cplx.get_heavy_chain())
                    lc_len = len(cplx.get_light_chain()) if cplx.get_light_chain() else 0
                    cdr_ranges = get_cdr_ranges_in_concat(cplx, ag_len, hc_len, lc_len)

                    # Extract per-CDR predictions
                    for cdr_type, (cdr_start, cdr_end) in cdr_ranges.items():
                        if cdr_start >= n or cdr_end > n:
                            continue  # Skip if CDR is out of bounds

                        cdr_seq = "".join(
                            VOCAB.idx_to_symbol(seq_indices[j].item())
                            for j in range(cdr_start, cdr_end)
                        )
                        gt_seq = "".join(
                            VOCAB.idx_to_symbol(true_S[j].item())
                            for j in range(cdr_start, cdr_end)
                        )
                        pred_ca = coords[cdr_start:cdr_end, 1, :].cpu().numpy()
                        true_ca = true_X[cdr_start:cdr_end, 1, :].cpu().numpy()

                        # Per-CDR metrics
                        aar_val = chimera_aar(cdr_seq, gt_seq) if gt_seq else 0.0
                        if len(pred_ca) > 0:
                            rmsd_val = kabsch_rmsd(pred_ca, true_ca)
                            tm_val = chimera_tm_score(pred_ca, true_ca)
                        else:
                            rmsd_val = float("inf")
                            tm_val = 0.0
                        liab = count_liabilities(cdr_seq)

                        metrics_by_cdr[cdr_type]["ppl"].append(ppl_val)
                        metrics_by_cdr[cdr_type]["aar"].append(aar_val)
                        metrics_by_cdr[cdr_type]["rmsd"].append(rmsd_val)
                        metrics_by_cdr[cdr_type]["tm_score"].append(tm_val)
                        metrics_by_cdr[cdr_type]["n_liabilities"].append(liab)

                        pred = {
                            "complex_id": cid,
                            "cdr_type": cdr_type,
                            "pred_sequence": cdr_seq,
                            "true_sequence": gt_seq,
                            "pred_coords": pred_ca,
                            "true_coords": true_ca,
                            "ppl": ppl_val,
                        }
                        predictions_by_cdr[cdr_type].append(pred)
                else:
                    # Single CDR or specific CDR list
                    smask_i = batch["smask"][tensor_offset:tensor_offset + n]
                    cdr_indices = smask_i.nonzero(as_tuple=True)[0]

                    if len(cdr_indices) > 0:
                        cdr_seq = "".join(
                            VOCAB.idx_to_symbol(seq_indices[j].item())
                            for j in cdr_indices
                        )
                        gt_seq = "".join(
                            VOCAB.idx_to_symbol(true_S[j].item())
                            for j in cdr_indices
                        )
                        pred_ca = coords[cdr_indices, 1, :].cpu().numpy()
                        true_ca = true_X[cdr_indices, 1, :].cpu().numpy()
                    else:
                        cdr_seq, gt_seq = "", ""
                        pred_ca = np.zeros((0, 3))
                        true_ca = np.zeros((0, 3))

                    aar_val = chimera_aar(cdr_seq, gt_seq) if gt_seq else 0.0
                    if len(pred_ca) > 0:
                        rmsd_val = kabsch_rmsd(pred_ca, true_ca)
                        tm_val = chimera_tm_score(pred_ca, true_ca)
                    else:
                        rmsd_val = float("inf")
                        tm_val = 0.0
                    liab = count_liabilities(cdr_seq)

                    metrics_by_cdr[cdr_lab]["ppl"].append(ppl_val)
                    metrics_by_cdr[cdr_lab]["aar"].append(aar_val)
                    metrics_by_cdr[cdr_lab]["rmsd"].append(rmsd_val)
                    metrics_by_cdr[cdr_lab]["tm_score"].append(tm_val)
                    metrics_by_cdr[cdr_lab]["n_liabilities"].append(liab)

                    pred = {
                        "complex_id": cid,
                        "cdr_type": cdr_lab,
                        "pred_sequence": cdr_seq,
                        "true_sequence": gt_seq,
                        "pred_coords": pred_ca,
                        "true_coords": true_ca,
                        "ppl": ppl_val,
                    }
                    predictions_by_cdr[cdr_lab].append(pred)

                tensor_offset += n
                idx += 1

    # Compute summary per CDR
    summary_by_cdr = {}
    for cdr_type, metrics in metrics_by_cdr.items():
        if metrics["ppl"]:  # Only if we have data for this CDR
            summary_by_cdr[cdr_type] = {k: float(np.mean(v)) for k, v in metrics.items() if v}

    return predictions_by_cdr, summary_by_cdr


def build_csv_row(cdr_type, summary):
    """Build a CSV row dict with mean±std formatted values.

    Args:
        cdr_type: CDR label (H1, H2, etc.)
        summary: dict from run_full_evaluation with {metric: {"mean": x, "std": y}}
    """
    row = {"cdr_type": cdr_type}
    for k in FULL_METRIC_KEYS:
        if k in summary:
            row[k] = f"{summary[k]['mean']:.2f}\u00b1{summary[k]['std']:.2f}"
        else:
            row[k] = ""
    return row


def save_test_csv(summary_by_cdr):
    """Save test metrics as CSV (one row per CDR type) to baseline directory.

    Format matches chimera_evaluate.py output for consistency with chimera_train.sh.
    """
    csv_path = os.path.join(_SCRIPT_DIR, "test_metrics.csv")
    fieldnames = ["cdr_type"] + list(FULL_METRIC_KEYS)

    rows = []
    for cdr_type in CDR_TYPES:
        if cdr_type in summary_by_cdr:
            rows.append(build_csv_row(cdr_type, summary_by_cdr[cdr_type]))

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
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
    cdr_lab = _cdr_label(c.cdr)
    run_name = f"dyab_{cdr_lab}_{c.split}"
    save_dir = os.path.join(c.output_dir, run_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save resolved config
    config_dict = vars(c)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load pre-cached dataset
    data_dir = os.path.abspath(c.data_dir)
    jsonl_path = os.path.join(data_dir, "all.jsonl")
    print(f"Loading dyAb dataset from {data_dir}...")
    dataset = RobustE2EDataset(jsonl_path, cdr=c.cdr, paratope=c.paratope)
    print(f"Dataset: {dataset.num_entry} complexes")

    # Load idx_to_cid mapping
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

    # Model (dyAbModel, not dyMEANModel)
    model = dyAbModel(
        c.embed_dim, c.hidden_size, VOCAB.MAX_ATOM_NUMBER,
        VOCAB.get_num_amino_acid_type(), VOCAB.get_mask_idx(),
        k_neighbors=c.k_neighbors,
        bind_dist_cutoff=c.bind_dist_cutoff,
        n_layers=c.n_layers,
        iter_round=c.iter_round,
        struct_only=c.struct_only,
        backbone_only=c.backbone_only,
        fix_channel_weights=c.fix_channel_weights,
        pred_edge_dist=not c.no_pred_edge_dist,
        cdr_type=c.cdr,
        paratope=c.paratope,
        flexible=c.flexible,
        module_type=c.module_type,
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
            num_workers=c.num_workers, collate_fn=E2EDataset.collate_fn)
    else:
        # Loaders
        train_loader = DataLoader(
            train_set, batch_size=c.batch_size, shuffle=True,
            num_workers=c.num_workers, collate_fn=E2EDataset.collate_fn)
        valid_loader = DataLoader(
            valid_set, batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=E2EDataset.collate_fn)
        test_loader = DataLoader(
            test_set, batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=E2EDataset.collate_fn)

        # Optimizer / Scheduler (per-step exponential decay, same as dyAb original)
        optimizer = torch.optim.Adam(model.parameters(), lr=c.lr)
        max_step = c.max_epoch * len(train_loader)
        log_alpha = log(c.final_lr / c.lr) / max_step if max_step > 0 else 0
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda step: exp(log_alpha * (step + 1)))

        # Callbacks
        early_stop = EarlyStopping(patience=c.patience, mode=c.es_mode)
        ckpt = ModelCheckpoint(save_dir, mode=c.ckpt_mode)

        # WandB
        config_dict["max_step"] = max_step
        wandb_run = setup_wandb(c.wandb_project, run_name, config_dict, enabled=c.use_wandb)

        # Training loop
        global_step = 0
        for epoch in range(c.max_epoch):
            t0 = time.time()
            (train_loss, train_snll, train_aar,
             train_struct, train_dock, global_step) = train_epoch(
                model, train_loader, optimizer, scheduler, device,
                c.grad_clip, global_step, max_step)

            val_loss, val_snll, val_aar = valid_epoch(model, valid_loader, device)

            # Val inference with CHIMERA metrics (per-CDR)
            _, val_metrics_by_cdr = run_inference_with_metrics(
                model, valid_set, valid_loader, device, c.cdr, idx_to_cid)

            # Aggregate val metrics across CDRs for logging
            val_metrics = {}
            for metric in METRIC_KEYS:
                vals = [m[metric] for m in val_metrics_by_cdr.values() if metric in m]
                if vals:
                    val_metrics[metric] = float(np.mean(vals))

            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]["lr"]
            is_best = ckpt.save(model, optimizer, scheduler, epoch, val_loss)

            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"train_aar={train_aar:.4f} val_aar={val_aar:.4f} "
                  f"val_chimera_aar={val_metrics.get('aar', 0):.4f} "
                  f"val_rmsd={val_metrics.get('rmsd', 0):.4f} "
                  f"val_tm={val_metrics.get('tm_score', 0):.4f} "
                  f"lr={lr:.6f} {'*' if is_best else ''} [{elapsed:.0f}s]")

            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss, "val_loss": val_loss,
                "train_snll": train_snll, "val_snll": val_snll,
                "train_aar": train_aar, "val_aar": val_aar,
                "train_struct_loss": train_struct, "train_dock_loss": train_dock,
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
    predictions_by_cdr, test_metrics_by_cdr = run_inference_with_metrics(
        model, test_set, test_loader, device, c.cdr, idx_to_cid)

    # Save predictions per CDR
    pred_base_dir = os.path.join(save_dir, "predictions")
    total_saved = 0
    n_complexes_by_cdr = {}
    for cdr_type, preds in predictions_by_cdr.items():
        if preds:
            pred_dir = os.path.join(pred_base_dir, cdr_type)
            save_predictions(preds, pred_dir)
            print(f"Saved {len(preds)} predictions for {cdr_type} to {pred_dir}")
            total_saved += len(preds)
            n_complexes_by_cdr[cdr_type] = len(preds)

    # Run full CHIMERA evaluation for each CDR (all 12 metrics)
    print("\nRunning full CHIMERA evaluation...")
    full_summary_by_cdr = {}
    for cdr_type in predictions_by_cdr:
        if predictions_by_cdr[cdr_type]:
            pred_dir = os.path.join(pred_base_dir, cdr_type)
            _, summary, _ = run_full_evaluation(
                pred_dir, c.split, c.data_root, cdr_type_hint=cdr_type,
                numbering_scheme=c.numbering_scheme)
            full_summary_by_cdr[cdr_type] = summary

    # Save test metrics CSV (one row per CDR, all 12 metrics)
    save_test_csv(full_summary_by_cdr)

    # Print test summary per CDR
    print(f"\nTest results:")
    for cdr_type, metrics in sorted(full_summary_by_cdr.items()):
        n_preds = n_complexes_by_cdr.get(cdr_type, 0)
        print(f"\n  {cdr_type} ({n_preds} complexes):")
        for k, v in metrics.items():
            print(f"    {k}: {v['mean']:.4f} \u00b1 {v['std']:.4f}")

    # Log test metrics to wandb (per-CDR + aggregated)
    if wandb_run:
        # Per-CDR metrics
        for cdr_type, metrics in full_summary_by_cdr.items():
            for k, v in metrics.items():
                wandb_run.log({f"test_{cdr_type}_{k}": v["mean"]})
            wandb_run.log({f"test_{cdr_type}_n": n_complexes_by_cdr.get(cdr_type, 0)})

        # Aggregate metrics across CDRs
        agg_metrics = {}
        for metric in FULL_METRIC_KEYS:
            vals = [m[metric]["mean"] for m in full_summary_by_cdr.values() if metric in m]
            if vals:
                agg_metrics[metric] = float(np.mean(vals))
        wandb_run.log({f"test_avg_{k}": v for k, v in agg_metrics.items()})
        wandb_run.log({"test_total_n": total_saved})
        wandb_run.finish()


if __name__ == "__main__":
    main()
