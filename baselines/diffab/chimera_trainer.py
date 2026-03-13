"""CHIMERA trainer for DiffAb baseline.

DiffAb is a multi-CDR diffusion model that generates all CDRs simultaneously.
We train once per split (not per CDR). At test time, all CDRs are generated
together and then decomposed into per-CDR prediction files for evaluation.

Usage:
    cd baselines/diffab
    python chimera_trainer.py
    python chimera_trainer.py --split temporal
    python chimera_trainer.py --split epitope_group --test_only --checkpoint path/to/best.pt
"""

import argparse
import copy
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPT_DIR)

from diffab.models import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.data import PaddingCollate, apply_patch_to_tensor
from diffab.utils.train import sum_weighted_losses, recursive_to
from diffab.utils.misc import seed_all, inf_iterator
from diffab.utils.transforms import MaskMultipleCDRs, MergeChains, PatchAroundAnchor
from diffab.utils.inference import RemoveNative
from diffab.utils.protein.constants import CDR, BBHeavyAtom

sys.path.insert(0, os.path.join(_SCRIPT_DIR, ".."))
from chimera_utils import (
    load_shared_config, load_split_ids,
    EarlyStopping, ModelCheckpoint,
    setup_wandb, save_predictions,
    run_full_evaluation, FULL_METRIC_KEYS,
)


log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# DiffAb CDR enum -> CHIMERA CDR label
CDR_ENUM_TO_LABEL = {
    CDR.H1: "H1", CDR.H2: "H2", CDR.H3: "H3",
    CDR.L1: "L1", CDR.L2: "L2", CDR.L3: "L3",
}
ALL_CDR_LABELS = ["H1", "H2", "H3", "L1", "L2", "L3"]

# Map from amino acid index to one-letter code
AA_INDEX_TO_ONE = {
    0: 'A', 1: 'R', 2: 'N', 3: 'D', 4: 'C', 5: 'Q', 6: 'E', 7: 'G',
    8: 'H', 9: 'I', 10: 'L', 11: 'K', 12: 'M', 13: 'F', 14: 'P', 15: 'S',
    16: 'T', 17: 'W', 18: 'Y', 19: 'V', 20: 'X', 21: '-',
}


def aa_tensor_to_seq(aa_tensor):
    """Convert amino acid index tensor to sequence string."""
    return "".join(AA_INDEX_TO_ONE.get(a.item(), "X") for a in aa_tensor)


class ChimeraLMDBDataset(Dataset):
    """Wraps DiffAb's LMDB cache, filtered by CHIMERA split IDs."""

    MAP_SIZE = 32 * (1024 * 1024 * 1024)

    def __init__(self, lmdb_path, complex_ids, transform=None):
        """
        Args:
            lmdb_path: path to structures.lmdb
            complex_ids: list of complex_id strings to include
            transform: callable applied to each structure dict
        """
        self.lmdb_path = lmdb_path
        self.ids_in_split = list(complex_ids)
        self.transform = transform
        self.db_conn = None

    def _connect_db(self):
        if self.db_conn is not None:
            return
        self.db_conn = lmdb.open(
            self.lmdb_path,
            map_size=self.MAP_SIZE,
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

    def get_structure(self, cid):
        self._connect_db()
        with self.db_conn.begin() as txn:
            raw = txn.get(cid.encode("utf-8"))
            if raw is None:
                raise KeyError(f"Complex {cid} not found in LMDB")
            return pickle.loads(raw)

    def __len__(self):
        return len(self.ids_in_split)

    def __getitem__(self, index):
        cid = self.ids_in_split[index]
        data = self.get_structure(cid)
        if self.transform is not None:
            data = self.transform(data)
        return data


def load_config():
    """Load baseline config.yaml merged with shared config."""
    config_path = os.path.join(_SCRIPT_DIR, "config.yaml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    shared = load_shared_config()
    return cfg, shared


def parse_args():
    parser = argparse.ArgumentParser(description="DiffAb CHIMERA trainer")
    parser.add_argument("--split", type=str, default=None,
                        choices=["epitope_group", "antigen_fold", "temporal"])
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max_iters", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument("--test_only", action="store_true",
                        help="Skip training, run test inference only")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint for test_only mode")
    return parser.parse_args()


def build_config(cfg, shared, args):
    """Build config namespace from YAML + shared config + CLI overrides."""
    ds = cfg.get("dataset", {})
    model_cfg = cfg.get("model", {})
    training = cfg.get("training", {})
    transforms = cfg.get("transforms", {})
    callbacks = cfg.get("callbacks", {})
    wandb_cfg = cfg.get("wandb", {})
    paths = shared["paths"]

    ns = SimpleNamespace(
        # Paths
        data_root=paths["data_root"],
        data_dir=os.path.join(paths["trans_baselines"], "diffab"),
        output_dir=os.path.join(paths["results_root"], "diffab"),
        # Dataset
        split=ds.get("split", "epitope_group"),
        # Model (passed as EasyDict to DiffAb's get_model)
        model_cfg=model_cfg,
        # Training
        seed=training.get("seed", 2022),
        gpu=training.get("gpu", 0),
        max_iters=training.get("max_iters", 200000),
        val_freq=training.get("val_freq", 1000),
        batch_size=training.get("batch_size", 16),
        max_grad_norm=training.get("max_grad_norm", 100.0),
        num_workers=training.get("num_workers", 4),
        loss_weights=training.get("loss_weights", {"rot": 1.0, "pos": 1.0, "seq": 1.0}),
        optimizer_cfg=training.get("optimizer", {}),
        scheduler_cfg=training.get("scheduler", {}),
        # Transforms
        patch_size=transforms.get("patch_size", 128),
        antigen_size=transforms.get("antigen_size", 128),
        # Callbacks
        patience=callbacks.get("early_stopping", {}).get("patience", 10),
        es_mode=callbacks.get("early_stopping", {}).get("mode", "min"),
        ckpt_mode=callbacks.get("checkpoint", {}).get("mode", "min"),
        # WandB
        use_wandb=wandb_cfg.get("enabled", False),
        wandb_project=wandb_cfg.get("project", "chimera_baselines"),
        # CDR numbering scheme
        numbering_scheme=cfg.get("numbering_scheme", "chothia"),
        # Test-only
        test_only=False,
        checkpoint=None,
    )

    # CLI overrides
    if args.split is not None:
        ns.split = args.split
    if args.gpu is not None:
        ns.gpu = args.gpu
    if args.seed is not None:
        ns.seed = args.seed
    if args.max_iters is not None:
        ns.max_iters = args.max_iters
    if args.batch_size is not None:
        ns.batch_size = args.batch_size
    if not args.no_wandb:
        ns.use_wandb = True
    if args.test_only:
        ns.test_only = True
    if args.checkpoint:
        ns.checkpoint = args.checkpoint

    return ns


def build_datasets(c, lmdb_path, split_ids, available_ids):
    """Build train/val/test datasets with appropriate transforms."""
    patch_args = dict(initial_patch_size=c.patch_size, antigen_size=c.antigen_size)

    # Training: random mask 1-6 CDRs + merge + patch
    from torchvision.transforms import Compose
    train_tfm = Compose([
        MaskMultipleCDRs(selection=None, augmentation=True),
        MergeChains(),
        PatchAroundAnchor(**patch_args),
    ])

    # Validation: mask all CDRs (fixed) + merge + patch
    val_tfm = Compose([
        MaskMultipleCDRs(
            selection=["H1", "H2", "H3", "L1", "L2", "L3"],
            augmentation=False,
        ),
        MergeChains(),
        PatchAroundAnchor(**patch_args),
    ])

    datasets = {}
    for subset in ("train", "val", "test"):
        ids = [cid for cid in split_ids.get(subset, []) if cid in available_ids]
        tfm = train_tfm if subset == "train" else val_tfm
        # Test set gets no transform here -- handled per-complex in inference
        if subset == "test":
            tfm = None
        datasets[subset] = ChimeraLMDBDataset(lmdb_path, ids, transform=tfm)
        log.info("  %s: %d / %d complexes", subset, len(ids), len(split_ids.get(subset, [])))
    return datasets


def train_step(model, batch, optimizer, loss_weights, max_grad_norm):
    """Single training iteration."""
    model.train()
    loss_dict = model(batch)
    loss = sum_weighted_losses(loss_dict, loss_weights)
    optimizer.zero_grad()
    loss.backward()
    grad_norm = clip_grad_norm_(model.parameters(), max_grad_norm)
    optimizer.step()
    return {k: v.item() for k, v in loss_dict.items()}, loss.item(), grad_norm


def validate(model, val_loader, device, loss_weights):
    """Run validation and return average loss."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for batch in val_loader:
            batch = recursive_to(batch, device)
            loss_dict = model(batch)
            loss = sum_weighted_losses(loss_dict, loss_weights)
            total_loss += loss.item()
            n_batches += 1
    return total_loss / max(n_batches, 1)


def run_test_inference(model, test_dataset, device, c):
    """Run test inference: generate all CDRs, decompose into per-CDR predictions.

    For each test complex:
    1. Load structure, mask all CDRs
    2. Extract ground truth per-CDR data from merged structure
    3. Apply PatchAroundAnchor + RemoveNative for model input
    4. model.sample() -> reconstruct_backbone_partially
    5. Decompose into per-CDR predictions

    Returns:
        dict {cdr_label: [pred_dict, ...]}
    """
    from torchvision.transforms import Compose
    model.eval()
    collate_fn = PaddingCollate(eight=False)

    all_predictions = {cdr: [] for cdr in ALL_CDR_LABELS}

    mask_tfm = Compose([
        MaskMultipleCDRs(
            selection=["H1", "H2", "H3", "L1", "L2", "L3"],
            augmentation=False,
        ),
        MergeChains(),
    ])

    inference_tfm = Compose([
        PatchAroundAnchor(
            initial_patch_size=c.patch_size,
            antigen_size=c.antigen_size,
        ),
        RemoveNative(remove_structure=True, remove_sequence=True),
    ])

    for idx in tqdm(range(len(test_dataset)), desc="Test inference"):
        cid = test_dataset.ids_in_split[idx]
        try:
            structure = test_dataset.get_structure(cid)
        except Exception as e:
            log.warning("Failed to load %s: %s", cid, e)
            continue

        # Apply masking + merge to get ground truth data
        try:
            data_merged = mask_tfm(copy.deepcopy(structure))
        except Exception as e:
            log.warning("Transform failed for %s: %s", cid, e)
            continue

        # Extract per-CDR ground truth from merged data
        cdr_flag = data_merged["cdr_flag"]
        gt_info = {}
        for cdr_enum, cdr_label in CDR_ENUM_TO_LABEL.items():
            cdr_mask = (cdr_flag == int(cdr_enum))
            if cdr_mask.sum() == 0:
                continue
            gt_seq = aa_tensor_to_seq(data_merged["aa"][cdr_mask])
            gt_ca = data_merged["pos_heavyatom"][cdr_mask, BBHeavyAtom.CA].numpy()
            gt_info[cdr_label] = {
                "mask": cdr_mask,
                "true_sequence": gt_seq,
                "true_coords": gt_ca,
            }

        if not gt_info:
            log.warning("No CDRs found for %s", cid)
            continue

        # Apply inference transforms for model input
        try:
            data_cropped = inference_tfm(copy.deepcopy(data_merged))
        except Exception as e:
            log.warning("Inference transform failed for %s: %s", cid, e)
            continue

        # Run model.sample()
        batch = collate_fn([data_cropped])
        batch = recursive_to(batch, device)

        with torch.no_grad():
            try:
                traj = model.sample(batch, sample_opt={
                    "sample_structure": True,
                    "sample_sequence": True,
                })
            except Exception as e:
                log.warning("Sampling failed for %s: %s", cid, e)
                continue

        # Reconstruct backbone
        aa_new = traj[0][2]  # Last step, amino acids
        pos_atom_new, mask_atom_new = reconstruct_backbone_partially(
            pos_ctx=batch["pos_heavyatom"],
            R_new=so3vec_to_rotation(traj[0][0]),
            t_new=traj[0][1],
            aa=aa_new,
            chain_nb=batch["chain_nb"],
            res_nb=batch["res_nb"],
            mask_atoms=batch["mask_heavyatom"],
            mask_recons=batch["generate_flag"],
        )

        # Apply patch back to full structure
        aa_full = apply_patch_to_tensor(
            data_merged["aa"], aa_new[0].cpu(), data_cropped["patch_idx"])
        pos_full = apply_patch_to_tensor(
            data_merged["pos_heavyatom"],
            pos_atom_new[0].cpu() + batch["origin"][0].view(1, 1, 3).cpu(),
            data_cropped["patch_idx"],
        )

        # Decompose into per-CDR predictions
        for cdr_label, gt in gt_info.items():
            cdr_mask = gt["mask"]
            pred_seq = aa_tensor_to_seq(aa_full[cdr_mask])
            pred_ca = pos_full[cdr_mask, BBHeavyAtom.CA].numpy()

            pred = {
                "complex_id": cid,
                "cdr_type": cdr_label,
                "pred_sequence": pred_seq,
                "true_sequence": gt["true_sequence"],
                "pred_coords": pred_ca,
                "true_coords": gt["true_coords"],
                "ppl": 0.0,  # Diffusion models don't produce PPL
            }
            all_predictions[cdr_label].append(pred)

    return all_predictions


def main():
    args = parse_args()
    cfg, shared = load_config()
    c = build_config(cfg, shared, args)
    seed_all(c.seed)

    device = torch.device(f"cuda:{c.gpu}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(c.gpu)
    log.info("Using device: %s", device)

    run_name = f"diffab_multicdrs_{c.split}"
    save_dir = Path(c.output_dir) / run_name
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save resolved config
    config_save = {k: v for k, v in vars(c).items()
                   if not k.startswith("_") and k != "model_cfg"}
    config_save["model_cfg"] = c.model_cfg
    with open(save_dir / "config.json", "w") as f:
        json.dump(config_save, f, indent=2, default=str)

    # Load LMDB and split IDs
    lmdb_path = os.path.join(c.data_dir, "processed", "structures.lmdb")
    idx_to_cid_path = os.path.join(c.data_dir, "idx_to_cid.json")
    with open(idx_to_cid_path) as f:
        idx_to_cid = json.load(f)
    available_ids = set(idx_to_cid)
    log.info("LMDB has %d structures", len(available_ids))

    split_ids = load_split_ids(c.split, c.data_root)
    log.info("Building datasets for split=%s...", c.split)
    datasets = build_datasets(c, lmdb_path, split_ids, available_ids)

    # Build model
    model_edict = EasyDict(c.model_cfg)
    model = get_model(model_edict).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)

    if c.test_only:
        # Load checkpoint and skip to test
        ckpt_path = c.checkpoint
        if ckpt_path is None:
            ckpt_path = str(save_dir / "checkpoints" / "best.pt")
        log.info("Loading checkpoint: %s", ckpt_path)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        if "model_state_dict" in ckpt:
            model.load_state_dict(ckpt["model_state_dict"])
        elif "model" in ckpt:
            model.load_state_dict(ckpt["model"])
    else:
        # Optimizer / Scheduler
        opt_cfg = c.optimizer_cfg
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=opt_cfg.get("lr", 1e-4),
            weight_decay=opt_cfg.get("weight_decay", 0.0),
            betas=(opt_cfg.get("beta1", 0.9), opt_cfg.get("beta2", 0.999)),
        )

        sched_cfg = c.scheduler_cfg
        if sched_cfg.get("type") == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                factor=sched_cfg.get("factor", 0.8),
                patience=sched_cfg.get("patience", 10),
                min_lr=sched_cfg.get("min_lr", 5e-6),
            )
        else:
            scheduler = None

        # DataLoaders
        collate_fn = PaddingCollate()
        train_loader = DataLoader(
            datasets["train"], batch_size=c.batch_size, shuffle=True,
            num_workers=c.num_workers, collate_fn=collate_fn,
        )
        val_loader = DataLoader(
            datasets["val"], batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=collate_fn,
        )
        train_iter = inf_iterator(train_loader)

        # Callbacks
        early_stop = EarlyStopping(patience=c.patience, mode=c.es_mode)
        ckpt_mgr = ModelCheckpoint(save_dir, mode=c.ckpt_mode)

        # WandB
        wandb_run = setup_wandb(c.wandb_project, run_name, config_save,
                                enabled=c.use_wandb)

        # Training loop (iteration-based)
        log.info("Starting training for %d iterations...", c.max_iters)
        loss_weights = EasyDict(c.loss_weights)

        for it in range(1, c.max_iters + 1):
            batch = recursive_to(next(train_iter), device)
            loss_parts, loss_val, grad_norm = train_step(
                model, batch, optimizer, loss_weights, c.max_grad_norm)

            if it % 100 == 0:
                lr = optimizer.param_groups[0]["lr"]
                log.info(
                    "Iter %06d | loss=%.4f rot=%.4f pos=%.4f seq=%.4f | "
                    "grad=%.2f lr=%.6f",
                    it, loss_val,
                    loss_parts.get("rot", 0), loss_parts.get("pos", 0),
                    loss_parts.get("seq", 0), grad_norm, lr,
                )

            if it % c.val_freq == 0:
                avg_val_loss = validate(model, val_loader, device, loss_weights)
                lr = optimizer.param_groups[0]["lr"]

                if scheduler is not None:
                    scheduler.step(avg_val_loss)

                is_best = ckpt_mgr.save(model, optimizer, scheduler, it, avg_val_loss)

                log.info(
                    "Iter %06d | val_loss=%.4f lr=%.6f %s",
                    it, avg_val_loss, lr, "*" if is_best else "",
                )

                log_dict = {
                    "step": it, "val_loss": avg_val_loss, "lr": lr,
                    "train_loss": loss_val,
                }
                log_dict.update({f"loss_{k}": v for k, v in loss_parts.items()})
                if wandb_run:
                    wandb_run.log(log_dict)

                if early_stop(avg_val_loss):
                    log.info("Early stopping at iteration %d", it)
                    break

        # Save final checkpoint (in case no validation was triggered)
        ckpt_mgr.save(model, optimizer, scheduler, it, float("inf"))

        # Load best model for test
        best_path = Path(save_dir) / "checkpoints" / "best.pt"
        if best_path.exists():
            log.info("Loading best model for test...")
            ckpt_mgr.load_best(model, device)
        else:
            log.info("No best checkpoint found, using current model state for test")

    # Test inference
    log.info("Running test inference...")
    all_predictions = run_test_inference(model, datasets["test"], device, c)

    # Save per-CDR predictions
    pred_base = save_dir / "predictions"
    n_complexes_by_cdr = {}
    for cdr_label in ALL_CDR_LABELS:
        preds = all_predictions[cdr_label]
        if not preds:
            log.info("No predictions for %s", cdr_label)
            continue
        cdr_dir = pred_base / cdr_label
        save_predictions(preds, cdr_dir)
        n_complexes_by_cdr[cdr_label] = len(preds)
        log.info("Saved %d predictions for %s", len(preds), cdr_label)

    # Run full CHIMERA evaluation for each CDR (all 12 metrics)
    log.info("Running full CHIMERA evaluation...")
    full_summary_by_cdr = {}
    for cdr_label in ALL_CDR_LABELS:
        if cdr_label in n_complexes_by_cdr:
            cdr_dir = pred_base / cdr_label
            _, summary, _ = run_full_evaluation(
                cdr_dir, c.split, c.data_root, cdr_type_hint=cdr_label,
                numbering_scheme=c.numbering_scheme)
            full_summary_by_cdr[cdr_label] = summary

    # Save test metrics CSV to baseline directory (all 12 metrics, same format as chimera_evaluate.py)
    import csv
    csv_path = Path(_SCRIPT_DIR) / "test_metrics.csv"
    fieldnames = ["cdr_type"] + list(FULL_METRIC_KEYS)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for cdr_label in ALL_CDR_LABELS:
            if cdr_label in full_summary_by_cdr:
                row = {"cdr_type": cdr_label}
                for k in FULL_METRIC_KEYS:
                    if k in full_summary_by_cdr[cdr_label]:
                        m = full_summary_by_cdr[cdr_label][k]
                        row[k] = f"{m['mean']:.2f}\u00b1{m['std']:.2f}"
                    else:
                        row[k] = ""
                writer.writerow(row)
    log.info("Saved test metrics CSV to %s", csv_path)

    # Print test summary per CDR
    print("\nTest results:")
    for cdr_label in ALL_CDR_LABELS:
        if cdr_label not in full_summary_by_cdr:
            continue
        n_preds = n_complexes_by_cdr.get(cdr_label, 0)
        print(f"\n  {cdr_label} ({n_preds} complexes):")
        for k, v in full_summary_by_cdr[cdr_label].items():
            print(f"    {k}: {v['mean']:.4f} \u00b1 {v['std']:.4f}")

    # Summary
    total = sum(len(v) for v in all_predictions.values())
    log.info("Test complete: %d total predictions across %d CDR types", total,
             sum(1 for v in all_predictions.values() if v))

    if not c.test_only and wandb_run:
        # Per-CDR metrics
        for cdr_label, metrics in full_summary_by_cdr.items():
            for k, v in metrics.items():
                wandb_run.log({f"test_{cdr_label}_{k}": v["mean"]})
            wandb_run.log({f"test_{cdr_label}_n": n_complexes_by_cdr.get(cdr_label, 0)})
        # Aggregate metrics
        agg_metrics = {}
        for metric in FULL_METRIC_KEYS:
            vals = [m[metric]["mean"] for m in full_summary_by_cdr.values() if metric in m]
            if vals:
                agg_metrics[metric] = float(np.mean(vals))
        wandb_run.log({f"test_avg_{k}": v for k, v in agg_metrics.items()})
        wandb_run.log({"test_n_predictions": total})
        wandb_run.finish()


if __name__ == "__main__":
    import lmdb  # noqa: ensure available
    main()
