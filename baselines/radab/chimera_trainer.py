"""CHIMERA trainer for RADAb baseline.

RADAb is a retrieval-augmented diffusion model (DiffAb variant) with ESM-2
embeddings and MSA transformer. It generates one CDR at a time (single-CDR
masking) because the retrieval system requires a specific CDR flag (1-6) to
look up reference sequences from FASTA files.

Key differences from vanilla DiffAb:
- Retrieval system: looks up similar CDR sequences from FASTA files
- ESM-2 650M embeddings in diffusion module (loaded automatically)
- MSA transformer refines predictions using retrieved sequences
- Extra batch keys: pdb_id, cdr_to_mask
- Model type: 'diffanti' (not 'diffab')
- AA ordering: alphabetical by 1-letter code (A=0, C=1, D=2, ...)

Usage:
    cd baselines/radab
    python chimera_trainer.py
    python chimera_trainer.py --split temporal
    python chimera_trainer.py --split epitope_group --test_only --checkpoint path/to/best.pt
"""

import argparse
import copy
import csv
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# RADAb's code lives in src/diffab/
sys.path.insert(0, os.path.join(_SCRIPT_DIR, "src"))

from diffab.models import get_model
from diffab.modules.common.geometry import reconstruct_backbone_partially
from diffab.modules.common.so3 import so3vec_to_rotation
from diffab.utils.data import PaddingCollate, apply_patch_to_tensor
from diffab.utils.train import sum_weighted_losses, recursive_to
from diffab.utils.misc import seed_all, inf_iterator
from diffab.utils.transforms import MaskSingleCDR, MaskMultipleCDRs, MergeChains, PatchAroundAnchor
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

# RADAb CDR enum -> CHIMERA CDR label
CDR_ENUM_TO_LABEL = {
    CDR.H1: "H1", CDR.H2: "H2", CDR.H3: "H3",
    CDR.L1: "L1", CDR.L2: "L2", CDR.L3: "L3",
}
ALL_CDR_LABELS = ["H1", "H2", "H3", "L1", "L2", "L3"]

# RADAb AA ordering: alphabetical by 1-letter code
AA_INDEX_TO_ONE = {
    0: 'A', 1: 'C', 2: 'D', 3: 'E', 4: 'F', 5: 'G', 6: 'H', 7: 'I',
    8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P', 13: 'Q', 14: 'R', 15: 'S',
    16: 'T', 17: 'V', 18: 'W', 19: 'Y', 20: 'X', 21: '-',
}


def aa_tensor_to_seq(aa_tensor):
    """Convert amino acid index tensor to sequence string."""
    return "".join(AA_INDEX_TO_ONE.get(a.item(), "X") for a in aa_tensor)


def patch_retrieval_paths(fasta_dir):
    """Monkey-patch RADAb's retrieval module to use split-specific FASTA files.

    RADAb hardcodes FASTA paths as './data/ref_seqs_final/...' in
    construct_seq_martix.py. We replace the retrieval functions to use
    our CHIMERA-specific per-split FASTA directory.
    """
    import diffab.utils.retrieval.construct_seq_martix as retrieval_mod

    fasta_map = {
        1: os.path.join(fasta_dir, "H_CDR1.fasta"),
        2: os.path.join(fasta_dir, "H_CDR2.fasta"),
        3: os.path.join(fasta_dir, "H_CDR3.fasta"),
        4: os.path.join(fasta_dir, "ref_sequences_chothia_CDR4.fasta"),
        5: os.path.join(fasta_dir, "ref_sequences_chothia_CDR5.fasta"),
        6: os.path.join(fasta_dir, "ref_sequences_chothia_CDR6.fasta"),
    }

    orig_find = retrieval_mod.find_protein_sequences
    orig_ressymb = retrieval_mod.ressymb_to_resindex
    from diffab.utils.retrieval.retrieve_utils import tensor_to_pdbid

    def _get_retrieved_1(pdb_id_tensor, length, CDR_flag):
        """Training retrieval: excludes self-reference to match test distribution.

        The original RADAb includes the native in training retrieval, but this
        causes the MSA transformer to learn to copy it, degrading test performance
        when the native is removed. We exclude the native during training too.
        """
        pdb_id = tensor_to_pdbid(pdb_id_tensor)
        fasta_file = fasta_map.get(CDR_flag)
        if fasta_file is None or not os.path.exists(fasta_file):
            return torch.full((15, length), 21).tolist()

        ref_sequences = orig_find(fasta_file, pdb_id)
        # Remove self-reference (first seq is native) -- same as _get_retrieved_2
        if ref_sequences:
            ref_sequences = [x for x in ref_sequences if x != ref_sequences[0]]
            if ref_sequences:
                ref_sequences = ref_sequences[1:]

        num_sequences = 15
        if not ref_sequences:
            return torch.full((num_sequences, length), 21).tolist()

        import random
        while len(ref_sequences) < num_sequences:
            ref_sequences.append(random.choice(ref_sequences))
        ref_sequences = ref_sequences[:num_sequences]

        ref_matrix_list = []
        for seq in ref_sequences:
            seq_tensor = torch.zeros(len(seq), dtype=torch.long)
            for i, aa in enumerate(seq):
                seq_tensor[i] = orig_ressymb.get(aa, 20)
            ref_matrix_list.append(seq_tensor)
        return ref_matrix_list

    def _get_retrieved_2(pdb_id_tensor, length, CDR_flag):
        """Inference retrieval: excludes self-reference to prevent leakage."""
        pdb_id = tensor_to_pdbid(pdb_id_tensor)
        fasta_file = fasta_map.get(CDR_flag)
        if fasta_file is None or not os.path.exists(fasta_file):
            return torch.full((15, length), 21).tolist()

        ref_sequences = orig_find(fasta_file, pdb_id)
        # Remove self-reference (first seq is native)
        if ref_sequences:
            ref_sequences = [x for x in ref_sequences if x != ref_sequences[0]]
            if ref_sequences:
                ref_sequences = ref_sequences[1:]

        num_sequences = 15
        if not ref_sequences:
            return torch.full((num_sequences, length), 21).tolist()

        import random
        while len(ref_sequences) < num_sequences:
            ref_sequences.append(random.choice(ref_sequences))
        ref_sequences = ref_sequences[:num_sequences]

        ref_matrix_list = []
        for seq in ref_sequences:
            seq_tensor = torch.zeros(len(seq), dtype=torch.long)
            for i, aa in enumerate(seq):
                seq_tensor[i] = orig_ressymb.get(aa, 20)
            ref_matrix_list.append(seq_tensor)
        return ref_matrix_list

    retrieval_mod.get_retrieved_sequences_class_test_CDRclass_1 = _get_retrieved_1
    retrieval_mod.get_retrieved_sequences_class_test_CDRclass_2 = _get_retrieved_2

    # Also patch the local references in diffab.models.diffab, which were bound
    # at import time via `from ... import` and won't see the module-level patch.
    import diffab.models.diffab as model_mod
    model_mod.get_retrieved_sequences_class_test_CDRclass_1 = _get_retrieved_1
    model_mod.get_retrieved_sequences_class_test_CDRclass_2 = _get_retrieved_2
    log.info("Patched retrieval to use FASTA dir: %s", fasta_dir)


class ChimeraLMDBDataset(Dataset):
    """Wraps RADAb's LMDB cache, filtered by CHIMERA split IDs."""

    MAP_SIZE = 32 * (1024 * 1024 * 1024)

    def __init__(self, lmdb_path, complex_ids, transform=None):
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
    parser = argparse.ArgumentParser(description="RADAb CHIMERA trainer")
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
        data_dir=os.path.join(paths["trans_baselines"], "radab"),
        output_dir=os.path.join(paths["results_root"], "radab"),
        # Dataset
        split=ds.get("split", "epitope_group"),
        # Model (passed as EasyDict to RADAb's get_model)
        model_cfg=model_cfg,
        # Training
        seed=training.get("seed", 2024),
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
    from torchvision.transforms import Compose
    patch_args = dict(initial_patch_size=c.patch_size, antigen_size=c.antigen_size)

    # Training: mask one random CDR per step (RADAb retrieval only supports single-CDR mode;
    # cdr_to_mask must be 1-6 for the FASTA retrieval to work, but MaskMultipleCDRs sets it to 7)
    train_tfm = Compose([
        MaskSingleCDR(selection=None, augmentation=True),
        MergeChains(),
        PatchAroundAnchor(**patch_args),
    ])

    # Validation: also single-CDR (random) to match training
    val_tfm = Compose([
        MaskSingleCDR(selection=None, augmentation=False),
        MergeChains(),
        PatchAroundAnchor(**patch_args),
    ])

    datasets = {}
    for subset in ("train", "val", "test"):
        ids = [cid for cid in split_ids.get(subset, []) if cid in available_ids]
        tfm = train_tfm if subset == "train" else val_tfm
        # Test set gets no transform -- handled per-complex in inference
        if subset == "test":
            tfm = None
        datasets[subset] = ChimeraLMDBDataset(lmdb_path, ids, transform=tfm)
        log.info("  %s: %d / %d complexes", subset, len(ids), len(split_ids.get(subset, [])))
    return datasets


def train_step(model, batch, optimizer, loss_weights, max_grad_norm):
    """Single training iteration."""
    model.train()
    optimizer.zero_grad()
    loss_dict = model(batch)
    loss = sum_weighted_losses(loss_dict, loss_weights)
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
    """Run test inference: generate one CDR at a time per complex.

    RADAb's retrieval system only works with single-CDR masking (cdr_to_mask
    must be 1-6 for the FASTA lookup). We iterate over each CDR type for each
    complex, masking and generating one CDR per forward pass.
    """
    from torchvision.transforms import Compose
    model.eval()
    collate_fn = PaddingCollate(eight=False)

    all_predictions = {cdr: [] for cdr in ALL_CDR_LABELS}

    for idx in tqdm(range(len(test_dataset)), desc="Test inference"):
        cid = test_dataset.ids_in_split[idx]
        try:
            structure = test_dataset.get_structure(cid)
        except Exception as e:
            log.warning("Failed to load %s: %s", cid, e)
            continue

        # Generate each CDR separately so retrieval gets the correct CDR flag
        for cdr_label in ALL_CDR_LABELS:
            mask_tfm = Compose([
                MaskSingleCDR(cdr_label, augmentation=False),
                MergeChains(),
            ])
            inference_tfm = Compose([
                PatchAroundAnchor(
                    initial_patch_size=c.patch_size,
                    antigen_size=c.antigen_size,
                ),
                RemoveNative(remove_structure=True, remove_sequence=True),
            ])

            try:
                data_merged = mask_tfm(copy.deepcopy(structure))
            except Exception as e:
                log.debug("Mask %s failed for %s: %s", cdr_label, cid, e)
                continue

            # Extract ground truth for this CDR
            cdr_flag = data_merged["cdr_flag"]
            cdr_enum = getattr(CDR, cdr_label)
            cdr_mask = (cdr_flag == int(cdr_enum))
            if cdr_mask.sum() == 0:
                continue
            gt_seq = aa_tensor_to_seq(data_merged["aa"][cdr_mask])
            gt_ca = data_merged["pos_heavyatom"][cdr_mask, BBHeavyAtom.CA].numpy()

            try:
                data_cropped = inference_tfm(copy.deepcopy(data_merged))
            except Exception as e:
                log.debug("Inference transform %s failed for %s: %s", cdr_label, cid, e)
                continue

            batch = collate_fn([data_cropped])
            batch = recursive_to(batch, device)

            with torch.no_grad():
                try:
                    traj = model.sample(batch, sample_opt={
                        "sample_structure": True,
                        "sample_sequence": True,
                    })
                except Exception as e:
                    log.warning("Sampling %s failed for %s: %s", cdr_label, cid, e)
                    continue

            # Reconstruct backbone
            aa_new = traj[0][2]
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

            pred_seq = aa_tensor_to_seq(aa_full[cdr_mask])
            pred_ca = pos_full[cdr_mask, BBHeavyAtom.CA].numpy()

            all_predictions[cdr_label].append({
                "complex_id": cid,
                "cdr_type": cdr_label,
                "pred_sequence": pred_seq,
                "true_sequence": gt_seq,
                "pred_coords": pred_ca,
                "true_coords": gt_ca,
                "ppl": 0.0,
            })

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

    # Patch retrieval module to use split-specific FASTA files
    fasta_dir = os.path.join(c.data_dir, "ref_seqs", c.split)
    if os.path.exists(fasta_dir):
        patch_retrieval_paths(fasta_dir)
    else:
        log.warning("Reference FASTA dir not found: %s (retrieval will return padding)", fasta_dir)

    run_name = f"radab_multicdrs_{c.split}"
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
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info("Model trainable parameters: %d", n_params)

    if c.test_only:
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
            filter(lambda p: p.requires_grad, model.parameters()),
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
            pin_memory=True,
        )
        val_loader = DataLoader(
            datasets["val"], batch_size=c.batch_size, shuffle=False,
            num_workers=c.num_workers, collate_fn=collate_fn,
            pin_memory=True,
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

        # Save final checkpoint
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
        for cdr_label, metrics in full_summary_by_cdr.items():
            for k, v in metrics.items():
                wandb_run.log({f"test_{cdr_label}_{k}": v["mean"]})
            wandb_run.log({f"test_{cdr_label}_n": n_complexes_by_cdr.get(cdr_label, 0)})
        agg_metrics = {}
        for metric in FULL_METRIC_KEYS:
            vals = [m[metric]["mean"] for m in full_summary_by_cdr.values() if metric in m]
            if vals:
                agg_metrics[metric] = float(np.mean(vals))
        wandb_run.log({f"test_avg_{k}": v for k, v in agg_metrics.items()})
        wandb_run.log({"test_n_predictions": total})
        wandb_run.finish()


if __name__ == "__main__":
    main()
