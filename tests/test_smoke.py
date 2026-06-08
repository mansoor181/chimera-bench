"""Smoke tests that run against the committed sample_data bundle.

These require no dataset download and no GPU. They verify that the public
loader API and the evaluation pipeline work end to end.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE = REPO_ROOT / "sample_data"

# Point the loaders at the committed sample before importing the package.
os.environ.setdefault("CHIMERA_DATA_ROOT", str(SAMPLE))

import chimera_bench as cb  # noqa: E402
from chimera_bench.evaluate import (  # noqa: E402
    _load_predictions, load_natives, evaluate_benchmark,
)


def test_load_split():
    sp = cb.load_split("epitope_group")
    assert set(sp) == {"train", "val", "test"}
    assert len(sp["test"]) > 0


def test_load_complex():
    cid = cb.load_split("epitope_group")["test"][0]
    feat = cb.load_complex(cid)
    assert feat["complex_id"] == cid
    assert isinstance(feat["heavy_sequence"], str)
    assert len(feat["heavy_ca_coords"]) == len(feat["heavy_sequence"])


def test_list_complexes():
    ids = cb.list_complexes("epitope_group", "test")
    assert ids == cb.load_split("epitope_group")["test"]


def test_dataset():
    ds = cb.ChimeraDataset("epitope_group", "test")
    assert len(ds) > 0
    assert "complex_id" in ds[0]


def test_cdr_indices():
    feat = cb.load_complex(cb.load_split("epitope_group")["test"][0])
    idx = cb.cdr_indices(feat, "H3", "imgt")
    assert len(idx) > 0


def test_missing_data_root_raises(monkeypatch):
    monkeypatch.delenv("CHIMERA_DATA_ROOT", raising=False)
    with pytest.raises(RuntimeError):
        cb.get_data_root()


def test_evaluation_end_to_end(tmp_path):
    """Perfect predictions should score AAR=1 and RMSD=0."""
    sp = cb.load_split("epitope_group")
    preds = tmp_path / "preds"
    preds.mkdir()
    for cid in sp["test"]:
        feat = cb.load_complex(cid)
        ab = np.concatenate([
            np.asarray(feat["heavy_ca_coords"]),
            np.asarray(feat["light_ca_coords"]),
        ], axis=0)
        seq = feat["heavy_sequence"] + feat["light_sequence"]
        torch.save({
            "complex_id": cid, "pred_sequence": seq,
            "pred_coords": ab, "pred_ab_full_coords": ab,
        }, preds / f"{cid}.pt")

    designs = _load_predictions(preds)
    natives = load_natives("epitope_group", subset="test", cdr_type="H3")
    out = evaluate_benchmark(designs, natives)

    assert len(out["per_complex"]) == len(sp["test"])
    assert out["summary"]["best_aar"]["mean"] == pytest.approx(1.0)
    assert out["summary"]["best_rmsd"]["mean"] == pytest.approx(0.0, abs=1e-6)


def test_all_twelve_metrics_present(tmp_path):
    cid = cb.load_split("epitope_group")["test"][0]
    feat = cb.load_complex(cid)
    ab = np.concatenate([
        np.asarray(feat["heavy_ca_coords"]),
        np.asarray(feat["light_ca_coords"]),
    ], axis=0)
    from chimera_bench.evaluate import DesignResult
    natives = load_natives("epitope_group", subset="test", cdr_type="H3")
    design = DesignResult(
        complex_id=cid,
        pred_sequence=feat["heavy_sequence"] + feat["light_sequence"],
        pred_coords=ab, pred_ab_full_coords=ab,
    )
    res = cb.evaluate_single(design, natives[cid])
    for key in ["aar", "caar", "n_liabilities", "rmsd", "tm_score",
                "fnat", "irmsd", "dockq", "epitope_f1",
                "chimera_s", "chimera_b"]:
        assert key in res, f"missing metric: {key}"
