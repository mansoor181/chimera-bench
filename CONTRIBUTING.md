# Contributing to CHIMERA-Bench

We welcome new methods, bug fixes, and improvements. This guide covers the two
most common contributions: **submitting a method to the leaderboard** and
**adding a new baseline**.

## Submitting a method to the leaderboard

You do not need to add your training code to this repo to appear on the
leaderboard. You only need to produce predictions in the standard format and
run the official evaluator.

### 1. Produce predictions

For each test complex, write one `.pt` file containing a dictionary:

| Key | Type | Required | Description |
|-----|------|:--------:|-------------|
| `complex_id` | `str` | yes | Must match a complex ID in the split (`{pdb}_{H}_{L}_{Ag}`) |
| `pred_sequence` | `str` | yes | Predicted heavy+light sequence (concatenated, one-letter) |
| `pred_coords` | `np.ndarray (N, 3)` | for structure metrics | Predicted CA coordinates, same length as `pred_sequence` |
| `pred_ab_full_coords` | `np.ndarray (N, 3)` | for interface metrics | Full antibody CA coordinates used for contacts/DockQ |

```python
import numpy as np, torch

torch.save(
    {
        "complex_id": "7x1m_C_D_M",
        "pred_sequence": heavy_seq + light_seq,
        "pred_coords": pred_ca,            # (N, 3)
        "pred_ab_full_coords": pred_ab_ca, # (N, 3)
    },
    f"predictions/{complex_id}.pt",
)
```

Write K files per complex (same `complex_id`) if your method is generative and
you want best-of-K and diversity metrics.

### 2. Run the official evaluator

```bash
export CHIMERA_DATA_ROOT=/path/to/chimera-bench-v1.0

python -m chimera_bench.evaluate \
    --predictions predictions/ \
    --split epitope_group \
    --cdr-type H3 \
    --output results/mymethod/epitope_group/H3/results.json
```

Repeat per split and CDR type. To aggregate all `results.json` files into a
leaderboard CSV:

```bash
python scripts/build_leaderboard.py --results results/ --output leaderboard.csv
```

### 3. Open a pull request

Include your `leaderboard.csv` rows (or the `results/` JSON), a one-line method
description, and a link to your code/paper. We re-run a subset to verify before
merging.

> Security: prediction `.pt` files are loaded with `weights_only=False` (needed
> for numpy arrays), which can execute arbitrary code via pickle. Only submit
> files you generated yourself, and never run someone else's prediction files
> without inspecting them.

## Adding a new baseline

Each baseline in `baselines/` follows a 5-file integration pattern:

| File | Purpose |
|------|---------|
| `config.yaml` | Hyperparameters; must set `numbering_scheme: imgt` or `chothia` |
| `preprocess.py` | Converts CHIMERA data into the method's native format |
| `chimera_trainer.py` | Training + test inference (`--split`, `--gpu`, `--test_only`) |
| `chimera_evaluate.py` | 12-metric evaluation wrapper |
| `chimera_train.sh` | End-to-end orchestration over all splits |

See an existing baseline (e.g. `baselines/diffab/`) and its README for the
reference structure.

## Development setup

```bash
pip install -e ".[test]"
pytest tests/ -q          # runs against the committed sample_data, no download needed
```

Please keep code minimal and dependency-light, follow the existing style, and
add a smoke test for new public functions.
