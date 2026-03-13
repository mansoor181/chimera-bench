#!/bin/bash
# Setup script for CHIMERA-Bench environment.
#
# Uses conda for base packages and bioconda tools, then pip for PyTorch,
# PyTorch Geometric, and protein ML tools (in that order to avoid conflicts).
#
# Usage:
#   bash setup_env.sh                  # default env name: chimera-bench
#   bash setup_env.sh my-env-name      # custom env name
#   CUDA_VERSION=11.8 bash setup_env.sh  # different CUDA version
set -e

ENV_NAME="${1:-chimera-bench}"
CUDA_VERSION="${CUDA_VERSION:-12.1}"

# Map CUDA version to PyTorch index URL
case "$CUDA_VERSION" in
    12.1) TORCH_INDEX="https://download.pytorch.org/whl/cu121" ;;
    11.8) TORCH_INDEX="https://download.pytorch.org/whl/cu118" ;;
    cpu)  TORCH_INDEX="https://download.pytorch.org/whl/cpu" ;;
    *)    echo "Unsupported CUDA_VERSION=$CUDA_VERSION (use 12.1, 11.8, or cpu)"; exit 1 ;;
esac

echo "=== Creating conda environment: $ENV_NAME ==="
conda env remove -n "$ENV_NAME" -y 2>/dev/null || true
conda env create -f environment.yml -n "$ENV_NAME"

eval "$(conda shell.bash hook)"
conda activate "$ENV_NAME"

echo "=== Installing PyTorch (CUDA $CUDA_VERSION) ==="
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 \
    --index-url "$TORCH_INDEX"

echo "=== Installing PyTorch Geometric ==="
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
    -f "https://data.pyg.org/whl/torch-2.2.0+cu${CUDA_VERSION//./}.html"

echo "=== Installing bioconda tools (ANARCI, muscle) ==="
conda install -y -n "$ENV_NAME" -c bioconda "muscle<5"
conda install -y -n "$ENV_NAME" -c bioconda anarci

echo "=== Installing protein tools ==="
pip install "fair-esm>=2.0.0"
pip install antiberty
pip install DockQ

echo "=== Installing ML utilities ==="
pip install wandb hydra-core>=1.3.0 easydict lmdb loguru einops

echo "=== Verifying installation ==="
python -c "
import torch
print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')
import numpy, scipy, pandas, Bio
print('Core packages OK')
import torch_geometric
print(f'PyTorch Geometric {torch_geometric.__version__}')
"

echo "=== Done ==="
echo "Activate with: conda activate $ENV_NAME"
