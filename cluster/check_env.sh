#!/bin/bash
# check_env.sh - build (or verify) the training environment on CRCD.
#
#   ssh crc
#   cd /ihome/kchen/evv13/multireflection
#   bash cluster/check_env.sh
#
# Run this once after cloning, and again whenever uv.lock changes - the job script does NOT
# sync, it just runs .venv/bin/python, so a stale venv would go unnoticed.
#
# CRCD permits user-level installs. The documented restriction is only against installing
# to the system ("Users do not have privileges to install Python packages to the system");
# conda envs, venvs and `pip install --user` are all explicitly supported, and uv is a
# single static binary in ~/.local/bin.

set -uo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-/ix1/kchen/evv/multireflection}"

cd "$PROJECT_DIR"

# The venv brings its own CPython and its own CUDA runtime, so no modules are needed.
# Purge anyway, so a module left over from an interactive session cannot leak in.
module purge

export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    echo "installing uv into ~/.local/bin (a single static binary, no system changes)"
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "uv     : $(uv --version)"

echo "syncing from uv.lock (~6 GB on first run, near-instant afterwards) ..."
uv sync || { echo "uv sync failed"; exit 1; }

PY="$PROJECT_DIR/.venv/bin/python"
echo "python : $PY ($($PY --version 2>&1))"
echo

"$PY" - <<'PYEOF'
import importlib.util as u, sys
# Exactly what train/train_resnet_direct.py imports. scipy / skimage / sklearn are NOT
# needed for training - they belong to utils/graph_eval.py and app/inference.py.
required, optional = ["torch", "numpy", "cv2", "tqdm"], ["wandb"]
missing = []
for name in required + optional:
    if u.find_spec(name) is None:
        print(f"  {name:10s} MISSING")
        if name in required:
            missing.append(name)
        continue
    print(f"  {name:10s} {getattr(__import__(name), '__version__', '?')}")
import torch
print()
print(f"  torch cuda build : {torch.version.cuda}")
print(f"  cuda available   : {torch.cuda.is_available()}  (False on the login node is expected)")
if missing:
    print(f"\nMISSING REQUIRED: {', '.join(missing)}")
    sys.exit(1)
print("\nall required imports satisfied")
PYEOF

mkdir -p logs "$DATA_ROOT/data" "$DATA_ROOT/runs"

echo
echo "next:"
echo "  1. upload the dataset:  rsync -avP dark512.tar.gz crc:$DATA_ROOT/data/"
echo "  2. log in to W&B once:  $PY -m wandb login"
echo "  3. submit:              sbatch cluster/train_l40s.slurm"
