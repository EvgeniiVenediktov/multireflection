#!/bin/bash
# eval_sweep_l40s.slurm - robustness sweep (utils/eval_sweep.py) of one checkpoint on one
# CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch --export=ALL,CKPT=/ix1/kchen/evv/multireflection/runs/<name>/<name>.pth,MODEL_RES=256 \
#       cluster/eval_sweep_l40s.slurm
#
# Every model runs off dark512: the SSIM stop test and the perturbations use the 512 px
# frame, which is then area-downscaled to MODEL_RES for the model (eval_batched.py
# --model-resolution), so models of different input sizes see the same frames.
# Knobs: CKPT (required), MODEL_RES (512), OUT_DIR (<checkpoint dir>/sweep), GRID_STEP (0.1),
# CONDITIONS (comma-separated subset, default all), EXTRA_ARGS (appended, word-split).
# Perturbed conditions run for perturbation seeds 0,1,2 (eval_sweep.py default); for one
# seed pass EXTRA_ARGS=--perturb-seeds=0.

#SBATCH --job-name=mfl-sweep
#SBATCH --cluster=gpu
#SBATCH --partition=l40s
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ihome/kchen/evv13/multireflection}"
DATA_ROOT="${DATA_ROOT:-/ix1/kchen/evv/multireflection}"
DATA_TAR="$DATA_ROOT/data/dark512.tar.gz"
VENV_PY="${VENV_PY:-$PROJECT_DIR/.venv/bin/python}"
CKPT="${CKPT:?set CKPT to the absolute checkpoint path}"
MODEL_RES="${MODEL_RES:-512}"
OUT_DIR="${OUT_DIR:-$(dirname "$CKPT")/sweep}"
GRID_STEP="${GRID_STEP:-0.1}"
CONDITIONS="${CONDITIONS:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
CPUS="${SLURM_CPUS_PER_TASK:-16}"

module purge

cd "$PROJECT_DIR"
[ -x "$VENV_PY" ] || { echo "no venv at $VENV_PY - run: bash cluster/check_env.sh"; exit 1; }
[ -f "$CKPT" ] || { echo "no checkpoint at $CKPT"; exit 1; }

echo "=================================================================="
echo "job         : ${SLURM_JOB_ID:-none} on $(hostname)"
echo "started     : $(date)"
echo "checkpoint  : $CKPT (input $MODEL_RES px)"
echo "output      : $OUT_DIR"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "=================================================================="

# Stage the image bank on node-local NVMe, as in train_l40s.slurm.
STAGE="${SLURM_SCRATCH:-/tmp/$USER-${SLURM_JOB_ID:-manual}}"
mkdir -p "$STAGE"
t0=$(date +%s)
cp "$DATA_TAR" "$STAGE/"
tar -xzf "$STAGE/$(basename "$DATA_TAR")" -C "$STAGE"
rm -f "$STAGE/$(basename "$DATA_TAR")"
DATA_DIR="$STAGE/dark512"
echo "staged $(ls "$DATA_DIR" | wc -l) files in $(( $(date +%s) - t0 ))s"

COND_ARGS=()
[ -n "$CONDITIONS" ] && COND_ARGS=(--conditions "$CONDITIONS")
srun --cpus-per-task="$CPUS" "$VENV_PY" utils/eval_sweep.py \
    --checkpoint "$CKPT" \
    --data-dir "$DATA_DIR" \
    --model-resolution "$MODEL_RES" \
    --grid-step "$GRID_STEP" \
    --workers "$CPUS" \
    --out-dir "$OUT_DIR" \
    "${COND_ARGS[@]}" \
    $EXTRA_ARGS

command -v crc-job-stats >/dev/null && crc-job-stats "${SLURM_JOB_ID:-}" || true
