#!/bin/bash
# noise_curve_l40s.slurm - noise robustness curve (utils/noise_curve.py) of one checkpoint on
# one CRCD L40S.
#
#   cd /ihome/kchen/evv13/multireflection
#   sbatch --export=ALL,CKPT=/ix1/kchen/evv/multireflection/runs/<name>/<name>.pth,MODEL_RES=256 \
#       cluster/noise_curve_l40s.slurm
#
# Every model runs off dark512: the SSIM stop test uses the 512 px frame, which is then
# area-downscaled to MODEL_RES for the model (utils/eval_batched.py --model-resolution), so
# models of different input sizes are compared on the same frames.
# Knobs: CKPT (required), MODEL_RES (512), LEVELS (utils/noise_curve.py default), SEEDS (0,1,2),
# STARTS_FILE (unset: grid), GRID_STEP (0.1), OUT_DIR (<checkpoint dir>/noise_curve), EXTRA_ARGS
# (appended, word-split).

#SBATCH --job-name=mfl-noise-curve
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
LEVELS="${LEVELS:-}"
SEEDS="${SEEDS:-0,1,2}"
STARTS_FILE="${STARTS_FILE:-}"
GRID_STEP="${GRID_STEP:-0.1}"
OUT_DIR="${OUT_DIR:-$(dirname "$CKPT")/noise_curve}"
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

STARTS_ARGS=()
[ -n "$STARTS_FILE" ] && STARTS_ARGS=(--starts-file "$STARTS_FILE") || STARTS_ARGS=(--grid-step "$GRID_STEP")
LEVELS_ARGS=()
[ -n "$LEVELS" ] && LEVELS_ARGS=(--levels "$LEVELS")
srun --cpus-per-task="$CPUS" "$VENV_PY" utils/noise_curve.py \
    --checkpoint "$CKPT" \
    --data-dir "$DATA_DIR" \
    --model-resolution "$MODEL_RES" \
    --perturb-seeds "$SEEDS" \
    --workers "$CPUS" \
    --out-dir "$OUT_DIR" \
    "${STARTS_ARGS[@]}" \
    "${LEVELS_ARGS[@]}" \
    $EXTRA_ARGS

command -v crc-job-stats >/dev/null && crc-job-stats "${SLURM_JOB_ID:-}" || true
